# src/optiverse/core/adapters/csv_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import shutil
import sys
import uuid
import csv
import os
import time
import stat
import atexit
import threading
from typing import Iterable, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class CsvWrapperConfig:
    """
    Configuration for the CSV + wrapper-based simulation adapter.

    Parameters
    ----------
    model_dir :
        Path to the model directory that contains the *wrapper file* (e.g., "wflow_Tana").
        This is typically a directory with model inputs, scripts, and your wrapper module.
    wrapper_file :
        Python file (inside model_dir) that defines:
            def search_and_apply_variables(model_folder: str) -> None
        The adapter imports this module dynamically from a per-evaluation working copy.
    variables_csv :
        Path (absolute or relative) to the variable declaration CSV with columns:
            Name, Upper_bound, Lower_bound
        If relative, it is first resolved relative to `config_dir` (if given),
        falling back to `model_dir`.
    objectives_csv :
        Path (absolute or relative) to the objective declaration CSV with columns:
            Name, Objective   (Objective ∈ {minimize, minimise, min, maximize, maximise, max})
        Resolution logic is identical to variables_csv.
    config_dir :
        Optional directory that points to the folder containing the JSON config file.
        When provided, relative CSV paths are resolved *first* against this directory,
        then against `model_dir`. This makes runs resilient to process CWD.
    work_root :
        Optional directory under which all per-evaluation working copies are created.
        If not provided, working copies are created under `model_dir.parent`.
    keep_work_on_error :
        If True, the working directory is preserved on exceptions for debugging.
        Otherwise, temp directories are deleted robustly with retries.
    """
    model_dir: Path
    wrapper_file: str = "wrapper_wflow.py"
    variables_csv: str = "variable_declaration.csv"
    objectives_csv: str = "objective_declaration.csv"
    config_dir: Path | None = None
    work_root: Path | None = None
    keep_work_on_error: bool = False


# --------------------------------------------------------------------------------------
# Robust deletion utilities (module-level): read-only bit clearing, retries, reaper
# --------------------------------------------------------------------------------------

def _handle_remove_readonly(func, p, exc_info):
    """Make path writable then retry the removal function."""
    try:
        os.chmod(p, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(p)
    except Exception:
        pass


def _safe_rmtree(path: Path, retries: int = 5, base_delay: float = 0.5) -> bool:
    """
    Robustly delete a directory tree on Windows/Unix.
    - Clears read-only bits via onerror.
    - Retries with exponential backoff.
    Returns True if the directory is gone (or never existed), False otherwise.
    """
    try:
        if not path.exists():
            return True
    except Exception:
        # If existence check races, try to delete anyway
        pass

    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
            return True
        except Exception:
            time.sleep(base_delay * (2 ** i))

    # Last attempt: ignore errors
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

    try:
        return not path.exists()
    except Exception:
        # If we cannot stat, assume failure and let reaper try again
        return False


# Background reaper that keeps trying to delete stubborn folders.
_REAP_QUEUE: set[Path] = set()
_REAPER_LOCK = threading.Lock()
_REAPER_STARTED = False


def _start_reaper_thread():
    global _REAPER_STARTED
    if _REAPER_STARTED:
        return
    _REAPER_STARTED = True

    def _reaper():
        # Tries continually with gentle sleeps; keeps process light.
        while True:
            with _REAPER_LOCK:
                tasks = list(_REAP_QUEUE)
                _REAP_QUEUE.clear()
            # Attempt deletes now
            for path in tasks:
                _safe_rmtree(path)
            # Sleep a bit before next scan
            time.sleep(1.0)

    t = threading.Thread(target=_reaper, name="optiverse-reaper", daemon=True)
    t.start()


def _reap_later(path: Path):
    """Queue a path for deletion by the background reaper (and at exit)."""
    with _REAPER_LOCK:
        _REAP_QUEUE.add(path)
    _start_reaper_thread()


@atexit.register
def _atexit_reap():
    """Last-chance cleanup at interpreter shutdown."""
    with _REAPER_LOCK:
        tasks = list(_REAP_QUEUE)
        _REAP_QUEUE.clear()
    for p in tasks:
        _safe_rmtree(p)


# --------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------

class CsvWrapperAdapter:
    """
    Simulation adapter that preserves your CSV + wrapper workflow.

    Per-evaluation flow (x -> f(x), minimization):
      1) Copy `model_dir` to a unique working directory (sandbox).
      2) Write `variable_values.csv` in the working directory from the decision vector x.
      3) Import <workdir>/<wrapper_file> and call `search_and_apply_variables(workdir)`.
      4) Read <workdir>/objective_values.csv, order objectives per declaration,
         and flip sign for 'maximize' objectives so the optimizer sees pure minimization.
      5) Delete the working directory (robust retries on Windows).

    This design mirrors your previous approach while removing global state and making
    path handling robust across different invocation contexts.
    """

    # --------- Public API expected by the optimizer ---------

    def __init__(self, cfg: CsvWrapperConfig):
        self.cfg = cfg
        self._load_declarations()

    def n_variables(self) -> int:
        return len(self._variables)

    def n_objectives(self) -> int:
        return len(self._objectives)

    def bounds(self) -> list[tuple[float, float]]:
        # Note: declaration stores (name, ub, lb) -> convert to (lb, ub) for consistency
        return [(float(lb), float(ub)) for (_, ub, lb) in self._variables]

    # Convenience accessors (useful for labeling CSV outputs in the CLI)
    def variable_names(self) -> list[str]:
        return [name for (name, _, _) in self._variables]

    def objective_names(self) -> list[str]:
        return [name for (name, _) in self._objectives]

    def run(self, x: np.ndarray, context: Dict[str, Any] | None = None) -> np.ndarray:
        """
        Execute one simulation evaluation.

        Parameters
        ----------
        x :
            Decision vector (len == n_variables).
        context :
            Optional evaluation context (unused here but kept for parity).

        Returns
        -------
        np.ndarray
            Objective vector (minimization).
        """
        work = self._make_workdir()
        mod_name = f"wrapper_{uuid.uuid4().hex[:8]}"
        result: np.ndarray | None = None
        success = False

        try:
            # 1) Write variable_values.csv in the working directory
            vv = [(name, float(val)) for (name, _, _), val in zip(self._variables, x.tolist())]
            self._write_variable_values(work, vv)

            # 2) Import wrapper from the working copy and call the hook
            wrapper_path = work / self.cfg.wrapper_file
            mod = self._import_module_from_file(wrapper_path, module_name=mod_name)
            if not hasattr(mod, "search_and_apply_variables"):
                raise RuntimeError(
                    f"Wrapper '{self.cfg.wrapper_file}' must define "
                    f"'search_and_apply_variables(model_folder: str)'."
                )

            # IMPORTANT (wrapper best practice):
            # - Ensure wrapper blocks until external processes exit.
            # - Avoid leaving file handles open (redirect/close stdout/stderr).
            mod.search_and_apply_variables(str(work))

            # 3) Read objective_values.csv and assemble objective vector (minimization)
            obj_vals = self._read_objective_values(work)
            out: list[float] = []
            for name, minimize in self._objectives:
                if name not in obj_vals:
                    raise KeyError(
                        f"Objective '{name}' not found in objective_values.csv. "
                        f"Available keys: {list(obj_vals.keys())}"
                    )
                v = float(obj_vals[name])
                out.append(v if minimize else -v)  # flip sign for maximize -> minimize

            result = np.array(out, dtype=float)
            success = True

            # --- SUCCESS-PATH CLEANUP BEFORE RETURN ---
            if not _safe_rmtree(work):
                _reap_later(work)

            return result

        except Exception:
            # Optionally keep the work dir for debugging when something goes wrong
            if self.cfg.keep_work_on_error:
                print(f"[CsvWrapperAdapter] Keeping work dir for inspection: {work}")
            else:
                if not _safe_rmtree(work):
                    _reap_later(work)
            raise

        finally:
            # Drop our dynamic module reference (tidier GC)
            if mod_name in sys.modules:
                try:
                    del sys.modules[mod_name]
                except Exception:
                    pass

            # Extra safety: if we somehow didn't delete above:
            if (not success) and (not self.cfg.keep_work_on_error):
                # (On exception the except-block already tried; this is belt & braces.)
                if work.exists():
                    if not _safe_rmtree(work):
                        _reap_later(work)

    # --------- Internal helpers ---------

    # Path resolution that is resilient to CWD and supports external declarations
    def _resolve_csv(self, p: str | Path) -> Path:
        """
        Resolve a CSV path robustly.

        Resolution order:
          1) If `p` is absolute -> return as-is.
          2) If `cfg.config_dir` is provided and (config_dir / p) exists -> return that.
          3) Use (model_dir / p).

        Returns an absolute, normalized path (resolve()).
        """
        p = Path(p)
        if p.is_absolute():
            return p.resolve()

        # Try alongside the JSON config file, if supplied
        if self.cfg.config_dir:
            cand = (self.cfg.config_dir / p).resolve()
            if cand.exists():
                return cand

        # Fallback: relative to the model directory
        return (self.cfg.model_dir / p).resolve()

    def _load_declarations(self) -> None:
        """
        Load variable and objective declarations from CSV files, with strict validation.
        """
        vpath = self._resolve_csv(self.cfg.variables_csv)
        opath = self._resolve_csv(self.cfg.objectives_csv)

        if not vpath.exists():
            raise FileNotFoundError(
                f"Variable declaration CSV not found at:\n  {vpath}\n"
                f"Hint: If your CSV is beside the JSON file, set variables_csv to a path "
                f"relative to the JSON (e.g., '../variable_declaration.csv') "
                f"and ensure CsvWrapperConfig.config_dir is set by the CLI."
            )
        if not opath.exists():
            raise FileNotFoundError(
                f"Objective declaration CSV not found at:\n  {opath}\n"
                f"Hint: If your CSV is beside the JSON file, set objectives_csv to a path "
                f"relative to the JSON (e.g., '../objective_declaration.csv') "
                f"and ensure CsvWrapperConfig.config_dir is set by the CLI."
            )

        # ---- Variables CSV: Name, Upper_bound, Lower_bound ----
        dfv = pd.read_csv(vpath)
        required_vars_cols = {"Name", "Upper_bound", "Lower_bound"}
        missing = required_vars_cols - set(dfv.columns)
        if missing:
            raise ValueError(
                f"Variables CSV at {vpath} is missing columns: {sorted(missing)}; "
                f"expected {sorted(required_vars_cols)}"
            )

        variables: list[tuple[str, float, float]] = []
        for r in dfv.itertuples(index=False):
            name = str(getattr(r, "Name"))
            ub = float(getattr(r, "Upper_bound"))
            lb = float(getattr(r, "Lower_bound"))
            variables.append((name, ub, lb))
        if not variables:
            raise ValueError(f"No variables found in {vpath}")
        self._variables = variables

        # ---- Objectives CSV: Name, Objective ----
        dfo = pd.read_csv(opath)
        required_obj_cols = {"Name", "Objective"}
        missing = required_obj_cols - set(dfo.columns)
        if missing:
            raise ValueError(
                f"Objectives CSV at {opath} is missing columns: {sorted(missing)}; "
                f"expected {sorted(required_obj_cols)}"
            )

        objectives: list[tuple[str, bool]] = []
        for r in dfo.itertuples(index=False):
            name = str(getattr(r, "Name"))
            direction = str(getattr(r, "Objective")).strip().lower()
            if direction in {"minimize", "minimise", "min"}:
                minimize = True
            elif direction in {"maximize", "maximise", "max"}:
                minimize = False
            else:
                raise ValueError(
                    f"Objective '{name}' has invalid direction '{direction}'. "
                    f"Use one of: minimize/minimise/min or maximize/maximise/max."
                )
            objectives.append((name, minimize))
        if not objectives:
            raise ValueError(f"No objectives found in {opath}")
        self._objectives = objectives

    # ------------- make work dir under either parent(model_dir) or work_root -------------
    def _make_workdir(self) -> Path:
        base = self.cfg.model_dir.resolve()
        if not base.exists():
            raise FileNotFoundError(
                f"Model directory not found at:\n  {base}\n"
                f"Ensure your config 'model_dir' points to a real directory."
            )

        parent = (self.cfg.work_root or base.parent).resolve()
        parent.mkdir(parents=True, exist_ok=True)  # ensure root exists

        uid = uuid.uuid4().hex[:10]
        work = parent / f"{base.name}_{uid}"
        shutil.copytree(base, work)
        return work

    def _write_variable_values(self, work: Path, vv: Iterable[Tuple[str, float]]) -> None:
        """
        Write per-evaluation variable assignments to variable_values.csv in the working dir.
        """
        out = work / "variable_values.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Name", "Value"])
            for name, val in vv:
                w.writerow([name, val])

    def _read_objective_values(self, work: Path) -> Dict[str, float]:
        """
        Read objective_values.csv produced by the wrapper in the working dir.
        """
        path = work / "objective_values.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected '{path.name}' not found in workdir:\n  {work}\n"
                f"Your wrapper must write this file with columns [Name, Value]."
            )
        df = pd.read_csv(path)
        required = {"Name", "Value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"objective_values.csv is missing columns {sorted(missing)}; "
                f"expected {sorted(required)}"
            )
        return dict(zip(df["Name"], df["Value"]))

    @staticmethod
    def _import_module_from_file(path: Path, module_name: str):
        """
        Import a Python module from a specific file path (workdir/<wrapper_file>).
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Wrapper file '{path.name}' not found in working directory:\n  {path.parent}\n"
                f"Ensure your wrapper is placed under model_dir and the config's 'wrapper_file' "
                f"is correct."
            )
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod