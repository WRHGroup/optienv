# src/optiverse/cli/app.py
from __future__ import annotations

import csv as _csv
import json
import re
import time
import importlib.util
import shutil
import uuid
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

import numpy as np
import typer

from ..algos.nsga2 import NSGA2
from ..algos.nsga3 import NSGA3
from ..core.adapters.csv_wrapper import CsvWrapperAdapter, CsvWrapperConfig  # used only in main

from ..utils.pareto import flag_nondominated  # programmatic API (epsilon-nondomination)

app = typer.Typer(help="OptiVerse: evolutionary multi-objective optimization for simulators.")

# ---- Optional logging (safe fallback if helper missing) ----
try:
    from ..utils.logging import setup_logging as _setup_logging  # type: ignore

    def _log_setup() -> None:
        _setup_logging()
except Exception:  # pragma: no cover
    import logging

    def _log_setup() -> None:
        root = logging.getLogger()
        if not root.handlers:
            h = logging.StreamHandler(stream=sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            root.addHandler(h)
        root.setLevel(logging.INFO)

# ======================================================================================
# Epsilon-nondomination wrappers (used everywhere instead of utils.nondominated_mask)
# ======================================================================================

def _nd_mask_eps(F: np.ndarray, *, eps: float = 1e-9, maximize_cols: Optional[List[int]] = None) -> np.ndarray:
    """
    Return a boolean mask of epsilon-nondominated rows using pareto.py.
    - F is a 2D array of objectives in THEIR CURRENT sense.
    - eps is the epsilon box size per objective; 1e-9 ≈ strict Pareto.
    """
    # IMPORTANT: pass F directly (single table), not [F]
    masks = flag_nondominated(
        tables=F,                       # <-- single table (not a list)
        objectives=None,
        epsilons=[float(eps)] * F.shape[1],
        maximize=maximize_cols,
        maximize_all=False,
    )
    # pareto.flag_nondominated returns:
    #   - for a single table: list[bool] of length len(F)
    #   - for multiple tables: list[list[bool]] (one list per table)
    # Be defensive and unwrap if needed:
    if isinstance(masks, list) and masks and isinstance(masks[0], (list, np.ndarray)):
        masks = masks[0]
    return np.asarray(masks, dtype=bool)


def _eps_thin(F: np.ndarray, eps: float) -> np.ndarray:
    """
    Epsilon thinning on a (minimization) front using pareto.py (returns indices to keep).
    This keeps at most one solution per epsilon-box.
    """
    if F.size == 0:
        return np.empty((0,), dtype=int)
    mask = _nd_mask_eps(F, eps=eps, maximize_cols=None)  # minimize view
    return np.nonzero(mask)[0]


# ======================================================================================
# Lean worker helpers (unchanged)
# ======================================================================================

def _worker_handle_remove_readonly(func, p, exc_info):
    try:
        import os, stat
        os.chmod(p, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(p)
    except Exception:
        pass

def _worker_safe_rmtree(path: Path, retries: int = 5, base_delay: float = 0.5) -> None:
    import time as _t, shutil as _sh
    for i in range(retries):
        try:
            _sh.rmtree(path, onerror=_worker_handle_remove_readonly)
            return
        except Exception:
            _t.sleep(base_delay * (2 ** i))
    try:
        _sh.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _worker_import_module_from_file(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Wrapper file '{path.name}' not found in working directory: {path.parent}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _eval_worker_lean(x: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    model_dir = Path(spec["model_dir"]).resolve()
    wrapper_file = str(spec.get("wrapper_file") or "wrapper_wflow.py")
    variables: List[str] = list(spec["variables"])
    objectives: List[Dict[str, Any]] = list(spec["objectives"])
    work_root = Path(spec["work_root"]).resolve() if spec.get("work_root") else model_dir.parent
    keep_on_err = bool(spec.get("keep_work_on_error", False))

    # 1) make workdir
    uid = uuid.uuid4().hex[:10]
    work = (work_root / f"{model_dir.name}_{uid}").resolve()
    shutil.copytree(model_dir, work)

    mod_name = f"wrapper_{uuid.uuid4().hex[:8]}"

    try:
        # 2) write variable_values.csv
        vv_path = work / "variable_values.csv"
        with vv_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["Name", "Value"])
            for name, val in zip(variables, x.tolist()):
                w.writerow([name, float(val)])

        # 3) import wrapper and run
        mod = _worker_import_module_from_file(work / wrapper_file, mod_name)
        if not hasattr(mod, "search_and_apply_variables"):
            raise RuntimeError(
                f"Wrapper '{wrapper_file}' must define 'search_and_apply_variables(model_folder: str)'."
            )
        mod.search_and_apply_variables(str(work))

        # 4) read objective_values.csv
        ov_path = work / "objective_values.csv"
        if not ov_path.exists():
            raise FileNotFoundError(f"Expected '{ov_path.name}' not found in workdir: {work}")
        obj_map: Dict[str, float] = {}
        with ov_path.open("r", newline="") as f:
            r = _csv.DictReader(f)
            if "Name" not in r.fieldnames or "Value" not in r.fieldnames:
                raise ValueError("objective_values.csv must contain columns ['Name','Value']")
            for row in r:
                obj_map[str(row["Name"])] = float(row["Value"])

        out: List[float] = []
        for o in objectives:
            name = o["name"]
            minimize = bool(o["minimize"])
            if name not in obj_map:
                raise KeyError(f"Objective '{name}' not found in objective_values.csv. Available: {list(obj_map.keys())}")
            v = float(obj_map[name])
            out.append(v if minimize else -v)  # ensure minimization view for algorithms
        res = np.array(out, dtype=float)
        return res

    except Exception:
        if keep_on_err:
            print(f"[worker] Keeping work dir for inspection: {work}")
        else:
            _worker_safe_rmtree(work)
        raise

    finally:
        if mod_name in sys.modules:
            try:
                del sys.modules[mod_name]
            except Exception:
                pass
        if not keep_on_err and work.exists():
            _worker_safe_rmtree(work)

# ======================================================================================
# Checkpoint helpers (unchanged)
# ======================================================================================

def _ckpt_path_default(output: Path) -> Path:
    return output / "checkpoint.npz"

def _save_checkpoint(path: Path,
                     gen: int,
                     pop: np.ndarray,
                     fit: np.ndarray,
                     rng_state: tuple,
                     seed: Optional[int],
                     var_names: List[str],
                     obj_names: List[str],
                     bounds: List[Tuple[float, float]],
                     history_path: Path,
                     model_dir: Path) -> None:
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    meta = {
        "gen": int(gen),
        "seed": int(seed) if seed is not None else -1,
        "var_names": np.array(var_names, dtype=object),
        "obj_names": np.array(obj_names, dtype=object),
        "history_path": str(history_path),
        "model_dir": str(model_dir),
    }
    np.savez(
        path,
        pop=pop,
        fit=fit,
        rng_state=np.array(rng_state, dtype=object),
        lb=lb,
        ub=ub,
        meta=np.array(meta, dtype=object),
    )

def _load_checkpoint(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        pop = z["pop"]
        fit = z["fit"]
        rng_state = tuple(z["rng_state"].tolist())  # type: ignore
        lb = z["lb"].astype(float)
        ub = z["ub"].astype(float)
        meta = z["meta"].item()  # dict
    gen = int(meta["gen"])
    seed = int(meta["seed"])
    if seed == -1:
        seed = None
    var_names = [str(x) for x in meta["var_names"]]
    obj_names = [str(x) for x in meta["obj_names"]]
    history_path = Path(str(meta["history_path"]))
    model_dir = Path(str(meta["model_dir"]))
    bounds = list(zip(lb.tolist(), ub.tolist()))
    return {
        "gen": gen,
        "pop": pop,
        "fit": fit,
        "rng_state": rng_state,
        "seed": seed,
        "var_names": var_names,
        "obj_names": obj_names,
        "bounds": bounds,
        "history_path": history_path,
        "model_dir": model_dir,
    }

# ======================================================================================
# search with NSGA-II / NSGA-III, resume/checkpoint, progress, lean workers
# ======================================================================================

@app.command("search")
def search(
    config: Path = typer.Option(..., "--config", "-c", help="Path to JSON config"),
    output: Path = typer.Option("results/", "--output", "-o", help="Output directory"),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed (overrides JSON; suffixes outputs)"),
    algo: str = typer.Option("nsga2", "--algo", help="Algorithm: 'nsga2' or 'nsga3'"),
    # NSGA-III knobs:
    ref_parts: int = typer.Option(12, "--ref-parts", help="NSGA-III: Das–Dennis divisions per axis (ignored if --ref-dirs-csv)"),
    ref_dirs_csv: Optional[Path] = typer.Option(None, "--ref-dirs-csv", help="NSGA-III: CSV file of reference directions (rows=dirs, cols=objectives)"),
    # parallelism & infra:
    max_workers: int = typer.Option(1, "--max-workers", "-j", min=1, help="Parallel workers"),
    backend: str = typer.Option("process", "--backend", help="Parallel backend: 'process' or 'thread'"),
    keep_work_on_error: bool = typer.Option(False, help="Keep failed work dirs for inspection"),
    work_root: Optional[Path] = typer.Option(None, help="Optional temp root for working copies"),
    # outputs:
    label_columns: bool = typer.Option(True, help="Use objective/variable names as column headers"),
    save_final_csvs: bool = typer.Option(False, help="Also write final population/fitness/nd CSVs"),
    progress: bool = typer.Option(True, help="Print progress during evaluations"),
    # resume/checkpoint:
    resume_from: Optional[Path] = typer.Option(None, "--resume-from", help="Path to checkpoint .npz to resume from"),
    resume_latest: bool = typer.Option(False, "--resume-latest", help="Resume from results/checkpoint.npz"),
    checkpoint_every: int = typer.Option(1, "--checkpoint-every", min=0, help="Save checkpoint every N generations (0=off)"),
    checkpoint_path: Optional[Path] = typer.Option(None, "--checkpoint-path", help="Checkpoint file path (default: results/checkpoint.npz)"),
):
    """
    Run a multi-objective search (NSGA-II or NSGA-III) and append every generation to a SINGLE CSV (history_[seed].csv).
    Workers use a lean evaluator (no pandas) to avoid heavy imports in subprocesses.
    """
    _log_setup()

    cfg_json = json.loads(config.read_text())
    m = cfg_json["model"]
    a = cfg_json.get("algorithm", {})

    # Resolve config-relative paths
    cfg_dir = config.parent.resolve()
    model_dir = (cfg_dir / m["model_dir"]).resolve()

    # algorithm parameters (population/generations)
    pop_size = int(a.get("population_size", 40))
    generations = int(a.get("generations", 50))

    # Build a local adapter ONCE in main to parse declarations & bounds (pandas only in main)
    local_adapter = CsvWrapperAdapter(
        CsvWrapperConfig(
            model_dir=model_dir,
            wrapper_file=m.get("wrapper_file", "wrapper_wflow.py"),
            variables_csv=m.get("variables_csv", "variable_declaration.csv"),
            objectives_csv=m.get("objectives_csv", "objective_declaration.csv"),
            config_dir=cfg_dir,
            work_root=work_root,
            keep_work_on_error=keep_work_on_error,
        )
    )
    bounds = local_adapter.bounds()
    var_names = local_adapter.variable_names()
    obj_names = local_adapter.objective_names()
    n_obj = len(obj_names)
    n_var = len(var_names)

    # Objective sense for lean workers (read once using stdlib CSV)
    def _resolve_csv(rel: str | Path) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p.resolve()
        cand = (cfg_dir / p).resolve()
        return cand if cand.exists() else (model_dir / p).resolve()

    obj_decl = _resolve_csv(m.get("objectives_csv", "objective_declaration.csv"))
    sense_map: Dict[str, bool] = {}
    with obj_decl.open("r", newline="") as f:
        r = _csv.DictReader(f)
        if "Name" not in r.fieldnames or "Objective" not in r.fieldnames:
            raise ValueError("objective_declaration.csv must contain columns ['Name','Objective']")
        for row in r:
            direction = str(row["Objective"]).strip().lower()
            if direction in {"minimize", "minimise", "min"}:
                sense_map[str(row["Name"])] = True
            elif direction in {"maximize", "maximise", "max"}:
                sense_map[str(row["Name"])] = False
            else:
                raise ValueError(f"Invalid direction '{direction}' in objective_declaration.csv")
    objectives_spec: List[Dict[str, Any]] = []
    for name in obj_names:
        if name not in sense_map:
            raise KeyError(f"Objective '{name}' not found in objective_declaration.csv")
        objectives_spec.append({"name": name, "minimize": sense_map[name]})

    # Spec blob for workers (small; no pandas import)
    worker_spec: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "wrapper_file": m.get("wrapper_file", "wrapper_wflow.py"),
        "variables": var_names,
        "objectives": objectives_spec,
        "work_root": str(work_root) if work_root else None,
        "keep_work_on_error": keep_work_on_error,
    }

    # Headers
    pop_headers = var_names if label_columns else [f"x{i}" for i in range(n_var)]
    fit_headers = obj_names if label_columns else None

    # ------------------------- choose algorithm -------------------------
    algo_name = algo.lower().strip()
    if algo_name == "nsga2":
        algorithm = NSGA2(population_size=pop_size, seed=seed)
        pop = algorithm.initialize(n_var, bounds)
    elif algo_name == "nsga3":
        # prepare reference directions
        if ref_dirs_csv:
            ref_dirs = np.loadtxt(ref_dirs_csv, delimiter=",")
            if ref_dirs.ndim == 1:
                ref_dirs = ref_dirs[None, :]
        else:
            # generate Das–Dennis (default choice in literature & libraries)
            def _das_dennis_local(M: int, p: int) -> np.ndarray:
                parts = []
                def rec(k, remaining, acc):
                    if k == M - 1:
                        parts.append(acc + [remaining])
                        return
                    for v in range(remaining + 1):
                        rec(k + 1, remaining - v, acc + [v])
                rec(0, p, [])
                return np.array(parts, dtype=float) / float(p)
            ref_dirs = _das_dennis_local(n_obj, int(ref_parts))

        if pop_size != len(ref_dirs):
            typer.echo(f"[search] NSGA-III: population_size={pop_size}, reference_dirs={len(ref_dirs)}; "
                       f"it is common to set them equal for a 1:1 niche match. Proceeding anyway.")

        algorithm = NSGA3(population_size=pop_size, n_obj=n_obj,
                          ref_dirs=ref_dirs if ref_dirs_csv else None,
                          ref_parts=ref_parts if not ref_dirs_csv else 12,
                          seed=seed)
        pop = algorithm.initialize(n_var, bounds)
    else:
        raise typer.BadParameter("Unknown --algo. Use 'nsga2' or 'nsga3'.")

    # history path and seed suffix
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / f"history{seed_suffix}.csv"

    # ----- progress-aware evaluator -----
    def _format_hms(seconds: float) -> str:
        s = int(seconds)
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    ExecPool = ThreadPoolExecutor if backend.lower().startswith("thread") else ProcessPoolExecutor

    def eval_population(X: np.ndarray, *, gen: int, total_gens: int, phase: str = "Gen") -> np.ndarray:
        n = len(X)
        if progress:
            typer.echo(f"[search] {phase} {gen}/{total_gens} – evaluating {n} candidates (workers={max_workers}, backend={backend})...")
            start = time.time()

        if max_workers == 1:
            results = []
            last_print = start if progress else 0.0
            for i, x in enumerate(X, 1):
                results.append(_eval_worker_lean(x, worker_spec))
                if progress:
                    now = time.time()
                    if (now - last_print) >= 1.0 or i == n:
                        pct = int((i / n) * 100)
                        typer.echo(f"  completed: {i:2d}/{n} ({pct:3d}%)  elapsed: {_format_hms(now - start)}")
                        last_print = now
            F = np.vstack(results)
        else:
            results = [None] * n  # type: ignore
            with ExecPool(max_workers=max_workers) as ex:
                futs = {ex.submit(_eval_worker_lean, X[i], worker_spec): i for i in range(n)}
                completed = 0
                last_print = time.time() if progress else 0.0
                for fut in as_completed(futs):
                    idx = futs[fut]
                    results[idx] = fut.result()
                    completed += 1
                    if progress:
                        now = time.time()
                        if (now - last_print) >= 1.0 or completed == n:
                            pct = int((completed / n) * 100)
                            typer.echo(f"  completed: {completed:2d}/{n} ({pct:3d}%)  elapsed: {_format_hms(now - start)}")
                            last_print = now
            F = np.vstack(results)

        if progress:
            typer.echo(f"[search] {phase} {gen} done in {_format_hms(time.time() - start)}")
        return F

    # -------------------- Resume or fresh start --------------------
    ckpt_path = checkpoint_path or _ckpt_path_default(output)
    start_gen = 0
    if resume_from or resume_latest:
        path = resume_from or ckpt_path
        if not path.exists():
            raise typer.BadParameter(f"Checkpoint not found at: {path}")
        ckpt = _load_checkpoint(path)
        if seed is not None and ckpt["seed"] is not None and seed != ckpt["seed"]:
            typer.echo(f"[search] WARNING: overriding checkpoint seed ({ckpt['seed']}) with --seed {seed}")
        if list(ckpt["var_names"]) != var_names:
            raise typer.BadParameter("Checkpoint variable names do not match current configuration.")
        if list(ckpt["obj_names"]) != obj_names:
            raise typer.BadParameter("Checkpoint objective names do not match current configuration.")
        if list(map(tuple, ckpt["bounds"])) != list(map(tuple, bounds)):
            raise typer.BadParameter("Checkpoint bounds do not match current configuration.")
        # RNG state & arrays
        np.random.set_state(ckpt["rng_state"])
        pop = ckpt["pop"]
        fit = ckpt["fit"]
        start_gen = int(ckpt["gen"])
        typer.echo(f"[search] Resuming from generation {start_gen}, checkpoint: {path}")
        if not history_path.exists():
            with history_path.open("w", newline="") as f:
                w = _csv.writer(f)
                hdr_fit = fit_headers if fit_headers is not None else [f"f{i}" for i in range(fit.shape[1])]
                w.writerow(["generation", "index"] + hdr_fit + pop_headers + ["nd"])
        if fit_headers is None:
            fit_headers = [f"f{i}" for i in range(fit.shape[1])]
    else:
        # Evaluate Gen 0
        fit = eval_population(pop, gen=0, total_gens=generations, phase="Gen")
        if fit_headers is None:
            fit_headers = [f"f{i}" for i in range(fit.shape[1])]
        with history_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["generation", "index"] + fit_headers + pop_headers + ["nd"])
        def _append_generation(g: int, P: np.ndarray, F: np.ndarray) -> None:
            # write epsilon-nondominated mask (minimization view) to the history
            nd_mask = _nd_mask_eps(F, eps=1e-9, maximize_cols=None).astype(int).tolist()
            with history_path.open("a", newline="") as f:
                w = _csv.writer(f)
                for i in range(P.shape[0]):
                    w.writerow([g, i] + F[i].tolist() + P[i].tolist() + [nd_mask[i]])
        _append_generation(0, pop, fit)
        if checkpoint_every and checkpoint_every > 0:
            _save_checkpoint(
                ckpt_path, gen=0, pop=pop, fit=fit,
                rng_state=np.random.get_state(),
                seed=seed, var_names=var_names, obj_names=obj_names,
                bounds=bounds, history_path=history_path, model_dir=model_dir,
            )
        start_gen = 0

    # re-create algorithm after resume (stateless ask/tell interface)
    if algo_name == "nsga2":
        algorithm = NSGA2(population_size=pop_size, seed=None)
    else:
        if ref_dirs_csv:
            ref_dirs = np.loadtxt(ref_dirs_csv, delimiter=",")
            if ref_dirs.ndim == 1:
                ref_dirs = ref_dirs[None, :]
        else:
            # re-generate Das–Dennis be consistent with earlier
            def _das_dennis_local(M: int, p: int) -> np.ndarray:
                parts = []
                def rec(k, remaining, acc):
                    if k == M - 1:
                        parts.append(acc + [remaining])
                        return
                    for v in range(remaining + 1):
                        rec(k + 1, remaining - v, acc + [v])
                rec(0, p, [])
                return np.array(parts, dtype=float) / float(p)
            ref_dirs = _das_dennis_local(n_obj, int(ref_parts))
        algorithm = NSGA3(population_size=pop_size, n_obj=n_obj,
                          ref_dirs=ref_dirs if ref_dirs_csv else None,
                          ref_parts=ref_parts if not ref_dirs_csv else 12,
                          seed=None)

    # Helper to append generation rows
    def append_generation(g: int, P: np.ndarray, F: np.ndarray) -> None:
        nd_mask = _nd_mask_eps(F, eps=1e-9, maximize_cols=None).astype(int).tolist()
        with history_path.open("a", newline="") as f:
            w = _csv.writer(f)
            for i in range(P.shape[0]):
                w.writerow([g, i] + F[i].tolist() + P[i].tolist() + [nd_mask[i]])

    # -------------------- Iterate remaining generations --------------------
    for g in range(start_gen + 1, generations + 1):
        children = algorithm.ask(pop, fit, bounds)
        fit_children = eval_population(children, gen=g, total_gens=generations, phase="Gen")
        pop, fit = algorithm.tell(pop, fit, children, fit_children)
        append_generation(g, pop, fit)

        if checkpoint_every and (g % checkpoint_every == 0):
            _save_checkpoint(
                ckpt_path, gen=g, pop=pop, fit=fit,
                rng_state=np.random.get_state(),
                seed=seed, var_names=var_names, obj_names=obj_names,
                bounds=bounds, history_path=history_path, model_dir=model_dir,
            )

    # Optional finals
    if save_final_csvs:
        with (output / f"population{seed_suffix}.csv").open("w", newline="") as f_:
            w = _csv.writer(f_)
            w.writerow(pop_headers)
            w.writerows(pop.tolist())
        with (output / f"fitness{seed_suffix}.csv").open("w", newline="") as f_:
            w = _csv.writer(f_)
            w.writerow(fit_headers)
            w.writerows(fit.tolist())
        nd_mask = _nd_mask_eps(fit, eps=1e-9, maximize_cols=None)
        with (output / f"nd_mask{seed_suffix}.csv").open("w", newline="") as f_:
            w = _csv.writer(f_)
            w.writerow(["nd"])
            w.writerows([[int(b)] for b in nd_mask.tolist()])

    typer.echo(
        f"Saved/updated single-file history at {history_path} "
        f"(algo={algo_name}, seed={seed}, workers={max_workers}, backend={backend}, "
        f"checkpoint={'on' if checkpoint_every else 'off'})"
    )

# ======================================================================================
# pareto-front (seed provenance + --epsilon)
# ======================================================================================

@app.command("front")
def front(
    epsilon: Optional[float] = typer.Option(
        None, "--epsilon", min=0.0, help="ε-box thinning of the final Pareto front (optional)"
    ),
):
    """
    Global Pareto front across ALL seeds & generations under results/.
    Adds 'seed' column; optional --epsilon to thin the final frontier (epsilon-nondomination).
    Output: results/pareto_front_all.csv
    """
    import pandas as pd

    results_dir = _results_dir("results")
    files = _find_history_files(results_dir)

    kept_frames = []
    obj_cols: list[str] | None = None
    sense: np.ndarray | None = None

    for path in files:
        df = pd.read_csv(path)
        if obj_cols is None:
            obj_cols = _detect_objective_columns(df.columns, results_dir)
            sense = _detect_objective_sense(results_dir, obj_cols)
        assert sense is not None

        F = df[obj_cols].to_numpy(dtype=float) 
        nd = _nd_mask_eps(F, eps=1e-9, maximize_cols=None)
        kept = df.loc[nd].copy()

        m_seed = re.search(r"seed(\d+)", path.name)
        seed_label = int(m_seed.group(1)) if m_seed else path.stem
        kept["seed"] = seed_label

        kept_frames.append(kept)

    union = pd.concat(kept_frames, ignore_index=True)
    F_union = union[obj_cols].to_numpy(dtype=float)  # type: ignore[arg-type]
    nd_global = _nd_mask_eps(F_union, eps=1e-9, maximize_cols=None)
    idx_final = np.where(nd_global)[0]

    if epsilon is not None and epsilon > 0.0:
        # epsilon-thin the global front (optional)
        F_front = F_union[idx_final]
        idx_keep_in_front = _eps_thin(F_front, eps=float(epsilon))
        idx_final = idx_final[idx_keep_in_front]

    out = results_dir / "pareto_front_all.csv"
    union.iloc[idx_final].to_csv(out, index=False)
    typer.echo(f"Global Pareto front saved to: {out}")

# ======================================================================================
# hypervolume (normalized, WIDE) — ALWAYS via pymoo; optional cumulative (monotone)
# ======================================================================================
@app.command("hypervolume")
def hypervolume(
    epsilon: Optional[float] = typer.Option(
        None, "--epsilon", min=0.0,
        help="ε-thinning before HV (applied PER-GENERATION ONLY; ignored in --cumulative to preserve monotonicity)."
    ),
    cumulative: bool = typer.Option(
        False, "--cumulative/--per-generation",
        help="Cumulative ND union HV (monotone) or per-generation ND HV (default)."
    ),
    normalize_scope: str = typer.Option(
        "global", "--normalize-scope",
        help="Normalization scope: 'global' (all seeds/files, default) or 'per-seed'.",
        case_sensitive=False
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Write a debug CSV for the first seed (per-gen vs cumulative) and dump a diagnostic CSV if monotonicity would be violated."
    ),
    progress: bool = typer.Option(
        True, "--progress/--no-progress",
        help="Print per-seed and per-generation progress while computing HV."
    ),
    print_every: int = typer.Option(
        1, "--print-every", min=1,
        help="Print progress every N generations (default: 1)."
    ),
):
    """
    Normalized hypervolume per generation (wide CSV; one column per seed), **always via pymoo**.
    - Default (per-generation): HV on each generation's (epsilon-)nondominated set (may dip).
    - --cumulative: HV on the cumulative nondominated union up to g (monotone).
    Output: results/hypervolume.csv
    """
    import pandas as pd
    import time as _t

    # Require pymoo (fail fast)
    try:
        from pymoo.indicators.hv import HV  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise typer.BadParameter(
            "Hypervolume requires pymoo. Install with: pip install pymoo"
        ) from exc

    def _format_hms(seconds: float) -> str:
        s = int(seconds)
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    results_dir = _results_dir("results")
    files = _find_history_files(results_dir)

    if progress:
        typer.echo(f"[hypervolume] Found {len(files)} history file(s) under {results_dir}")

    df0 = pd.read_csv(files[0])
    obj_cols = _detect_objective_columns(df0.columns, results_dir)
    sense = _detect_objective_sense(results_dir, obj_cols)

    def _seed_label(path: Path) -> int | str:
        m = re.search(r"seed(\d+)", path.name)
        return int(m.group(1)) if m else path.stem

    def load_minimized(path: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        F = df[obj_cols].to_numpy(dtype=float)
        G = df["generation"].astype(int).to_numpy()
        return F, G

    eps = 1e-12
    if normalize_scope.lower() == "global":
        g_min, g_max = None, None
        t0_norm = _t.time()
        for p in files:
            F_min, _ = load_minimized(p)
            m = np.nanmin(F_min, axis=0)
            M = np.nanmax(F_min, axis=0)
            if g_min is None:
                g_min, g_max = m, M
            else:
                g_min = np.minimum(g_min, m)
                g_max = np.maximum(g_max, M)
        assert g_min is not None and g_max is not None
        g_span = np.maximum(g_max - g_min, eps)
        if progress:
            typer.echo(f"[hypervolume] Global normalization stats computed in {_format_hms(_t.time()-t0_norm)}")

    typer.echo(
        f"[hypervolume] mode={'cumulative' if cumulative else 'per-generation'}, "
        f"normalize={normalize_scope}, epsilon={epsilon if (epsilon and not cumulative) else 'ignored in cumulative'}"
    )

    per_seed_frames = []
    first_seed_dual = None  # (per-gen DF, cumulative DF) when debug=True & cumulative=True

    # Process each seed file
    for ix, path in enumerate(files):
        seed_lbl = _seed_label(path)
        t0_seed = _t.time()

        if progress:
            typer.echo(f"[hypervolume] Seed {seed_lbl}: loading & minimizing ...")

        F_min, G = load_minimized(path)

        if normalize_scope.lower() == "per-seed":
            m = np.nanmin(F_min, axis=0); M = np.nanmax(F_min, axis=0)
            span = np.maximum(M - m, eps)
        else:
            m, span = g_min, g_span  # global stats

        F_norm = (F_min - m) / span
        good = np.isfinite(F_norm).all(axis=1)
        n_bad = (np.size(good) - int(good.sum()))
        if n_bad and progress:
            typer.echo(f"[hypervolume]   Seed {seed_lbl}: dropping {n_bad} invalid row(s) (NaN/Inf)")

        F_norm, G = F_norm[good], G[good]
        F_norm = np.clip(F_norm, 0.0, 1.0)

        rp = np.ones(F_norm.shape[1], dtype=float) + 1e-9  # strict dominance
        gens = np.unique(G)
        n_gens = len(gens)
        seed_col = f"seed{seed_lbl}"

        if progress:
            typer.echo(f"[hypervolume]   Seed {seed_lbl}: {n_gens} generation(s) | M={F_norm.shape[1]}")

        rows_pg, rows_cu = [], []
        if cumulative:
            # Cumulative ND union (monotone) — NO ε-thinning here by design.
            F_acc = np.empty((0, F_norm.shape[1]), dtype=float)
            for j, g in enumerate(gens, 1):
                Fg = F_norm[G == g]
                if Fg.size:
                    F_acc = np.vstack([F_acc, Fg])

                nd_mask = _nd_mask_eps(F_acc, eps=1e-9, maximize_cols=None)
                F_front_cu = F_acc[nd_mask]
                hv_val = float(HV(ref_point=rp)(F_front_cu.astype(np.float64, copy=False)))
                rows_cu.append({"generation": int(g), seed_col: hv_val})

                # Progress line every print_every steps (or last)
                if progress and (j % print_every == 0 or j == n_gens):
                    typer.echo(
                        f"[hypervolume]   Seed {seed_lbl} | gen {int(g)} ({j}/{n_gens}) "
                        f"| cumulative front size={F_front_cu.shape[0]} | HV={hv_val:.6f}"
                    )

                # Monotonicity guard
                if len(rows_cu) >= 2:
                    prev = rows_cu[-2][seed_col]; curr = rows_cu[-1][seed_col]
                    if curr + 1e-12 < prev:
                        if debug:
                            pd.DataFrame(F_front_cu, columns=obj_cols).to_csv(
                                results_dir / f"hv_dbg_cumulative_drop_at_g{int(g)}_{path.name}.csv",
                                index=False
                            )
                        raise RuntimeError(
                            f"Cumulative HV decreased at generation {g} for {path.name}: {curr:.6f} < {prev:.6f}. "
                            f"Check normalization and ND inputs."
                        )

                # Optional: also compute per-generation (first seed) for side-by-side debug
                if debug and ix == 0:
                    if Fg.size == 0:
                        rows_pg.append({"generation": int(g), seed_col + "_pergen": 0.0})
                    else:
                        ndg = _nd_mask_eps(Fg, eps=1e-9, maximize_cols=None)
                        Fg_front = Fg[ndg]
                        if epsilon and epsilon > 0.0 and Fg_front.shape[0] > 0:
                            idx = _eps_thin(Fg_front, eps=float(epsilon))
                            Fg_front = Fg_front[idx] if idx.size > 0 else np.empty((0, Fg_front.shape[1]))
                        rows_pg.append({
                            "generation": int(g),
                            seed_col + "_pergen": float(HV(ref_point=rp)(Fg_front.astype(np.float64, copy=False)))
                        })

            df_cu = pd.DataFrame(rows_cu)
            per_seed_frames.append(df_cu)
            if debug and ix == 0:
                df_pg = pd.DataFrame(rows_pg)
                first_seed_dual = (df_pg, df_cu)
        else:
            # Per-generation ND HV (may dip). Optional ε-thinning HERE only.
            for j, g in enumerate(gens, 1):
                Fg = F_norm[G == g]
                if Fg.size == 0:
                    rows_pg.append({"generation": int(g), seed_col: 0.0})
                    if progress and (j % print_every == 0 or j == n_gens):
                        typer.echo(f"[hypervolume]   Seed {seed_lbl} | gen {int(g)} ({j}/{n_gens}) | empty batch")
                    continue

                nd = _nd_mask_eps(Fg, eps=1e-9, maximize_cols=None)
                F_front = Fg[nd]
                if epsilon and epsilon > 0.0 and F_front.shape[0] > 0:
                    idx_keep = _eps_thin(F_front, eps=float(epsilon))
                    F_front = F_front[idx_keep] if idx_keep.size > 0 else np.empty((0, F_front.shape[1]))

                hv_val = float(HV(ref_point=rp)(F_front.astype(np.float64, copy=False)))
                rows_pg.append({"generation": int(g), seed_col: hv_val})

                if progress and (j % print_every == 0 or j == n_gens):
                    typer.echo(
                        f"[hypervolume]   Seed {seed_lbl} | gen {int(g)} ({j}/{n_gens}) "
                        f"| front size={F_front.shape[0]} | HV={hv_val:.6f}"
                    )

            per_seed_frames.append(pd.DataFrame(rows_pg))

        if progress:
            typer.echo(f"[hypervolume] Seed {seed_lbl} done in {_format_hms(_t.time()-t0_seed)}")

    # Merge wide table
    hv_wide = None
    for df_seed in per_seed_frames:
        hv_wide = df_seed if hv_wide is None else hv_wide.merge(df_seed, on="generation", how="outer")

    hv_wide = hv_wide.sort_values("generation")
    seed_cols = sorted(
        [c for c in hv_wide.columns if c != "generation"],
        key=lambda s: int(s.removeprefix("seed")) if s.startswith("seed") and s[4:].isdigit() else s,
    )
    hv_wide = hv_wide[["generation"] + seed_cols]

    out = results_dir / "hypervolume.csv"
    hv_wide.to_csv(out, index=False)

    if debug and first_seed_dual is not None and cumulative:
        pg, cu = first_seed_dual
        dbg = pg.merge(cu, on="generation", how="outer").sort_values("generation")
        dbg_out = results_dir / "hypervolume_debug_first_seed.csv"
        dbg.to_csv(dbg_out, index=False)
        typer.echo(f"[hypervolume] Debug (first seed per-gen vs cumulative) → {dbg_out}")

    typer.echo(
        f"[hypervolume] Saved: {out} "
        f"(mode={'cumulative' if cumulative else 'per-generation'}, normalize={normalize_scope})"
    )


# ======================================================================================
# Utilities (helpers for analysis commands) — unchanged
# ======================================================================================

def _results_dir(default: str = "results") -> Path:
    """Return the absolute results folder, or raise a clear error if missing."""
    rd = Path(default).resolve()
    if not rd.exists():
        raise typer.BadParameter(f"Results directory not found: {rd}")
    return rd

def _find_history_files(results_dir: Path) -> list[Path]:
    """Find all history*.csv directly under results/ (non-recursive)."""
    files = sorted(results_dir.glob("history*.csv"))
    if not files:
        raise typer.BadParameter(f"No history*.csv files found under: {results_dir}")
    return files

def _detect_objective_columns(df_columns: Iterable[str], results_dir: Path) -> list[str]:
    """
    Auto-detect objective columns for analysis:
      1) If unlabeled, expect f0,f1,... in the CSV header and use those.
      2) Otherwise, read objective_declaration.csv (near results/) and match its Name column.
    """
    cols = list(df_columns)

    fcols = [c for c in cols if re.fullmatch(r"f\d+", c)]
    if fcols:
        return fcols

    import pandas as pd
    candidates = [
        results_dir / "objective_declaration.csv",
        results_dir.parent / "objective_declaration.csv",
        results_dir.parent.parent / "objective_declaration.csv",
    ]
    for decl in candidates:
        if decl.exists():
            dfo = pd.read_csv(decl)
            if "Name" not in dfo.columns:
                continue
            names = dfo["Name"].astype(str).tolist()
            if all(n in cols for n in names):
                return names

    raise typer.BadParameter(
        "Could not auto-detect objective columns.\n"
        "- If your history has unlabeled objectives, they must be named f0,f1,...\n"
        "- If labeled, place `objective_declaration.csv` next to results/ (or its parent), "
        "and ensure history was written with --label-columns."
    )

def _detect_objective_sense(results_dir: Path, obj_cols: list[str]) -> np.ndarray:
    """
    Return +1 for 'minimize' and -1 for 'maximize' (per objective),
    reading objective_declaration.csv near results/. If not found or names do not match,
    assume history is already in pure minimization.
    """
    import pandas as pd
    candidates = [
        results_dir / "objective_declaration.csv",
        results_dir.parent / "objective_declaration.csv",
        results_dir.parent.parent / "objective_declaration.csv",
    ]
    for decl in candidates:
        if decl.exists():
            dfo = pd.read_csv(decl)
            if {"Name", "Objective"}.issubset(dfo.columns):
                name_to_dir = {
                    str(row["Name"]): str(row["Objective"]).strip().lower()
                    for _, row in dfo.iterrows()
                }
                if all(c in name_to_dir for c in obj_cols):
                    return np.array([
                        +1.0 if name_to_dir[c] in {"min", "minimize", "minimise"} else -1.0
                        for c in obj_cols
                    ], dtype=float)
    return np.ones(len(obj_cols), dtype=float)

# ======================================================================================
# version
# ======================================================================================

@app.command()
def version():
    try:
        from .. import __version__
    except Exception:  # pragma: no cover
        __version__ = "0.0.0+unknown"
    typer.echo(__version__)


if __name__ == "__main__":
    app()