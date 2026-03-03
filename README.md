# OptiVerse

**OptiVerse** is a lightweight, *simulator‑friendly* framework for **multi‑objective optimization** (MOO). It wraps your external models with a simple CSV interface and lets you run evolutionary algorithms—currently **NSGA‑II** and **NSGA‑III**—at scale. It focuses on:

- **Reproducible experiments**: single‑file per‑seed histories, optional checkpoints/resume, and fixed seeds.
- **Simulator integration**: CSV adapter that reads variable assignments and writes objective values—no code rewrites inside your model.
- **Many‑objective capability**: NSGA‑III with reference directions for high‑dimensional objectives (≥ 4).
- **Actionable analysis**: global Pareto front extraction and **normalized** hypervolume (HV) computation without a user‑provided reference point, including density control via `--epsilon`.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Install](#install)
  - [From PyPI](#from-pypi)
  - [For Development](#for-development)
- [Core Concepts](#core-concepts)
  - [CSV Wrapper Contract](#csv-wrapper-contract)
  - [Run Configuration JSON](#run-configuration-json)
  - [Single‑File History](#singlefile-history)
- [CLI Usage](#cli-usage)
  - [`search` (NSGA‑II / NSGA‑III)](#search-nsga-ii--nsga-iii)
  - [`front`](#front)
  - [`hypervolume` (normalized, wide output)](#hypervolume-normalized-wide-output)
- [NSGA‑III Reference Directions](#nsgaiii-reference-directions)
- [Resume / Checkpointing](#resume--checkpointing)
- [Progress Output & Parallel Backends](#progress-output--parallel-backends)
- [Examples](#examples)
  - [3‑Objective Toy](#3objective-toy)
  - [6‑Objective Toy (NSGA‑III vs NSGA‑II)](#6objective-toy-nsgaiii-vs-nsga-ii)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Citation / Acknowledgement](#citation--acknowledgement)

---

## Features

- **Algorithms**
  - **NSGA‑II**: fast non‑dominated sorting + crowding distance (strong baseline).
  - **NSGA‑III**: fast non‑dominated sorting + **reference‑direction niching** (robust diversity for many objectives).
- **CSV model adapter**
  - Reads `variable_values.csv` (decision vector).
  - Calls your wrapper function (you control the simulation).
  - Reads `objective_values.csv` (objective values).
  - Converts *maximize* objectives to *minimization* internally (consistent sorting & indicators).
- **Results management**
  - **One** history CSV per seed (`results/history_seed{SEED}.csv`) containing all generations.
  - Optional final CSVs: population, fitness, and non‑dominated mask.
- **Analysis**
  - `front`: global non‑dominated set across all seeds + `seed` provenance column, optional ε‑box thinning.
  - `hypervolume`: **normalized** to [0,1] across all histories; **no ref point needed**; **wide** output (one HV column per seed).
- **Robustness for Windows/Clusters**
  - Lean worker processes (no heavy imports) and aggressive cleanup of temp folders (read‑only clearing, retries, background reaper, `atexit`).

---

## Requirements

- **Python**: 3.10+  
- **OS**: Linux / macOS / Windows
- **Optional**: `pymoo` (recommended for HV with > 2 objectives)
  ```bash
  pip install pymoo
  ```

---

## Install

### From PyPI

```bash
pip install optiverse
```

> Tip: Use a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows).

### For Development

```bash
git clone https://github.com/WRHGroup/optiverse.git
cd optiverse
pip install -e .
```

Reinstall after code changes to ensure the console script picks them up:
```bash
pip install -e .
```

---

## Core Concepts

### CSV Wrapper Contract

Your model’s **wrapper** (a Python file inside `model_dir`) must define:

```python
def search_and_apply_variables(model_folder: str) -> None:
    ...
```

At runtime, OptiVerse:

1. Copies `model_dir` to a temp **work** folder.
2. Writes `work/variable_values.csv` (columns: `Name,Value`).
3. Imports your wrapper from the working copy and calls `search_and_apply_variables(work)`.
4. Expects the wrapper to write `work/objective_values.csv` (columns: `Name,Value`).
5. Converts any *maximize* objectives to minimization (sign flip).
6. Deletes the working copy (unless `--keep-work-on-error`).

**Tip (Windows/external tools)**  
Ensure your wrapper waits for external processes to finish and **closes file handles**:
```python
import subprocess, os
with open(os.devnull, "wb") as devnull:
    subprocess.run(["your-sim.exe"], check=True, stdout=devnull, stderr=devnull, close_fds=True)
```

### Run Configuration JSON

A minimal `run_sim.json`:
```json
{
  "model": {
    "model_dir": "./toy_model",
    "wrapper_file": "wrapper_toy.py",
    "variables_csv": "../variable_declaration.csv",
    "objectives_csv": "../objective_declaration.csv"
  },
  "algorithm": {
    "population_size": 100,
    "generations": 200
  }
}
```

- `model_dir`: folder containing your wrapper file and any model assets.
- `wrapper_file`: Python file implementing `search_and_apply_variables(...)`.
- `variables_csv`: CSV with columns `Name,Upper_bound,Lower_bound`.
- `objectives_csv`: CSV with columns `Name,Objective` where `Objective∈{minimize,minimise,min,maximize,maximise,max}`.

### Single‑File History

`results/history_seed{SEED}.csv` (all generations in one file) with columns:

```
generation,index,<objective_1>,...,<objective_M>,<x1>,...,<xN>,nd
```

- Objectives appear first (either your names if `--label-columns` was used or fallback `f0,f1,...`).
- Variables follow.
- `nd` is a per‑generation non‑dominated mask (1 = non‑dominated among that generation’s population).

---

## CLI Usage

### `search` (NSGA‑II / NSGA‑III)

Run NSGA‑II:
```bash
optiverse search \
  -c path/to/run_sim.json \
  --algo nsga2 \
  -j 4 --seed 7 \
  --label-columns \
  --no-save-final-csvs
```

Run NSGA‑III with reference directions:
```bash
optiverse search \
  -c path/to/run_sim.json \
  --algo nsga3 --ref-parts 4 \
  -j 4 --seed 7 \
  --label-columns --no-save-final-csvs
```

**Key options**

- `--algo`: `nsga2` (default) or `nsga3`.
- **NSGA‑III only**
  - `--ref-parts <p>`: build Das–Dennis simplex‑lattice reference directions (count = `binom(M+p-1, p)`).
  - `--ref-dirs-csv <file>`: supply custom reference directions (rows=dirs, cols=#objectives). Rows are auto‑normalized; overrides `--ref-parts`.
- Parallelism & infra
  - `-j/--max-workers <N>`: workers (processes by default).
  - `--backend process|thread`: use `thread` on RAM‑tight nodes.
  - `--work-root <dir>`: where temp model copies are created.
  - `--keep-work-on-error`: keep working copy if a failure occurs.
- Outputs
  - `--label-columns`: write objective/variable names as headers.
  - `--no-save-final-csvs`: only write the single history file.
- Resume/checkpoints
  - `--checkpoint-every <N>`: write a checkpoint every N generations.
  - `--checkpoint-path <file>`: where to store the checkpoint (default `results/checkpoint.npz`).
  - `--resume-latest` / `--resume-from <file>`: continue from a checkpoint.

---

### `front`

Compute a **global** non‑dominated set across **all** `results/history*.csv`:

```bash
optiverse front
# → results/pareto_front_all.csv
```

- Automatically converts objective sense using your declaration (if available).
- Adds a `seed` column (provenance).
- `--epsilon <eps>`: optional ε‑box thinning to control density (e.g., `0.01`).

---

### `hypervolume` (normalized, wide output)

Compute **normalized** HV per generation for **each seed**, and write a **wide** CSV (one HV column per seed):

```bash
optiverse hypervolume
# → results/hypervolume.csv
# columns: generation, seed7, seed11, ...
```

- No reference point needed: objectives are first converted to minimization and normalized to **[0,1]** using global min/max across all histories; HV uses a fixed ref \(r=\mathbf{1}\).
- `--epsilon <eps>`: optionally thin each generation’s front first (speeds up HV, smoother curves). Try `0.01`.

> For objectives > 2, install `pymoo` to enable the HV indicator. Without it, a 2‑D fallback is used.

---

## NSGA‑III Reference Directions

NSGA‑III uses **reference‑direction niching** (instead of crowding distance) to maintain a **uniform spread** across the Pareto front—especially important for many objectives.

- Default generation: **Das–Dennis** simplex‑lattice with `--ref-parts p`. Number of directions:
  \[
  \binom{M+p-1}{p}
  \]
- **Best practice**: **population size ≈ number of reference directions** (≈ one survivor per niche).
- High‑dimensional setups (e.g., \(M\ge 6\)):
  - Keep `--ref-parts` small (e.g., 3–4), **or**
  - Provide a **custom** `--ref-dirs-csv` with your desired count and distribution (e.g., 100–300 well‑spaced directions).

---

## Resume / Checkpointing

Enable checkpoints during a run:

```bash
optiverse search ... --checkpoint-every 1 --checkpoint-path results/checkpoint.npz
```

Resume later:

```bash
# Resume from default checkpoint
optiverse search ... --resume-latest

# Or resume from a specific path
optiverse search ... --resume-from path/to/checkpoint.npz
```

Checkpoints store generation index, population, fitness, RNG state, variable/objective names, bounds, history path, and model_dir—so resuming is safe and consistent.

---

## Progress Output & Parallel Backends

During evaluation you’ll see:

```
[search] Gen 5/50 – evaluating 126 candidates (workers=8, backend=process)...
  completed:  63/126 ( 50%)  elapsed: 00:01:12
  completed: 126/126 (100%)  elapsed: 00:02:21
[search] Gen 5 done in 00:02:21
```

- `--backend process` (default) leverages multiple CPUs safely for external simulators.
- `--backend thread` can reduce memory overhead on RAM‑constrained Windows nodes.

---

## Examples

### 3‑Objective Toy

Structure:
```
examples/toy_3obj/
├─ toy_model/
│  └─ wrapper_toy.py
├─ variable_declaration.csv
├─ objective_declaration.csv
└─ run_sim_example.json
```

Run:
```bash
cd examples/toy_3obj

# NSGA‑II
optiverse search -c run_sim_example.json \
  --algo nsga2 -j 2 --seed 7 --label-columns --no-save-final-csvs

# NSGA‑III
optiverse search -c run_sim_example.json \
  --algo nsga3 --ref-parts 8 -j 2 --seed 7 --label-columns --no-save-final-csvs

# Global front and HV
optiverse front --epsilon 0.02
optiverse hypervolume
```

### 6‑Objective Toy (NSGA‑III vs NSGA‑II)

Structure:
```
examples/toy_6obj/
├─ toy_model/
│  └─ wrapper_toy6.py
├─ variable_declaration.csv
├─ objective_declaration.csv
└─ run_sim_example.json
```

For **M=6**, `--ref-parts 4` ⇒ 126 reference directions; match pop≈126 for NSGA‑III:

```bash
cd examples/toy_6obj

# NSGA‑II baseline
optiverse search -c run_sim_example.json \
  --algo nsga2 -j 4 --seed 7 --label-columns --no-save-final-csvs

# NSGA‑III with matched pop size (edit JSON or override before running)
# Example using jq to set population to 126:
jq '.algorithm.population_size=126' run_sim_example.json > run_sim_example_126.json

optiverse search -c run_sim_example_126.json \
  --algo nsga3 --ref-parts 4 -j 4 --seed 7 --label-columns --no-save-final-csvs

# Compare HV across seeds/generations
optiverse hypervolume
```

You should see NSGA‑III achieve **higher normalized HV** and better spread in 6 objectives for a similar evaluation budget.

---

## Troubleshooting

- **Temp model copies persist on Windows**
  - We clear read‑only flags and retry with exponential backoff; a background reaper thread and `atexit` attempt final cleanup.
  - Ensure your wrapper **does not `chdir`**, **closes all file handles**, and **blocks** for child processes to finish (`subprocess.run(..., check=True, close_fds=True)`).

- **Many‑objective HV errors**
  - Install `pymoo` for HV in ≥ 3 objectives:
    ```bash
    pip install pymoo
    ```

- **Resuming with changed CSVs**
  - The checkpoint validates variables, objectives, and bounds. If you changed them, start a fresh run.

- **NSGA‑III population vs reference directions**
  - It’s common to set `population_size ≈ #reference_directions`. If they differ you’ll get a note, but the run proceeds.

---

## Roadmap

- More MOEAs: **MOEA/D**, **IBEA**, **SMS‑EMOA** (HV‑based survival).
- Reference‑direction factories (Riesz‑energy, multi‑layer designs) directly in the CLI.
- Rich plots & PDF reports.
- Surrogate‑assisted runs for expensive simulators.

---

## License

**MIT** — see [LICENSE](LICENSE).

---

## Citation / Acknowledgement

If OptiVerse helps your research or engineering work, please cite or acknowledge it. This greatly helps the project grow and justifies continued development.

---
