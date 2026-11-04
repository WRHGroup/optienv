#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import csv
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def create_tbl_files(model_folder):
    # Load CSV written by the calibration tool (MOEA/NSGAII)
    df = pd.read_csv(os.path.join(model_folder, "variable_values.csv"))
    output_folder = model_folder

    # Split 'Name' (e.g. "thetaS.3") into Parameter and calibration class id (handles names without a dot safely)
    split = df['Name'].astype(str).str.split('.', n=1, expand=True)
    df['Parameter'] = split[0]
    df['ClassID'] = split[1] if split.shape[1] > 1 else None

    df['ClassID'] = pd.to_numeric(
        df['ClassID'].astype(str).str.extract(r'(\d+)', expand=False),
        errors='coerce'
    ).fillna(1).astype(int)

    # Get unique parameters
    parameters = df['Parameter'].unique()

    # Iterate through parameters and save to separate .tbl files
    # Line format: <class_id> 1 <,12] <value>
    # update_staticmaps.py reads only the first column (class id) and the last column (value).
    for param in parameters:
        df_param = df[df['Parameter'] == param].sort_values('ClassID')
        lines = [f"{int(row.ClassID)}\t1\t<,12]  {float(row.Value):.6f}"
                 for row in df_param.itertuples(index=False)]

        filename = os.path.join(output_folder, f"{param}.tbl")
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))


def update_static_maps(model_folder):
    # Optional env overrides for the updater (unset = original behaviour)
    scope = os.environ.get("WRAPPER_BOUNDS_SCOPE")   # "all" (default) or "calibrated"
    strict = os.environ.get("WRAPPER_STRICT")        # "1" to abort on out of range .tbl values

    import update_staticmaps
    sys.argv = [sys.argv[0],
                "--staticmap", os.path.join(model_folder, "staticmaps.nc"),
                "--param_dir", model_folder,
                "--backup"]
    if scope:
        sys.argv += ["--bounds_scope", scope]
    if strict and strict.strip().lower() in ("1", "true", "yes"):
        sys.argv += ["--strict"]
    update_staticmaps.main()


def run_wflow(model_folder):
    # Linux/cluster friendly: run the .sh with bash in the model folder
    script = os.path.join(model_folder, "Wflow-julia.sh")
    if not os.path.isfile(script):
        print(f"[wrapper] ERROR: {script} not found", file=sys.stderr)
        try:
            print("[wrapper] Listing model folder:", file=sys.stderr)
            print("\n".join(os.listdir(model_folder)), file=sys.stderr)
        except Exception as e:
            print(f"[wrapper] Failed to list {model_folder}: {e}", file=sys.stderr)
        raise FileNotFoundError(script)
    subprocess.run(["bash", "Wflow-julia.sh"], cwd=model_folder, check=True)


def calculate_metrics(model_folder):
    # Pass optional env overrides for warm up years / epsilon (keeps your prior workflow)
    wy = os.environ.get("WRAPPER_WARMUP_YEARS")  # optional
    epsf = os.environ.get("WRAPPER_EPS_FACTOR")  # optional

    import compute_metrics
    sys.argv = [sys.argv[0],
                "--sim", os.path.join(model_folder, "run_default", "output.csv"),
                "--obs", os.path.join(model_folder, "observed_streamflow.csv"),
                "--out_dir", model_folder]
    if wy:
        sys.argv += ["--warmup_years", str(int(wy))]
    if epsf:
        sys.argv += ["--eps_factor", str(float(epsf))]

    compute_metrics.main()


def prepare_objectives(model_folder):
    input_file  = os.path.join(model_folder, "metrics.txt")
    output_file = os.path.join(model_folder, "objective_values.csv")
    names = ["KGEp", "logNSE", "bias_score"]
    with open(input_file, "r") as f:
        values = f.readline().strip().split(",")
    if len(values) != 3:
        raise RuntimeError(f"[wrapper] Expected 3 costs in metrics.txt, got {len(values)}")
    with open(output_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Value"])
        w.writerows(zip(names, values))


def search_and_apply_variables(model_folder):
    create_tbl_files(model_folder)
    update_static_maps(model_folder)
    run_wflow(model_folder)
    calculate_metrics(model_folder)
    prepare_objectives(model_folder)


def simulate(model_folder):
    run_wflow(model_folder)
    calculate_metrics(model_folder)
    prepare_objectives(model_folder)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_folder")
    p.add_argument("--mode", choices=["run-only", "apply-and-run", "simulate", "calibrate-style"],
                   default="apply-and-run")
    args = p.parse_args()

    if args.mode == "run-only":
        run_wflow(args.model_folder)

    elif args.mode == "apply-and-run":
        create_tbl_files(args.model_folder)
        update_static_maps(args.model_folder)
        run_wflow(args.model_folder)

    elif args.mode == "simulate":
        simulate(args.model_folder)

    elif args.mode == "calibrate-style":
        search_and_apply_variables(args.model_folder)
