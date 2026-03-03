#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import csv
import pandas as pd

def create_tbl_files(model_folder):

    # Load CSV
    df = pd.read_csv(os.path.join(model_folder, "variable_values.csv"))
    output_folder = model_folder

    # Split 'Name' into Parameter and Layer (handles names without a dot safely)
    split = df['Name'].astype(str).str.split('.', n=1, expand=True)
    df['Parameter'] = split[0]
    df['Layer'] = split[1]

    df['Layer'] = pd.to_numeric(
        df['Layer'].str.extract(r'(\d+)', expand=False),
        errors='coerce'
    ).fillna(1).astype(int)

    # Get unique parameters
    parameters = df['Parameter'].unique()

    # Iterate through parameters and save to separate .tbl files
    for param in parameters:
        df_param = df[df['Parameter'] == param].sort_values('Layer')

        # Build lines (two spaces before the value) and write
        lines = [f"{int(row.Layer)}\t1\t<,12]  {float(row.Value):.6f}"
                 for row in df_param.itertuples(index=False)]

        filename = os.path.join(output_folder, f"{param}.tbl")
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

def update_static_maps(model_folder):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import robust_update_staticmaps
    sys.argv = [sys.argv[0],
                "--staticmap", (model_folder + "/staticmaps.nc"),
                "--param_dir", model_folder,
                "--backup"]
    robust_update_staticmaps.main()

def run_wflow(model_folder):
    path = os.path.join(os.getcwd(), model_folder, "Wflow-julia.bat")
    subprocess.run([path], shell=True)

def calculate_metrics(model_folder):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import compute_metrics
    sys.argv = [sys.argv[0],
                "--sim", (model_folder + "/run_default/output.csv"),
                "--obs", (model_folder + "/observed_streamflow.csv"),
                "--out_dir", model_folder]

    compute_metrics.main()

def prepare_objectives(model_folder):
    input_file  = (model_folder + "/metrics.txt")
    output_file = (model_folder + "/objective_values.csv")
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