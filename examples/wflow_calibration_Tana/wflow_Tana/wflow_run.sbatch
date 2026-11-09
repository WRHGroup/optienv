#!/bin/bash
set -euo pipefail

# Clean environment
module --force purge
# module load gcc arrow/17.0.0 gdal/3.9.1 proj hdf5/1.14.2 netcdf/4.9.2

# Julia setup
export PATH=/project/6087451/alaminie/Nile/apps/julia-1.11.6/bin:$PATH
export JULIA_DEPOT_PATH=/project/6087451/alaminie/Nile/julia_depot
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Wflow for 2km resolution and 30 yrs
julia --project=/project/6087451/alaminie/Nile/envs/wflow -e '
using Wflow
Wflow.run("wflow_sbm.toml")
'
