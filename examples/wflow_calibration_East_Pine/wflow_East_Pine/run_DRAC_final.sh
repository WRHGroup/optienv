#!/bin/bash
#SBATCH --account=def-mbasheer
#SBATCH --time=0-06:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

# Single Wflow run: Julia uses all allocated cores
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_GC_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11          # venv is cp311
module load hdf5/1.14.2
module load netcdf/4.9.2
module load proj/9.2.0
module load gdal/3.9.1
module load arrow/17.0.0

# HDF5/NetCDF safety on shared FS
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=${SLURM_TMPDIR:-/tmp}
export TEMP=$TMPDIR
export TMP=$TMPDIR

# Same metric settings as calibration (used only with --mode calibrate-style)
export WRAPPER_WARMUP_YEARS=0
export WRAPPER_EPS_FACTOR=0.01

MODEL_FOLDER="/lustre07/scratch/alaminie/wflow_calibration_East_Pine/wflow_East_Pine"

# Python packages for the wrapper (pandas, numpy, xarray, netCDF4)
source /project/6087451/alaminie/Nile/envs/optienv/bin/activate
export PYTHONPATH=""

chmod +x "${MODEL_FOLDER}/Wflow-julia.sh"
cd "${MODEL_FOLDER}"

# Apply calibrated params from variable_values.csv -> staticmaps.nc, then run Wflow once
python "${MODEL_FOLDER}/wrapper_wflow_final.py" "${MODEL_FOLDER}" --mode apply-and-run