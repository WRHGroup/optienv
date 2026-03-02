#!/bin/bash
#SBATCH --account=def-mbasheer
#SBATCH --time=0-03:58
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --array=1

# Use all Slurm-allocated CPUs for Julia threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Optional: match Julia’s garbage collection threads
export JULIA_NUM_GC_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.10
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

export WRAPPER_WARMUP_YEARS=0
export WRAPPER_EPS_FACTOR=0
#export WRAPPER_WARMUP_YEARS=1      # default already 1
#export WRAPPER_EPS_FACTOR=0.01     # log-NSE epsilon factor

MODEL_FOLDER="/scratch/alaminie/wflow_calibration/wflow_Tana"  

# Make sure the run script is executable
chmod +x "${MODEL_FOLDER}/Wflow-julia.sh"

# Activate HydroMT_Wflow venv (Fir)
source /project/6087451/alaminie/Nile/HydroMT_Wflow/bin/activate

# Apply calibrated params from variable_values.csv -> staticmaps.nc, then run Wflow once
python wrapper_wflow.py "${MODEL_FOLDER}" --mode apply-and-run
