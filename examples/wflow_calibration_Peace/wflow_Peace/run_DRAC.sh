#!/bin/bash
#SBATCH --account=def-mbasheer
#SBATCH --time=2-12:58
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-2%1

# ---------------------------------------------------------------------------
# Threading: parallelism lives at the POPULATION level (optienv -j), so every
# wflow/Julia process must stay single-threaded. Setting JULIA_NUM_THREADS to
# $SLURM_CPUS_PER_TASK while also running -j $SLURM_CPUS_PER_TASK would put
# 8 x 8 = 64 threads on 8 allocated cores.
# ---------------------------------------------------------------------------
export JULIA_NUM_THREADS=1
export JULIA_NUM_GC_THREADS=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11          # venv is cp311; 3.10 mismatches it
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

# Wrapper environment
export WRAPPER_WARMUP_YEARS=0   # 0 for this short test example; for a full multiyear run, use 1 to 3 years
export WRAPPER_EPS_FACTOR=0.01

source /project/6087451/alaminie/Nile/envs/optienv/bin/activate
export PYTHONPATH=""

# Run from the calibration folder (where run_sim.json sits, next to wflow_Tana),
# so -c run_sim.json resolves wherever sbatch is called from.
cd /lustre07/scratch/alaminie/GitHub_Optienv_Addis/wflow_calibration_Peace || exit 1

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
OUTDIR="results/seed${TASK_ID}"

# ---------------------------------------------------------------------------
# Resuming after a stop (time limit, cancellation, or days later):
#   1. Check the checkpoints exist:  ls results/seed*/checkpoint.npz
#   2. Resubmit this same script:    sbatch run_DRAC.sh
#      No edits needed: if results/seedN/checkpoint.npz exists, --resume-latest
#      is added automatically and each seed continues from its last generation.
#   3. To resume only some seeds:    sbatch --array=3,5 run_DRAC.sh
#   4. To extend a finished run: raise "generations" in run_sim.json, then
#      resubmit; resume continues toward the new total.
# Note: --resume-latest raises BadParameter when no checkpoint exists, which is
# why it is only passed when the checkpoint file is found.
# ---------------------------------------------------------------------------
RESUME=""
[[ -f "${OUTDIR}/checkpoint.npz" ]] && RESUME="--resume-latest"

optienv search -c run_sim.json \
  --algo nsga2 -j "$SLURM_CPUS_PER_TASK" \
  --seed "$TASK_ID" \
  -o "$OUTDIR" \
  --work-root "$SLURM_TMPDIR" \
  --checkpoint-every 1 \
  $RESUME

# Post-processing (optienv 0.1.1 replaces the old optiverse commands):
#   optienv front       -o results/ --epsilon 0.01
#   optienv hypervolume -o results/ --cumulative



