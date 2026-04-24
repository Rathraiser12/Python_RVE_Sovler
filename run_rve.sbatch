#!/bin/bash -l
#SBATCH --job-name=rve
#SBATCH --partition=work
#SBATCH --constraint=icx
#SBATCH --time=17:00:00
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

set -euo pipefail

module load python/3.12-conda
eval "$(conda shell.bash hook)"
conda activate rve
cd $SLURM_SUBMIT_DIR
# keep BLAS libs from oversubscribing threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p logs meshes

# run exactly once on this node (don’t parallelize via srun ntasks)
# srun -n 1 --cpu-bind=cores python -u create_rve_mesh.py
srun -n 1 --cpu-bind=cores python -u generate_working.py
