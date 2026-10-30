#!/bin/bash
# NCSA Delta: one run_sweep_v3 point per array task, one GPU each.
# Submit with slurm/submit_delta.sh, which sizes --array from run_sweep_v3's point list.
# Extra arguments are passed to run_sweep_v3 (e.g. -bs 1250 --codes 288,360).
#SBATCH --job-name=syndrilla_sweep
#SBATCH --account=bebv-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out

set -euo pipefail

CONDA_BASE=${CONDA_BASE:-/projects/bebv/cpeng4/miniforge3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate syndrilla
# compile the CUDA kernels with the env's nvcc (CUDA 13.0, matches torch), not the cudatoolkit module
export CUDA_HOME=$CONDA_PREFIX

cd "$SLURM_SUBMIT_DIR"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# keep the GPU Slurm assigned: run_sweep_v3 sets CUDA_VISIBLE_DEVICES to the --gpus value
python run_sweep_v3.py --task-index "$SLURM_ARRAY_TASK_ID" --gpus "${CUDA_VISIBLE_DEVICES:-0}" "$@"
