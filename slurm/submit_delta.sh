#!/bin/bash
# Submit the run_sweep_v3 point list to NCSA Delta as one array job.
# Run from anywhere with the syndrilla env active; arguments go to run_sweep_v3, e.g.
#   bash slurm/submit_delta.sh
#   bash slurm/submit_delta.sh --codes 288,360 -bs 312
# PARTITION picks the Slurm partition (default gpuA100x4). A -preempt partition also requeues tasks it kills;
# they resume from their result file:
#   PARTITION=gpuA100x4-preempt bash slurm/submit_delta.sh --codes 288 --rates 0.001
set -euo pipefail

cd "$(dirname "$0")/.."
N=$(python run_sweep_v3.py --count-tasks "$@")
if [ "$N" -lt 1 ]; then
    echo "no points to run"
    exit 1
fi
mkdir -p slurm/logs
python run_sweep_v3.py --dry-run "$@" | grep -E "Total points|Codes|Error Rates|Algorithms|KNOBS"
PARTITION=${PARTITION:-gpuA100x4}
SBATCH_ARGS=(--partition="$PARTITION")
case "$PARTITION" in
    *-preempt) SBATCH_ARGS+=(--requeue --open-mode=append) ;;
esac
sbatch "${SBATCH_ARGS[@]}" --array=0-$((N - 1)) slurm/delta_sweep.sbatch "$@"
