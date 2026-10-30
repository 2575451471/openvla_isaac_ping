#!/usr/bin/env bash
# Explicit CPU preflight or single-GPU trial; never invokes the legacy pipeline.
set -euo pipefail
ROOT="/home/dt/openvla/shenle_projects"
PYTHON="/home/dt/miniconda3/envs/openvla/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:---help}" in
  --check)
    shift
    if [[ "$#" -ne 0 ]]; then echo "--check accepts no extra arguments" >&2; exit 2; fi
    exec "$PYTHON" "$ROOT/scripts/dataset_raw/prepare_pose_v2_training.py"
    ;;
  --train)
    shift
    exec "$PYTHON" "$ROOT/scripts/temp/finetune_pose_v2.py" "$@"
    ;;
  --help|-h)
    echo "CPU preflight: bash $ROOT/scripts/temp/run_pose_v2_trial.sh --check"
    echo "GPU trial (200 updates): bash $ROOT/scripts/temp/run_pose_v2_trial.sh --train"
    ;;
  *) echo "Choose --check or --train" >&2; exit 2 ;;
esac
