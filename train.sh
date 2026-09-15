#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  cat <<'HELP'
Usage:
  bash train.sh HF_USER/DATASET [additional lerobot-train arguments]
  DATASET_ROOT=/absolute/dataset/path bash train.sh local/my_dataset

Environment options:
  GPU=2 BATCH_SIZE=8 STEPS=20000 NUM_WORKERS=4 SAVE_FREQ=5000
  JOB_NAME=smolvla_first_run OUTPUT_DIR=/absolute/new/output/path
  DATASET_ROOT=/absolute/dataset/path

Example smoke run after supplying your dataset:
  STEPS=2 BATCH_SIZE=1 bash train.sh HF_USER/DATASET

Extra arguments are passed to lerobot-train, for example a camera rename map.
Use a new output directory for each run. No model is uploaded to the Hub.
HELP
  exit 0
fi

DATASET_REPO_ID="$1"
shift
TRAIN_BIN="$PROJECT_DIR/.venv/bin/lerobot-train"
if [[ ! -x "$TRAIN_BIN" ]]; then
  echo "Missing training environment. See README.md." >&2
  exit 1
fi
if [[ -n "${DATASET_ROOT:-}" && ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "DATASET_ROOT must contain meta/info.json: $DATASET_ROOT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export TOKENIZERS_PARALLELISM=false
TRAIN_STEPS="${STEPS:-20000}"
if [[ ! "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer." >&2
  exit 1
fi
WARMUP_STEPS=1000
if (( TRAIN_STEPS <= WARMUP_STEPS )); then
  WARMUP_STEPS=$((TRAIN_STEPS / 10))
fi
RUN_NAME="${JOB_NAME:-smolvla_$(date -u +%Y%m%dT%H%M%S)_$$}"
RUN_OUTPUT="${OUTPUT_DIR:-$PROJECT_DIR/outputs/train/$RUN_NAME}"
if [[ -e "$RUN_OUTPUT" ]]; then
  echo "Output already exists; choose a new OUTPUT_DIR: $RUN_OUTPUT" >&2
  exit 1
fi

ARGS=(
  "--policy.path=lerobot/smolvla_base"
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--policy.device=cuda"
  "--policy.push_to_hub=false"
  "--batch_size=${BATCH_SIZE:-8}"
  "--num_workers=${NUM_WORKERS:-4}"
  "--steps=$TRAIN_STEPS"
  "--policy.scheduler_warmup_steps=$WARMUP_STEPS"
  "--policy.scheduler_decay_steps=$TRAIN_STEPS"
  "--save_freq=${SAVE_FREQ:-5000}"
  "--output_dir=$RUN_OUTPUT"
  "--job_name=$RUN_NAME"
  "--wandb.enable=false"
)
if [[ -n "${DATASET_ROOT:-}" ]]; then
  ARGS+=("--dataset.root=$DATASET_ROOT")
fi
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/check_dataset.py" \
  --report "$PROJECT_DIR/reports/${RUN_NAME}_preflight.json" \
  --training-args "${ARGS[@]}" "$@"

echo "GPU: $CUDA_VISIBLE_DEVICES | Dataset: $DATASET_REPO_ID | Output: $RUN_OUTPUT"
exec "$TRAIN_BIN" "${ARGS[@]}" "$@"
