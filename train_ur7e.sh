#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?請設定 DATASET_ROOT 為資料集根目錄，例如 DATASET_ROOT=/path/to/dataset bash train_ur7e.sh}"
export DATASET_ROOT
export GPU="${GPU:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export STEPS="${STEPS:-20000}"
exec bash "$PROJECT_DIR/train.sh" local/ur7e_drawer \
  '--policy.input_features={"observation.state":{"type":"STATE","shape":[7]},"observation.images.left":{"type":"VISUAL","shape":[3,480,640]}}' \
  --policy.adapt_to_pi_aloha=false \
  --policy.use_delta_joint_actions_aloha=false \
  --log_freq=10 \
  "$@"
