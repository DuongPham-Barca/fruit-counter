#!/usr/bin/env bash
# Train with the readable train_model.py entrypoint.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTPUT=${OUTPUT:-outputs/fruit_counter_net_r50_coco_scratch}
args=(--output "$OUTPUT")
if [[ -f "$OUTPUT/checkpoints/last.pt" ]]; then
  echo "Resuming from $OUTPUT/checkpoints/last.pt"
  args+=(--resume)
fi

echo "TensorBoard: tensorboard --logdir $OUTPUT/logs/tensorboard"
exec python train_model.py "${args[@]}" "$@"
