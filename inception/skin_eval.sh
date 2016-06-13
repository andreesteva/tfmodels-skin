bazel build inception/skin_eval
SKIN_DATA_DIR="/ssd/esteva/skindata4/nine-way-even"
TRAIN_DIR="/archive/esteva/experiments/skindata4/baseline/train"
EVAL_DIR="/archive/esteva/experiments/skindata4/baseline/eval"

bazel-bin/inception/skin_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${SKIN_DATA_DIR}" \
  --subset=validation \
  --num_examples=14712 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once

