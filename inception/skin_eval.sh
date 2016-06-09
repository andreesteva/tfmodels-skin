bazel build inception/skin_eval
SKIN_DATA_DIR="/ssd/esteva/skindata4/nine-way"
TRAIN_DIR="/archive/esteva/experiments/skin_train"
EVAL_DIR="/archive/esteva/experiments/skin_eval"

bazel-bin/inception/skin_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${SKIN_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once

