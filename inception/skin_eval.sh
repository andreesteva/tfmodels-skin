bazel build inception/skin_eval
SKIN_DATA_DIR="/ssd/esteva/skindata4/three-way-even"
TRAIN_DIR="/archive/esteva/experiments/skindata4/three-way/train"
EVAL_DIR="/archive/esteva/experiments/skindata4/three-way/eval"

bazel-bin/inception/skin_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${SKIN_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once
# --num_examples=14712 \

