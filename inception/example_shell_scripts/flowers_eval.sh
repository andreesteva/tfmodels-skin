bazel build inception/flowers_eval
FLOWERS_DATA_DIR="/ssd/esteva/flowers-data"
TRAIN_DIR="/archive/esteva/experiments/flowers_train"
EVAL_DIR="/archive/esteva/experiments/flowers_eval"

bazel-bin/inception/flowers_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factfor=1 \
  --run_once
