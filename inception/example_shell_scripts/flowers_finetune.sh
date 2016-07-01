bazel build inception/flowers_train
MODEL_PATH="/ssd/esteva/pretrained_models/inception-v3/model.ckpt-157585"
FLOWERS_DATA_DIR="/ssd/esteva/flowers-data"
TRAIN_DIR="/archive/esteva/experiments/flowers_train"

bazel-bin/inception/flowers_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
