bazel build inception/skin_train
MODEL_PATH="/ssd/esteva/pretrained_models/inception-v3/model.ckpt-157585"
SKIN_DATA_DIR="/ssd/esteva/skindata4/nine-way"
TRAIN_DIR="/archive/esteva/experiments/skin_train"

bazel-bin/inception/skin_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${SKIN_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
