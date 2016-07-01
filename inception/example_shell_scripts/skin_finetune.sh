bazel build inception/skin_train

# Original ImageNet Pre-trained Model (don't change this)
MODEL_PATH="/ssd/esteva/pretrained_models/inception-v3/model.ckpt-157585"

# Dataset to train on (make sure its buffer-filled. Should have *-even in the name)
SKIN_DATA_DIR="/ssd/esteva/skindata4/three-way-even"

# Directory to dump results into.
TRAIN_DIR="/archive/esteva/experiments/skindata4/three-way/train"


bazel-bin/inception/skin_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${SKIN_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
