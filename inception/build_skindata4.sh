TRAIN_DIR=/archive/esteva/skindata4/splits/nine-way/train-even
VALIDATION_DIR=/archive/esteva/skindata4/splits/nine-way/test
LABELS_FILE=/archive/esteva/skindata4/splits/nine-way/labels.txt
OUTPUT_DIRECTORY=/archive/esteva/tmp-shards/nine-way-even

# build the preprocessing script.
bazel build inception/build_image_data

# convert the data.
# Aim to have roughly 1024 images per shard.
# Also, num_threads needs to divide train_shards and validation_shards
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=672 \
  --validation_shards=16 \
  --num_threads=16

