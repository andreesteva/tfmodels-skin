TRAIN_DIR=/archive/esteva/skindata4/splits/nine-way/train-even
VALIDATION_DIR=/archive/esteva/skindata4/splits/nine-way/test
LABELS_FILE=/archive/esteva/skindata4/splits/nine-way/labels.txt
OUTPUT_DIRECTORY=/ssd/esteva/tmp-shards/nine-way-even

# build the preprocessing script.
bazel build inception/build_image_data

# convert the data.
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8
