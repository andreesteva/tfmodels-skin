# Must be run in a local workspace
bazel build inception/skin_train
bazel build inception/skin_eval
bazel build inception/build_image_data
