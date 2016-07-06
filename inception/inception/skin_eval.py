# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to evaluate Inception on the skin data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.skin_data import SkinData

import lib
from lib.learning import clumping_utils as cu


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/archive/esteva/experiments/skindata4/baseline/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/archive/esteva/experiments/skindata4/baseline/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('labels_file', '/archive/esteva/skindata4/splits/nine-way/labels.txt',
                           """The file with the classnames listed in it.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 14712,
                            """Number of examples to run. Note that the connected componenets"""
                            """validation dataset contains 14712 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

tf.app.flags.DEFINE_string('mapping_file',
                           '',
                           """Defines a mapping from train to validation indicating how classes will be merged."""
                           """If this is specified, we sum probabilities to the validation class level."""
                           """Entries in this file must be of the form:
                                [validation-class-0] [training-class-0]
                                [validation-class-0] [training-class-1]
                                [validation-class-0] [training-class-2]
                                [validation-class-1] [training-class-3]
                                ...
                           """
                           )


def _eval_once(saver, summary_writer, top_1_op, softmax_op, labels_op, summary_op, num_classes):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    softmax_op: The softmax op
    labels_op: The labels op
    summary_op: Summary op.
    num_classes: the number of classes in the dataset (including background class)
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      correct_per_class = np.zeros(num_classes)
      count_per_class = np.zeros(num_classes)
      if FLAGS.mapping_file:
          mapping = ['__unused_background_class__ __unused_background_class']
          _mapping = [line.strip() for line in open(FLAGS.mapping_file).readlines()]
          mapping.extend(_mapping)
          val_classes = np.unique([m.split()[0] for m in mapping])
          num_val_classes = len(val_classes)
          correct_per_class = np.zeros(num_val_classes)
          count_per_class = np.zeros(num_val_classes)

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        top_1, softmax, labels = sess.run([top_1_op, softmax_op, labels_op])
        if FLAGS.mapping_file:
            softmax = cu.mergeProbabilities(softmax, mapping)
        preds = np.argmax(softmax, axis=1)
        for ll, p in zip(labels, preds):
          count_per_class[ll] += 1
          correct_per_class[ll] += ll == p
        count_top_1 += np.sum(top_1)
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Remove background class
      acc_per_class = 1.0 * correct_per_class / count_per_class
      acc_per_class = acc_per_class[1:]
      print(acc_per_class)
      print(correct_per_class)
      print(count_per_class)
      mean_accuracy = np.mean(acc_per_class)
      print('Mean Accuracy Per Class: %0.3f' % mean_accuracy)
      print('Per class accuracies:')
      if FLAGS.mapping_file:
          classnames = val_classes[1:]
      else:
          classnames = [line.strip() for line in tf.gfile.FastGFile(FLAGS.labels_file).readlines()]
      for e, name in zip(acc_per_class, classnames):
        print('%0.3f %s' % (e, name))

      # Compute precision @ 1.
      precision_at_1 = count_top_1 / total_sample_count
      print('%s: precision @ 1 = %.4f [%d examples]' %
            (datetime.now(), precision_at_1, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Mean Accuracy', simple_value=mean_accuracy)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes)
    softmax = tf.nn.softmax(logits)

    # Calculate predictions.
    top_1_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      _eval_once(saver, summary_writer, top_1_op, softmax, labels, summary_op, num_classes)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(unused_argv=None):
  num_classes = len([line for line in open(FLAGS.labels_file).readlines() if line.strip()])
  dataset = SkinData(subset=FLAGS.subset, num_classes=num_classes)
  assert dataset.data_files()
  if FLAGS.mapping_file:
      print('Using mapping file %s' % FLAGS.mapping_file)
      mapping = [line.strip().split()[1] for line in open(FLAGS.mapping_file).readlines()]
      synset = [line.strip() for line in open(FLAGS.labels_file).readlines()]
      assert len(mapping) == len(synset), \
              'Length of mapping & synset do not match: %s, %s' % (FLAGS.mapping_file, FLAGS.labels_file)
      for i, (m,s) in enumerate(zip(mapping, synset)):
          assert m == s, 'Mapping issue entry %d, %s is not %s' % (i, m, s)

  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
