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
"""A binary to train Inception on the skin data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import os
import numpy as np
import tensorflow as tf

from inception import inception_train
from inception.skin_data import SkinData

tf.app.flags.DEFINE_string('labels_file', '/archive/esteva/skindata4/splits/nine-way/labels.txt',
                           """The file with the classnames listed in it.""")

FLAGS = tf.app.flags.FLAGS


def main(_):
# dataset = SkinData(subset=FLAGS.subset)
  num_classes = len([line for line in open(FLAGS.labels_file).readlines() if line.strip()])
  dataset = SkinData(subset=FLAGS.subset, num_classes=num_classes)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.train_dir):
    files = glob.glob(os.path.join(FLAGS.train_dir, 'model.ckpt-*'))
    if len(files) > 0:
      last_iter = np.sort([int(f.split('-')[1].split('.')[0]) for f in files])[-1]
      FLAGS.current_step = last_iter
      FLAGS.pretrained_model_checkpoint_path = os.path.join(
              FLAGS.train_dir, 'model.ckpt-%d' % last_iter)
      FLAGS.fine_tune = False
      print('Continuing training from %s' % FLAGS.pretrained_model_checkpoint_path)
  else:
    tf.gfile.MakeDirs(FLAGS.train_dir)
  inception_train.train(dataset)


if __name__ == '__main__':
  tf.app.run()
