"""A binary to evaluate Inception on the two-class skin data test sets using SS curves and output plots..
"""

from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import math
import os.path
import time
from sklearn.metrics import average_precision_score, roc_curve, auc

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.skin_data import SkinData # imports FLAGS.data_dir

import lib
from lib.learning import vis_utils as vu
from lib.learning import clumping_utils as cu
from lib.learning.vis_derms import people_SS

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_dir',
                           '/media/esteva/ExtraDrive1/ThrunResearch/tf_experiments/nine-way/test',
                           """Directory where to plot results and write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir',
                           '/media/esteva/ExtraDrive1/ThrunResearch/tf_experiments/nine-way/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('labels_file',
                           '/media/esteva/ExtraDrive1/ThrunResearch/splits/nine-way/train',
                           """The file with the classnames listed in it.""")
tf.app.flags.DEFINE_string('test_name', 'Epidermal Test',
                           """The title of the plots to use, and what to save them as. """)
tf.app.flags.DEFINE_string('people', '',
                           """The dermtest results to include."""
                           """Options:"""
                           """edinburgh_epidermal[_action]"""
                           """edinburgh_epic_pigmented[_action]"""
                           """dermoscopy[_action]"""
                           )

#tf.app.flags.DEFINE_string('data_dir', '/tmp/mydata',
#                           """Path to the processed data, i.e. """
#                           """TFRecord of Example protos.""")


# Flags governing the data used for the test.
tf.app.flags.DEFINE_integer('num_examples', 14712,
                            """Number of examples to run.""")

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

# Placeholder for _eval_once to extract global step
GLOBAL_STEP = []

def _eval_once(saver, softmax_op, labels_op, num_classes):
    """Runs Eval once.

    Args:
    saver: Saver.
    softmax_op: The softmax op
    labels_op: The labels op
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
            GLOBAL_STEP.append(global_step)
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
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            correct_per_class = np.zeros(num_classes)
            count_per_class = np.zeros(num_classes)

            mapping = ['__unused_background_class__ __unused_background_class__']
            _mapping = [line.strip() for line in open(FLAGS.mapping_file).readlines()]
            mapping.extend(_mapping)
            val_classes = np.unique([m.split()[0] for m in mapping])
            num_val_classes = len(val_classes)
            correct_per_class = np.zeros(num_val_classes)
            count_per_class = np.zeros(num_val_classes)

            print('%s: starting evaluation on (%s) for %d images.' % (
                datetime.now(), FLAGS.test_name, FLAGS.num_examples))
            start_time = time.time()
            total_softmax = []
            all_labels = []
            while step < num_iter and not coord.should_stop():
                softmax, labels = sess.run([softmax_op, labels_op])
                all_labels.append(labels)
                if FLAGS.mapping_file:
                    softmax = cu.mergeProbabilities(softmax, mapping)
                total_softmax.append(softmax)

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = FLAGS.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                        'sec/batch)' % (datetime.now(), step, num_iter,
                                        examples_per_sec, sec_per_batch))
                    start_time = time.time()

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    return np.vstack(total_softmax), np.hstack(all_labels)


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

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        probabilities, labels = _eval_once(saver, softmax, labels, num_classes)
    return probabilities, labels


def get_ss_curve(labels, probs, people, title, show_legend=True, textcolor='#4D5B66'):
    """Returns the SS curve figure handle and prints best results for the data."""

    fpr, tpr, thresholds = roc_curve(labels, probs)
    tnr = 1 - fpr
    ss = np.concatenate((tpr.reshape((-1,1)), tnr.reshape((-1,1)), thresholds.reshape((-1,1))), axis=1)
    best = np.argmax(np.mean(ss[:, :2], axis=1))
    best_sens = ss[best, 0]
    best_spec = ss[best, 1]
    best_thresh = ss[best, 2]
    auc_ss = auc(fpr, tpr)

    print 'Best sens', best_sens
    print 'Best spec', best_spec
    print 'Best thresh', best_thresh
    fig, ax = vu.plotSSCurve(tpr, tnr, people=people, title=title, show_legend=show_legend,
            textcolor=textcolor, plot_results=False)
    return fig, ax


def main(_):

    # Build dataset object.
    num_classes = len([line for line in open(FLAGS.labels_file).readlines() if line.strip()])
    dataset = SkinData(subset='validation', num_classes=num_classes)
    assert dataset.data_files()

    # Read in synset and mapping.
    print('Using mapping file %s' % FLAGS.mapping_file)
    mapping = [line.strip().split()[0] for line in open(FLAGS.mapping_file).readlines()]
    mapping = np.unique(mapping)
    assert len(mapping) == 3, \
            'Mapping is supported for binary classification only, currently len(mapping)=%d' % len(mapping)

    mapping = [line.strip().split()[1] for line in open(FLAGS.mapping_file).readlines()]
    synset = [line.strip() for line in open(FLAGS.labels_file).readlines()]

    assert len(mapping) == len(synset), \
          'Length of mapping & synset do not match: %s, %s' % (FLAGS.mapping_file, FLAGS.labels_file)
    for i, (m,s) in enumerate(zip(mapping, synset)):
        assert m == s, 'Mapping issue entry %d, %s is not %s' % (i, m, s)

    # Extract probabilities from test set and keep the first FLAGS.num_examples images.
    # Tensorflow queues will just keep wrapping around the same shards.
    t = time.time()
    probabilities, labels = evaluate(dataset)
    probabilities = probabilities[:FLAGS.num_examples]
    labels = labels[:FLAGS.num_examples]

    print 'Elapsed Time: %0.2f sec.' % (time.time() - t)

    # Eliminate background class.
    p = probabilities.copy()
    p = p[:, 1:]

    # Extract malignant probabilities into 'p'.
    p /= np.sum(p, axis=1).reshape((-1,1))
    p = p[:,1]

    # Calculate the SS curve.
    l = labels - 1
    if FLAGS.people:
        people = people_SS(FLAGS.people)
        figname = os.path.join(FLAGS.test_dir,
                FLAGS.test_name.replace(' ', '_') + '_' + FLAGS.people +
                '-ckpt-' + str(GLOBAL_STEP[-1]) + '.svg')
    else:
        figname = os.path.join(FLAGS.test_dir,
                FLAGS.test_name.replace(' ', '_') +
                '-ckpt-' + str(GLOBAL_STEP[-1]) + '.svg')
        people = []
    fig, ax = get_ss_curve(l, p, people, FLAGS.test_name)

    # Draw the SS curve to file.
    fig.savefig(figname)

if __name__ == '__main__':
    tf.app.run()
