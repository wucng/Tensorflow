#!/usr/bin/env python2.7
# -*- coding: UTF-8 -*-
# -*- coding: utf8 -*-


r"""Train and export a simple Softmax Regression TensorFlow model.
The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    export_dir
"""

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

import mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/wu/pytest/MNIST/MNIST_data', 'Working directory.')
tf.app.flags.DEFINE_string('model_dir_ckpt', '/home/wu/pytest/MNIST/model', 'ckpt model directory.')

tf.app.flags.DEFINE_integer('flag', 1, '1 train ;0 test; -1 Export pb model .')

FLAGS = tf.app.flags.FLAGS


def main(_):
  """
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  """
  if FLAGS.training_iteration <= 0:
    print 'Please specify a positive value for training iteration.'
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print 'Please specify a positive value for version number.'
    sys.exit(-1)

  # Train model
  print 'Training model...'
  mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
  sess = tf.InteractiveSession()


  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  # Similar to
  # x=tf.placeholder('float',shape=[None,784],name='x')

  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  # sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
  # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in xrange(10)]))
  prediction_classes = table.lookup(tf.to_int64(indices)) # Predictive category label

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

  # ckpt model save
  saver = tf.train.Saver()

  # Check whether the checkpoint file has been saved before verifying
  if not os.path.exists(FLAGS.model_dir_ckpt):os.makedirs(FLAGS.model_dir_ckpt)
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir_ckpt)
  if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
  else:
      tf.global_variables_initializer().run()


  if FLAGS.flag==1: # train
      for step in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        if step%100==0:
            print ('training accuracy %g' % sess.run(
                accuracy, feed_dict={x: batch[0], y_: batch[1]}))
            saver.save(sess, os.path.join(FLAGS.model_dir_ckpt, 'model.ckpt'), global_step=step)

      print 'Done training!'
  if FLAGS.flag == 0:  # test
      print 'testing accuracy %g' % sess.run(
          accuracy, feed_dict={x: mnist.test.images,
                               y_: mnist.test.labels})
      print 'Done testing!'


  if FLAGS.flag == -1:  # Export pb model
      # Export model
      # WARNING(break-tutorial-inline-code): The following code snippet is
      # in-lined in tutorials, please update tutorial documents accordingly
      # whenever code changes.

      # export_path_base = sys.argv[-1]
      export_path_base ='/home/wu/pytest/MNIST/mnist_model'
      export_path = os.path.join(
          tf.compat.as_bytes(export_path_base),
          tf.compat.as_bytes(str(FLAGS.model_version)))
      print 'Exporting trained model to', export_path

      # Build a pb model
      builder = tf.saved_model.builder.SavedModelBuilder(export_path)

      # Build the signature_def_map.
      classification_inputs = tf.saved_model.utils.build_tensor_info(
          serialized_tf_example)
      classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
          prediction_classes)
      classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

      classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                      classification_inputs
              },
              outputs={
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                      classification_outputs_classes,
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                      classification_outputs_scores
              },
              method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

      tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
      tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': tensor_info_x},
              outputs={'scores': tensor_info_y},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

      legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  classification_signature,
          },
          legacy_init_op=legacy_init_op)

      builder.save()

      print 'Done exporting!'


if __name__ == '__main__':
  tf.app.run()
