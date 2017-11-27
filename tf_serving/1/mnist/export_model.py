"""This script will train and export a linear regression model into .pb, which 
will is to be served by tensorflow serving. 
"""

import os
import sys

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer('training_iteration', 300,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    sess = tf.InteractiveSession()

    x = tf.placeholder('float', shape=[None, 28*28*1])
    y_ = tf.placeholder('float', shape=[None, 10])
    w = tf.get_variable('w', shape = [28*28*1,10], initializer = tf.truncated_normal_initializer)
    b = tf.get_variable('b', shape = [10], initializer = tf.zeros_initializer)

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x, w) + b

    # ms_loss = tf.reduce_mean((y - y_)**2)
    ms_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(ms_loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # train_x = np.random.randn(1000, 3)
    # let the model learn the equation of y = x1 * 1 + x2 * 2 + x3 * 3
    # train_y = np.sum(train_x * np.array([1,2,3]) + np.random.randn(1000, 3) / 100, axis = 1).reshape(-1, 1)
    mnist=input_data.read_data_sets("../MNIST_data",one_hot=True)


    train_loss = []

    for _ in range(FLAGS.training_iteration):
        train_x, train_y = mnist.train.next_batch(50)
        loss, _ = sess.run([ms_loss, train_step], feed_dict={x: train_x, y_: train_y})
        train_loss.append(loss)
    print('Training error %g' % loss)
    print('testing accuracy %g' % sess.run(
        accuracy, feed_dict={x: mnist.test.images,
                             y_: mnist.test.labels}))

    print('Done training!')
    # Export model
    export_path_base = FLAGS.work_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'input': tensor_info_x},
          outputs={'output': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          'prediction':
              prediction_signature,
      },
      legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')

if __name__ == '__main__':
    tf.app.run()
