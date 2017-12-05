import tensorflow as tf
import logging
import math
import numpy
from datetime import datetime

log = logging.getLogger(__name__)


from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


IMAGE_PIXELS=28

# Parameters
hidden_units = 128  # NN隐藏层
batch_size = 128  # 每批次训练的样本数


def feed_dict(batch):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
    xs = numpy.array(images)
    xs = xs.astype(numpy.float32)
    xs = xs / 255.0  # 数据归一化
    ys = numpy.array(labels)
    ys = ys.astype(numpy.uint8)
    return (xs, ys)

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),  # 5X5的卷积模板
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(IMAGE_PIXELS // 4) * (IMAGE_PIXELS // 4) * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([10]))
}

# Placeholders or QueueRunner/Readers for input data
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")  # mnist 28*28*1
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])  # mnist 数据 28x28x1 (灰度图 波段为1)
# tf.summary.image("x_img", x_img)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)   # strides中间两个为1 表示x,y方向都不间隔取样
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME') # strides中间两个为2 表示x,y方向都间隔1个取样

# 改成卷积模型
conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
conv1 = maxpool2d(conv1, k=2)
conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
conv2 = maxpool2d(conv2, k=2)
fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc1 = tf.nn.relu(fc1)
y = tf.nn.softmax(tf.add(tf.matmul(fc1, weights['out']), biases['out']))

'''
hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b) # tf.nn.add(tf.nn.matmul(x,hid_w),hid_b)
hid = tf.nn.relu(hid_lin)

y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
'''
# global_step = tf.Variable(0)

global_step = tf.Variable(0, name="global_step", trainable=False)

# loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# tf.summary.scalar("loss", loss)

train_op = tf.train.AdagradOptimizer(0.0001).minimize(
    loss, global_step=global_step)

# Test trained model
label = tf.argmax(y_, 1, name="label")
prediction = tf.argmax(y, 1, name="prediction")
correct_prediction = tf.equal(prediction, label)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
# tf.summary.scalar("acc", accuracy)

saver = tf.train.Saver()
# summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    logging.basicConfig(level=logging.INFO)
    sess.run(init_op)


    for _ in range(1000):
        # using feed_dict
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed = {x: batch_xs, y_: batch_ys}


        _, step = sess.run([train_op, global_step], feed_dict=feed)
        # print accuracy and save model checkpoint to HDFS every 100 steps
        if (step % 100 == 0):
            # print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step,
            #                                            sess.run(accuracy, {x: batch_xs, y_: batch_ys})))
            log.info("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step,
                                                          sess.run(accuracy, {x: batch_xs, y_: batch_ys})))
            # summary_writer.add_summary(summary, step)

            labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict={x:mnist.test.images,y_:mnist.test.labels})

            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l, p in
                       zip(labels, preds)]
            # tf_feed.batch_results(results)
            # print("acc: {0}".format(acc))
            log.info("acc: {0}".format(acc))
    save_path = saver.save(sess, "./logdir/checkpoint.ckpt")
    print('Saver path:', save_path)
