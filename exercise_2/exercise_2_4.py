import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_data, _file


def reshape(data):
    return np.reshape(data, [-1, int(np.sqrt(data.shape)), int(np.sqrt(data.shape)), 1])


def weight(shape):
    initial_weight = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial_weight, trainable=False)


def bias(shape):
    initial_bias = tf.constant(.1, shape=shape)
    return tf.Variable(initial_bias, trainable=False)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def minibatches(inputs, targets, mbs, shuffle):

    idx = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(inputs) - mbs + 1, mbs):
        if shuffle:
            batch_idx = idx[i:i + mbs]
        else:
            batch_idx = slice(i, idx)
        yield inputs[batch_idx], targets[batch_idx]


def main(file):

    eta = 1e-4
    lambda_ = 0.04
    mbs = 10
    epochs = 25
    graphs = True
    num_classes = 10
    two_convs = True
    test_acc_list_one_conv = []
    test_acc_list_two_convs = []


    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes)
    x_tr = x_tr[0]
    y_tr = y_tr[0]
    logger.info('loading finished')

    x = tf.Variable(tf.constant(x_tr, shape=(1, 28, 28, 1)))
    #x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    w_conv1 = weight([5, 5, 1, 32])
    b_conv1 = bias([32])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    w_conv2 = weight([5, 5, 32, 64])
    b_conv2 = bias([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    w_fc = weight([7 * 7 * 64, 1024])
    b_fc = bias([1024])

    h_pool_fc = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc = tf.nn.relu(tf.matmul(h_pool_fc, w_fc) + b_fc)

    w_out = weight([1024, 10])
    b_out = bias([10])

    pred = tf.matmul(h_fc, w_out) + b_out

    loss = lambda_ * tf.reduce_mean(x)**2 - pred[0][0]
    train = tf.train.GradientDescentOptimizer(eta).minimize(loss)

    with tf.Session() as sess:
        x = tf.Variable(tf.constant(x_tr, shape=(1, 28, 28, 1)))
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            print(sess.run(loss))
            sess.run(train, {x: sess.run(x)})
        plt.imshow(sess.run(x).reshape((28, 28)), cmap='Greys_r')
        plt.show()


if __name__ == '__main__':

    logger = logging.getLogger('ex2_2')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main(file=_file)