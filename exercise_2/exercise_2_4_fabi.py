import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_data, _file


def reshape(data):
    return np.reshape(data, [-1, int(np.sqrt(data.shape)), int(np.sqrt(data.shape)), 1])


def weight(shape, name='', trainable=True):
    if trainable:
        initial_weight = tf.truncated_normal(shape, stddev=.1)
    else:
        initial_weight = np.load(_params_file + '.npz')[name]
    return tf.Variable(initial_weight, trainable=trainable)


def bias(shape, name='', trainable=True):
    if trainable:
        initial_bias = tf.constant(.1, shape=shape)
    else:
        initial_bias = np.load(_params_file + '.npz')[name]
    return tf.Variable(initial_bias, trainable=trainable)


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
    lambda_ = .5
    filename = 'output_layer'

    # initialize variable input
    x = tf.Variable(tf.constant(.0, shape=[1, 28, 28, 1]), dtype=tf.float32)

    w_conv1 = weight([5, 5, 1, 32], 'w_conv1', False)
    b_conv1 = bias([32], 'b_conv1', False)
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight([5, 5, 32, 64], 'w_conv2', False)
    b_conv2 = bias([64], 'b_conv2', False)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_fc = weight([7 * 7 * 64, 1024], 'w_fc', False)
    b_fc = bias([1024], 'b_fc', False)
    h_pool_fc = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc = tf.nn.relu(tf.matmul(h_pool_fc, w_fc) + b_fc)

    w_out = weight([1024, 10], 'w_out', False)
    b_out = bias([10], 'b_out', False)

    pred = tf.matmul(h_fc, w_out) + b_out

    loss = lambda_ * tf.reduce_mean(x)**2 - pred
    # optimize for activation of first unit in output layer
    train = tf.train.GradientDescentOptimizer(eta).minimize(loss[0][0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        img = sess.run(x).reshape((28,28))
        plt.subplot(2, 2, 1)
        plt.title('before')
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')

        for i in range(100):
            sess.run(train)

        img2 = sess.run(x).reshape((28, 28))
        plt.subplot(2, 2, 2)
        plt.title('after')
        plt.imshow(img2, cmap='Greys_r')
        plt.axis('off')
        plt.savefig('maximum_activation_neuron_in_{}.png'.format(filename))

if __name__ == '__main__':

    _params_file='all_params'

    logger = logging.getLogger('ex2_4')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main(file=_file)