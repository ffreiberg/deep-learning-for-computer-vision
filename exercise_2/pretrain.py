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
    return tf.Variable(initial_weight)


def bias(shape):
    initial_bias = tf.constant(.1, shape=shape)
    return tf.Variable(initial_bias)


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
    mbs = 10
    epochs = 25
    num_classes = 10
    _params_file = 'all_params'

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
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

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    train = tf.train.AdamOptimizer(eta).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            begin = time.time()
            for i, b in enumerate(minibatches(x_tr, y_tr, mbs, shuffle=True)):
                batch_x, batch_y = b
                if i % 100 == 0:
                    train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y})
                    logger.info('Epoch {}, step {} ({} / {} samples): accuracy: {:.2f}%'.format(e, i, mbs * i,
                                                                                                x_tr.shape[0],
                                                                                                train_acc * 100))
                train.run(feed_dict={x: batch_x, y: batch_y})
            end = time.time()
            logger.info('epoch training took {:.3f}s'.format(end - begin))
        logger.info('test accuracy:: {:.2f}%'.format(acc.eval(feed_dict={x: x_te, y: y_te}) * 100))

    _w_conv1, _b_conv1, _w_conv2, _b_conv2, _w_fc, _b_fc, _w_out, _b_out = sess.run([w_conv1, b_conv1, w_conv2, b_conv2, w_fc, b_fc])
    np.savez(_params_file, w_conv1=_w_conv1, b_conv1=_b_conv1, w_conv2=_w_conv2, b_conv2=_b_conv2, w_fc=_w_fc, b_fc=_b_fc, w_out=_w_out, b_out=_b_out)

if __name__ == '__main__':
    logger = logging.getLogger('pretrain')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main(file=_file)
