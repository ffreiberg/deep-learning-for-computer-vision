import time
import logging
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_data import load_data, _file


#only works if 0 < alpha < 1 (what alpha should be)
def leaky_relu(x, alpha):
    return tf.maximum(x, tf.abs(alpha) * x)


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


'''
test accuracy:: 95.19%
Optimum value for alpha after training: 0.00016570596199017018
'''
def ex_prelu(file):
    eta = 1e-4
    mbs = 100
    epochs = 25
    graphs = True
    prelu = True
    num_classes = 10
    test_acc_list = []

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    if prelu:
        a = tf.Variable(initial_value=.05, dtype=tf.float32)
        alphas = [a]
        graphs = False
    else:
        alphas = [.9, .5, .25, .1, .05, .01, .005, .001]

    # loop over different learning rates to find the best one
    for alpha in alphas:

        train_acc_list = []
        train_loss_list = []

        # first conv layer
        w_conv1 = weight([5, 5, 1, 32])
        b_conv1 = bias([32])
        h_conv1 = leaky_relu(conv2d(x, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        w_conv2 = weight([5, 5, 32, 64])
        b_conv2 = bias([64])
        h_conv2 = leaky_relu(conv2d(h_pool1, w_conv2) + b_conv2, alpha)
        h_pool2 = max_pool_2x2(h_conv2)
        w_fc = weight([7 * 7 * 64, 1024])
        b_fc = bias([1024])

        h_pool_fc = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        h_fc = leaky_relu(tf.matmul(h_pool_fc, w_fc) + b_fc, alpha)

        w_out = weight([1024, 10])
        b_out = bias([10])

        pred = tf.matmul(h_fc, w_out) + b_out

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        train = tf.train.AdamOptimizer(eta).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # training and evaluation
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):
                begin = time.time()
                for i, b in enumerate(minibatches(x_tr, y_tr, mbs, shuffle=True)):
                    batch_x, batch_y = b
                    if graphs:
                        train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y})
                        train_acc_list.append(train_acc * 100)
                        train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                        train_loss_list.append(train_loss)
                    if i % 100 == 0:
                        train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y})
                        train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                        logger.info('Epoch {}, step {} ({} / {} samples): \tloss: {:.6f}\taccuracy: {:.2f}%'.format(e, i, mbs * i,
                                                                                                    x_tr.shape[0],
                                                                                                    train_loss,
                                                                                                    train_acc * 100))
                    train.run(feed_dict={x: batch_x, y: batch_y})
                end = time.time()
                logger.info('epoch training took {:.3f}s'.format(end - begin))
            logger.info('test accuracy:: {:.2f}%'.format(acc.eval(feed_dict={x: x_te, y: y_te}) * 100))
            alpha_after = sess.run(a)
            logger.info('Optimum value for alpha after training: {}'.format(alpha_after))
            if graphs:
                test_acc = acc.eval(feed_dict={x: x_te, y: y_te}) * 100
                test_acc_list.append(test_acc)

        if graphs:
            train_acc_arr = np.asarray(train_acc_list)
            plt.title('alpha: {} mbs: {}'.format(alpha, mbs))
            plt.xlabel('training steps')
            plt.ylabel('accuracy')
            plt.ylim(0, 100)
            plt.plot(train_acc_arr)
            plt.savefig('leaky_relu_training_data_acc_alpha_{}_mbs_{}.png'.format(alpha, mbs))
            # plt.show()
            plt.close()

            train_loss_arr = np.asarray(train_loss_list)
            plt.title('alpha: {} mbs: {}'.format(alpha, mbs))
            plt.xlabel('training steps')
            plt.ylabel('loss')
            plt.plot(train_loss_arr)
            plt.savefig('leaky_relu_training_data_loss_alpha_{}_mbs_{}.png'.format(alpha, mbs))
            # plt.show()
            plt.close()

    if graphs:
        test_acc_arr = np.asarray(test_acc_list)
        plt.title('accuracy on test set with mbs: {}'.format(mbs))
        plt.xticks(np.arange(len(alphas)), alphas)
        plt.ylabel('accuracy')
        plt.xlabel('alpha values')
        plt.ylim(90, 100)
        plt.plot(test_acc_arr)
        plt.savefig('leaky_relu_test_data_mbs_{}.png'.format(mbs))
        # plt.show()
        plt.close()


def ex_batch_norm(file):
    eta = 1e-4
    mbs = 100
    epochs = 25
    num_classes = 10
    train_acc_list = []
    train_loss_list = []

    graphs = True

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    a = tf.Variable(initial_value=.05, dtype=tf.float32)
    is_training = tf.placeholder(tf.bool)


    w_conv1 = weight([5, 5, 1, 32])
    b_conv1 = bias([32])
    o_conv1= conv2d(x, w_conv1) + b_conv1
    bn_conv1 = tf.contrib.layers.batch_norm(o_conv1, center=True, scale=True, is_training=is_training)
    h_conv1 = leaky_relu(bn_conv1, a)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight([5, 5, 32, 64])
    b_conv2 = bias([64])
    o_conv2 = conv2d(h_pool1, w_conv2) + b_conv2
    bn_conv2 = tf.contrib.layers.batch_norm(o_conv2, center=True, scale=True, is_training=is_training)
    h_conv2 = leaky_relu(bn_conv2, a)
    h_pool2 = max_pool_2x2(h_conv2)

    w_fc = weight([7 * 7 * 64, 1024])
    b_fc = bias([1024])
    h_pool_fc = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    o_fc = tf.matmul(h_pool_fc, w_fc) + b_fc
    bn_fc = tf.contrib.layers.batch_norm(o_fc, center=True, scale=True, is_training=is_training)
    h_fc = leaky_relu(bn_fc, a)

    w_out = weight([1024, 10])
    b_out = bias([10])

    pred = tf.matmul(h_fc, w_out) + b_out

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    train = tf.train.AdamOptimizer(eta).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # training and evaluation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            begin = time.time()
            for i, b in enumerate(minibatches(x_tr, y_tr, mbs, shuffle=True)):
                batch_x, batch_y = b
                if graphs:
                    train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y, is_training:True})
                    train_acc_list.append(train_acc * 100)
                    train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y, is_training:True})
                    train_loss_list.append(train_loss)
                if i % 100 == 0:
                    train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y, is_training:True})
                    train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y, is_training:True})
                    logger.info(
                        'Epoch {}, step {} ({} / {} samples): \tloss: {:.6f}\taccuracy: {:.2f}%'.format(e, i, mbs * i,
                                                                                                        x_tr.shape[0],
                                                                                                        train_loss,
                                                                                                        train_acc * 100))
                train.run(feed_dict={x: batch_x, y: batch_y, is_training:True})
            end = time.time()
            logger.info('epoch training took {:.3f}s'.format(end - begin))
        logger.info('test accuracy:: {:.2f}%'.format(acc.eval(feed_dict={x: x_te, y: y_te, is_training:False}) * 100))
        alpha_after = sess.run(a)
        logger.info('Optimum value for alpha after training: {}'.format(alpha_after))

        if graphs:
            train_acc_arr = np.asarray(train_acc_list)
            plt.title('training accuracy mbs: {}'.format(mbs))
            plt.xlabel('training steps')
            plt.ylabel('accuracy')
            plt.ylim(0, 100)
            plt.plot(train_acc_arr)
            plt.savefig('leaky_relu_bn_training_data_acc_mbs_{}.png'.format(mbs))
            # plt.show()
            plt.close()

            train_loss_arr = np.asarray(train_loss_list)
            plt.title('training loss mbs: {}'.format(mbs))
            plt.xlabel('training steps')
            plt.ylabel('loss')
            plt.plot(train_loss_arr)
            plt.savefig('leaky_relu_bn_training_data_loss_mbs_{}.png'.format(mbs))
            # plt.show()
            plt.close()


if __name__ == '__main__':
    logger = logging.getLogger('ex3_3')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # ex_prelu(file=_file)
    ex_batch_norm(_file)