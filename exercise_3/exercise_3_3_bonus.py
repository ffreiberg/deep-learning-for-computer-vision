import time
import logging
import numpy as np
import tensorflow as tf
from load_data import load_data, _file, load_cifar10

#only works if 0 < alpha < 1 (what alpha should be)
def leaky_relu(x, alpha):
    return tf.maximum(x, tf.abs(alpha) * x)


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


def conv2d(name, x, shape, strides, a, is_training, bn=True):
    with tf.variable_scope('l_' + name):
        w_conv = tf.get_variable('w_' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
        o_conv = tf.nn.conv2d(x, w_conv, strides=strides, padding='SAME')
        if bn:
            bn_conv = tf.layers.batch_normalization(o_conv, center=True, scale=True, training=is_training)
            h_conv = leaky_relu(bn_conv, a)
        else:
            h_conv = leaky_relu(o_conv, a)
    return h_conv


def fc(name, x, shape, new_shape,a, is_training):
    with tf.variable_scope('l_' + name):
        w_fc = tf.get_variable('w_' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
        if new_shape != None:
            h_pool_fc = tf.reshape(x, new_shape)
            o_fc = tf.matmul(h_pool_fc, w_fc)
        else:
            o_fc = tf.matmul(x, w_fc)
        bn_fc = tf.layers.batch_normalization(o_fc, center=True, scale=True, training=is_training)
        h_fc = leaky_relu(bn_fc, a)
    return h_fc


def ex_bonus(cifar10=True):

    eta = 1e-4
    mbs = 100
    epochs = 50
    num_classes = 10

    strides = [1, 1, 1, 1]

    logger.info('Loading CIFAR-10...')
    x_tr, y_tr, x_te, y_te = load_cifar10()
    logger.info('Finished loading')

    x = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
    x_in = tf.reshape(x, [-1, 32, 32, 3])

    y = tf.placeholder(tf.float32, [None, 10])
    a = tf.Variable(initial_value=.05, dtype=tf.float32)

    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # with tf.variable_scope('l_conv1'):
    #     w_conv1 = tf.get_variable('w_conv1', [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    #     o_conv1 = tf.nn.conv2d(x_in, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    #     bn_conv1 = tf.layers.batch_normalization(o_conv1, center=True, scale=True, training=is_training)
    #     h_conv1 = leaky_relu(bn_conv1, a)

    h_conv1_1 = conv2d('conv1_1', x_in, [3, 3, 3, 64], strides, a, is_training)

    h_conv1_2 = conv2d('conv1_2', h_conv1_1, [3, 3, 64, 64], strides, a, is_training)

    # size after pooling 16 x 16

    h_pool1 = tf.nn.max_pool(h_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('l_conv2'):
    #     w_conv2 = tf.get_variable('w_conv2', [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    #     o_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
    #     bn_conv2 = tf.layers.batch_normalization(o_conv2, center=True, scale=True, training=is_training)
    #     h_conv2 = leaky_relu(bn_conv2, a)

    h_conv2_1 = conv2d('conv2_1', h_pool1, [3, 3, 64, 128], strides, a, is_training)

    h_conv2_2 = conv2d('conv2_2', h_conv2_1, [3, 3, 128, 128], strides, a, is_training)

    #size after pooling 8 x 8

    h_pool2 = tf.nn.max_pool(h_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('l_conv3'):
    #     w_conv3 = tf.get_variable('w_conv3', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    #     o_conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
    #     bn_conv3 = tf.layers.batch_normalization(o_conv3, center=True, scale=True, training=is_training)
    #     h_conv3 = leaky_relu(bn_conv3, a)

    h_conv3_1 = conv2d('conv3_1', h_pool2, [3, 3, 128, 256], strides, a, is_training)
    h_conv3_2 = conv2d('conv3_2', h_conv3_1, [3, 3, 256, 256], strides, a, is_training)
    h_conv3_3 = conv2d('conv3_3', h_conv3_2, [3, 3, 256, 256], strides, a, is_training)
    h_conv3_4 = conv2d('conv3_4', h_conv3_3, [3, 3, 256, 256], strides, a, is_training)

    # size after pooling 4 x 4

    h_pool3 = tf.nn.max_pool(h_conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_conv4_1 = conv2d('conv4_1', h_pool3, [3, 3, 256, 512], strides, a, is_training)
    h_conv4_2 = conv2d('conv4_2', h_conv4_1, [3, 3, 512, 512], strides, a, is_training)
    h_conv4_3 = conv2d('conv4_3', h_conv4_2, [3, 3, 512, 512], strides, a, is_training)
    h_conv4_4 = conv2d('conv4_4', h_conv4_3, [3, 3, 512, 512], strides, a, is_training)

    # size after pooling 2 x 2

    h_pool4 = tf.nn.max_pool(h_conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_conv5_1 = conv2d('conv5_1', h_pool4, [3, 3, 512, 512], strides, a, is_training)
    h_conv5_2 = conv2d('conv5_2', h_conv5_1, [3, 3, 512, 512], strides, a, is_training)
    h_conv5_3 = conv2d('conv5_3', h_conv5_2, [3, 3, 512, 512], strides, a, is_training)
    h_conv5_4 = conv2d('conv5_4', h_conv5_3, [3, 3, 512, 512], strides, a, is_training)

    # size after pooling 1 x 1

    h_pool5 = tf.nn.max_pool(h_conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('l_fc1'):
    #     w_fc = tf.get_variable('w_fc1', [4 * 4 * 128, 1024], initializer=tf.contrib.layers.xavier_initializer())
    #     h_pool_fc = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
    #     o_fc = tf.matmul(h_pool_fc, w_fc)
    #     bn_fc = tf.layers.batch_normalization(o_fc, center=True, scale=True, training=is_training)
    #     h_fc = leaky_relu(bn_fc, a)

    h_fc1 = fc('fc1', h_pool5, [1 * 1 * 512, 1024], [-1, 1 * 1 * 512], a, is_training)

    h_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # with tf.variable_scope('l_fc2'):
    #     w_fc2 = tf.get_variable('w_fc2', [1024, 384], initializer=tf.contrib.layers.xavier_initializer())
    #     o_fc2 = tf.matmul(h_dropout, w_fc2)
    #     bn_fc2 = tf.layers.batch_normalization(o_fc2, center=True, scale=True, training=is_training)
    #     h_fc2 = leaky_relu(bn_fc2, a)

    h_fc2 = fc('fc2', h_dropout, [1024, 384], None, a, is_training)

    h_dropout2 = tf.nn.dropout(h_fc2, keep_prob=keep_prob)

    with tf.variable_scope('out'):
        w_out = tf.get_variable('w_out', [384, 10], initializer=tf.contrib.layers.xavier_initializer())
        b_out = tf.get_variable('b_out', [10], initializer=tf.constant_initializer(.1))

    pred = tf.matmul(h_dropout2, w_out) + b_out

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    with tf.control_dependencies(update_ops):
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
                if i % 100 == 0:
                    train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y, is_training:True, keep_prob:0.5})
                    train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y, is_training:True, keep_prob:0.5})
                    logger.info(
                        'Epoch {}, step {} ({} / {} samples): \tloss: {:.6f}\taccuracy: {:.2f}%'.format(e + 1, i, mbs * i,
                                                                                                        x_tr.shape[0],
                                                                                                        train_loss,
                                                                                                        train_acc * 100))
                train.run(feed_dict={x: batch_x, y: batch_y, is_training:True, keep_prob:0.5})
            end = time.time()
            logger.info('epoch training took {:.3f}s'.format(end - begin))

        test_acc = 0
        for b in minibatches(x_te, y_te, mbs, shuffle=True):
            batch_x, batch_y = b
            _acc = acc.eval(feed_dict={x: batch_x, y: batch_y, is_training:False, keep_prob:1.})
            test_acc += _acc

        test_acc *= (100 / (len(y_te) / mbs))

        # test_loss, test_acc = sess.run([loss, acc], feed_dict={x: x_te, y: y_te, is_training:False, keep_prob:1.})
        # logger.info('test loss: {:.6f}\t accuracy: {:.2f}%'.format(test_loss, test_acc * 100))
        logger.info('test accuracy: {:.2f}%'.format(test_acc))


if __name__ == '__main__':
    logger = logging.getLogger('ex3_3_bonus')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    import datetime
    fh = logging.FileHandler('bonus_{}.log'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')))
    logger.addHandler(fh)

    seed = 1337

    np.random.seed(seed=seed)

    ex_bonus()



'''
def ex_bonus():

    eta = 1e-4
    mbs = 100
    epochs = 25
    num_classes = 10

    # logger.info('Loading CIFAR-10...')
    # x_tr, y_tr, x_te, y_te = load_cifar10()
    # logger.info('Finished loading')

    logger.info('loading data from file  {}'.format(_file))
    x_tr, y_tr, x_te, y_te = load_data(_file, num_classes)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    a = tf.Variable(initial_value=.05, dtype=tf.float32)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    w_conv1 = weight([3, 3, 1, 32])
    o_conv1 = conv2d(x, w_conv1)
    bn_conv1 = tf.layers.batch_normalization(o_conv1, center=True, scale=True, training=is_training)
    h_conv1 = leaky_relu(bn_conv1, a)

    w_conv2 = weight([3, 3, 32, 64])
    o_conv2 = conv2d(h_conv1, w_conv2)
    bn_conv2 = tf.layers.batch_normalization(o_conv2, center=True, scale=True, training=is_training)
    h_conv2 = leaky_relu(bn_conv2, a)
    h_pool2 = max_pool_2x2(h_conv2)

    w_conv3 = weight([3, 3, 64, 128])
    o_conv3 = conv2d(h_pool2, w_conv3)
    bn_conv3 = tf.layers.batch_normalization(o_conv3, center=True, scale=True, training=is_training)
    h_conv3 = leaky_relu(bn_conv3, a)
    h_pool3 = max_pool_2x2(h_conv3)

    w_fc = weight([7 * 7 * 128, 1024])
    h_pool_fc = tf.reshape(h_pool3, [-1, 7 * 7 * 128])
    o_fc = tf.matmul(h_pool_fc, w_fc)
    # bn_fc = tf.contrib.layers.batch_norm(o_fc, center=True, scale=True, is_training=is_training)
    bn_fc = tf.layers.batch_normalization(o_fc, center=True, scale=True, training=is_training)
    h_fc = leaky_relu(bn_fc, a)

    h_dropout = tf.nn.dropout(h_fc, keep_prob=keep_prob)

    w_out = weight([1024, 10])
    b_out = bias([10])

    pred = tf.matmul(h_dropout, w_out) + b_out

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    with tf.control_dependencies(update_ops):
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
                if i % 100 == 0:
                    train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y, is_training:True, keep_prob:0.5})
                    train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y, is_training:True, keep_prob:0.5})
                    logger.info(
                        'Epoch {}, step {} ({} / {} samples): \tloss: {:.6f}\taccuracy: {:.2f}%'.format(e, i, mbs * i,
                                                                                                        x_tr.shape[0],
                                                                                                        train_loss,
                                                                                                        train_acc * 100))
                train.run(feed_dict={x: batch_x, y: batch_y, is_training:True, keep_prob:0.5})
            end = time.time()
            logger.info('epoch training took {:.3f}s'.format(end - begin))
        test_loss, test_acc = sess.run([loss, acc], feed_dict={x: x_te, y: y_te, is_training:False, keep_prob:1.})
        logger.info('test loss: {:.6f}\t accuracy: {:.2f}%'.format(test_loss, test_acc * 100))

'''