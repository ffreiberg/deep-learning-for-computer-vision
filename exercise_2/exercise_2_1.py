import h5py
import numpy as np
import tensorflow as tf
import logging

def load_data(file):

    with h5py.File(file) as hf:
        x_train = normalize_data(np.array(hf.get(tr_x)).astype(np.float32))
        y_train = np.array(hf.get(tr_y))
        x_test = normalize_data(np.array(hf.get(te_x)).astype(np.float32))
        y_test = np.array(hf.get(te_y))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    return x_train, y_train, x_test, y_test


def normalize_data(data):

    data /= data.max()
    data -= data.mean()

    return data


def one_hot(data, num_classes):

    data = np.eye(num_classes)[data.flatten()]

    return data


def main(file):

    epochs = 1000

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file)
    logger.info('loading finished')

    y_tr = one_hot(y_tr, 10)
    y_te = one_hot(y_te, 10)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.matmul(x, w) + b

    loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(tf.nn.softmax(pred)), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for i in range(epochs):
        sess.run(train_step, feed_dict={x: x_tr, y: y_tr})
        if i % 100 == 0:
            _loss = sess.run(loss, feed_dict={x: x_tr, y: y_tr})
            print(_loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print('Accuracy on test set: {0:.2f}%'.format(sess.run(acc, feed_dict={x: x_te, y: y_te}) * 100))

if __name__ == '__main__':

    logger = logging.getLogger('ex2')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


    te_x = 'test_x'
    te_y = 'test_y'
    tr_x = 'train_x'
    tr_y = 'train_y'

    _file = 'pmjt_sample_20161116/train_test_file_list.h5'

    main(file=_file)