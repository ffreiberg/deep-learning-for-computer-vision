import h5py
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt

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
    data -= data.mean()
    data /= data.std()

    return data


def one_hot(data, num_classes):

    data = np.eye(num_classes)[data.flatten()]

    return data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(file):

    epochs = 5

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file)
    logger.info('loading finished')

    y_tr = one_hot(y_tr, 10)
    y_te = one_hot(y_te, 10)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    #-------------first convolution---------------------
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_tr_conv = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_tr_conv, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    #-------------second convolution---------------------
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #-------------fully connected 1-----------------------
    W_fully1 = weight_variable([7 * 7 * 64, 1024])
    b_fully1 = bias_variable([1024])

    h_pool2flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fully = tf.nn.relu(tf.matmul(h_pool2flat, W_fully1) + b_fully1)

    #-------------Dropout---------------------
    keep_prob = tf.placeholder(tf.float32)
    h_fully_drop = tf.nn.dropout(h_fully, keep_prob)

    #------------fully conected 2-------------------------
    W_fully2 = weight_variable([1024, 10])
    b_fully2 = bias_variable([10])

    pred = tf.matmul(h_fully_drop, W_fully2) + b_fully2


    #---------------training------------------------------
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    mini_batch_size = 200
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            # connect trainig data and labels for shuffeling and seperate to minibatches
            training_data = np.column_stack((x_tr, y_tr))
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, x_tr.shape[0], mini_batch_size)]

            # iterate through minibatches and apply gradient descent with Adam Optimizer
            for mini_batch in mini_batches:
                xtr = mini_batch[:,np.arange(784)]
                ytr = mini_batch[:,-10:]

                train_step.run(feed_dict={x: xtr, y_: ytr, keep_prob: 0.5})


            train_accuracy = accuracy.eval(feed_dict={x: xtr, y_: ytr, keep_prob: 1.0})
            print('Epoch %d, training accuracy %g %%' % (i, train_accuracy * 100))


        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: x_te, y_: y_te, keep_prob: 1.0}))

        # only for me to testing
        for i in range(10):
            print(y_te[i])
            test = np.argmax(sess.run(pred, feed_dict={
                x: x_te[i].reshape((1,784)), y_: y_te[i].reshape((1,10)), keep_prob: 1.0}))
            print(test)
            print(np.argmax(y_te[i]))
            print("-------------------------------")
            #plt.imshow(x_te[i].reshape((28, 28)))
            #plt.show()


if __name__ == '__main__':

    logger = logging.getLogger('ex2')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    te_x = 'test_x'
    te_y = 'test_y'
    tr_x = 'train_x'
    tr_y = 'train_y'

    _file = 'train_test_file_list.h5'

    main(file=_file)