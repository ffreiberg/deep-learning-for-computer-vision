import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_data():
    x_lat = np.load('X_lat.npy').transpose([2, 0, 1])
    y_lat = np.load('labels_lat.npy')
    return x_lat, y_lat


def normalize(X):
    for i in range(X.shape[0]):
        # zero mean
        X[i, ...] = X[i, ...] - np.mean(X[i, ...].ravel())
        # unit std
        X[i, ...] = X[i, ...] / np.std(X[i, ...].ravel())
    return X


def one_hot(data, num_classes):
    data = np.eye(num_classes)[data.flatten()]
    return data


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


def separate_test_and_training(x, y):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x_test = x[idx[:500]]
    y_test = y[idx[:500]]

    x_train = x[idx[500:]]
    y_train = y[idx[500:]]

    return x_train, y_train, x_test, y_test


def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def conv2d_transpose(x, W, stride, outputshape):
    return tf.nn.conv2d_transpose(x, W, strides=stride, padding="SAME", output_shape=outputshape)


def main():
    x_lat, y_lat = load_data()
    x_lat = normalize(x_lat)
    y_lat = one_hot(y_lat, 14)

    x_train, y_train, x_test, y_test = separate_test_and_training(x_lat, y_lat)

    is_training = tf.placeholder(tf.bool, name='phase')

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 14])

    minibatchsize = 300

    #-------------encoding-----------------
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    conv1 = conv2d(x, W_conv1, [1, 2, 2, 1]) + b_conv1
    batch1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=is_training)
    h_conv1 = tf.nn.relu(batch1)


    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    conv2 = conv2d(h_conv1, W_conv2, [1, 4, 4, 1]) + b_conv2
    batch2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=is_training)
    h_conv2 = tf.nn.relu(batch2)


    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    conv3 = conv2d(h_conv2, W_conv3, [1, 4, 4, 1]) + b_conv3
    batch3 = tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=is_training)
    h_conv3 = tf.nn.relu(batch3)


    h_flat = tf.reshape(h_conv3, [-1, 128])

    W_fc1 = weight_variable([128, 1024])
    b_fc1 = bias_variable([1024])
    fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
    batch_fc1 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=is_training)
    h_fc1 = tf.nn.relu(batch_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 14])
    b_fc2 = bias_variable([14])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    #-----------Minimize MSE Loss-function with Adam-----------
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    loss = tf.losses.mean_squared_error(labels=y_, predictions=h_fc2)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(100):
            print("Epoch: ", e)
            for i, batch in enumerate(minibatches(x_train, y_train, minibatchsize, True)):
                batch_x, batch_y = batch
                batch_x = batch_x.reshape((minibatchsize, 28, 28, 1))
                train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5, is_training: True})
            print("Loss: ", sess.run(loss, feed_dict={x: x_train.reshape(x_train.shape[0], 28, 28, 1), y_: y_train, keep_prob: 1.0, is_training: True}))

            #validation loss
            print("Validation Loss: ", sess.run(loss, feed_dict={x: x_test.reshape(x_test.shape[0], 28, 28, 1), y_: y_test, keep_prob: 1.0, is_training: False}))

        #validation at the end
        output = sess.run(h_fc2, feed_dict={x: x_test.reshape((x_test.shape[0], 28, 28, 1)), y_: y_test, keep_prob: 1.0, is_training: False})
        output = np.argmax(output, axis=1)
        output = one_hot(output, 14)

        counter = 0
        for i in range(500):
            if np.all(output[i] == y_test[i]):
                counter += 1
        print("correct: ", (100 / 500) * counter, "%")



if __name__ == "__main__":
    main()