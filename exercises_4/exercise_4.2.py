import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_data():
    x_cyr = np.load('X_cyr.npy').transpose([2, 0, 1])
    x_lat = np.load('X_lat.npy').transpose([2, 0, 1])
    y_cyr = np.load('labels_cyr.npy')
    y_lat = np.load('labels_lat.npy')
    return x_cyr, y_cyr, x_lat, y_lat


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




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def conv2d_transpose(x, W, stride, outputshape):
    return tf.nn.conv2d_transpose(x, W, strides=stride, padding="SAME", output_shape=outputshape)


def main():
    x_cyr, y_cyr, x_lat, y_lat = load_data()
    x_cyr = normalize(x_cyr)
    x_lat = normalize(x_lat)
    y_cyr = one_hot(y_cyr, 14)
    y_lat = one_hot(y_lat, 14)

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 14])


    minibatchsize = 1000


    #-------------encoding-----------------
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    conv1 = conv2d(x, W_conv1, [1, 2, 2, 1]) + b_conv1
    h_conv1 = tf.nn.relu(conv1)


    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    conv2 = conv2d(h_conv1, W_conv2, [1, 4, 4, 1]) + b_conv2
    h_conv2 = tf.nn.relu(conv2)


    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    conv3 = conv2d(h_conv2, W_conv3, [1, 4, 4, 1]) + b_conv3
    h_conv3 = tf.nn.relu(conv3)



    #------------decoding-------------------
    W_trans_conv3 = weight_variable([3, 3, 64, 128])
    b_trans_conv3 = bias_variable([64])
    trans_conv3 = conv2d_transpose(h_conv3, W_trans_conv3, [1, 4, 4, 1], [minibatchsize, 4, 4, 64]) + b_trans_conv3
    h_trans_conv3 = tf.nn.relu(trans_conv3)


    W_trans_conv2 = weight_variable([3, 3, 32, 64])
    b_trans_conv2 = bias_variable([32])
    trans_conv2 = conv2d_transpose(h_trans_conv3, W_trans_conv2, [1, 4, 4, 1], [minibatchsize, 14, 14, 32]) + b_trans_conv2
    h_trans_conv2 = tf.nn.relu(trans_conv2)


    W_trans_conv1 = weight_variable([5, 5, 1, 32])
    b_trans_conv1 = bias_variable([1])
    trans_conv1 = conv2d_transpose(h_trans_conv2, W_trans_conv1, [1, 2, 2, 1], [minibatchsize, 28, 28, 1]) + b_trans_conv1
    h_trans_conv1 = tf.nn.relu(trans_conv1)

    loss = h_trans_conv1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(loss, {x: x_cyr[0:minibatchsize].reshape((minibatchsize,28,28,1))})
        pass


    #plt.imshow(a[0].reshape((28, 28)), cmap='Greys')
    #plt.show()


if __name__ == "__main__":
    main()