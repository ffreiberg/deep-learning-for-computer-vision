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


def minibatches(inputs_x, inputs_y, targets_x, targets_y, mbs, shuffle):
    idx = np.arange(len(inputs_x))
    if shuffle:
         np.random.shuffle(idx)
    for i in range(0, len(inputs_x) - mbs + 1, mbs):
         if shuffle:
             batch_idx = idx[i:i + mbs]
             label_index = []
             for index in batch_idx:
                 if np.all(inputs_y[index] == targets_y[index]):
                     label_index.append(index)
                 else:
                     index_iterator = index
                     while index_iterator < len(targets_x):
                         index_iterator += 1
                         if np.all(inputs_y[index] == targets_y[index_iterator]):
                             label_index.append(index_iterator)
                             break
                         else:
                             if index_iterator == len(targets_x) -1:
                                 index_iterator = 0
         #print(np.all(inputs_y[batch_idx] == targets_y[label_index]))
         yield inputs_x[batch_idx], targets_x[label_index]


def separate_test_and_training(inputs_x, inputs_y, targets_x, targets_y, size):
    idx = np.arange(len(inputs_x))
    np.random.shuffle(idx)
    test_index = idx[0:size]
    test_inputs_x = inputs_x[test_index]
    test_inputs_y = inputs_y[test_index]

    test_targets_x = targets_x[test_index]
    test_targets_y = targets_y[test_index]

    counter = 0
    while counter < size:
        if np.all(test_inputs_y[counter] == test_targets_y[counter]):
            pass
        else:
            new_counter = counter
            while new_counter < size:
                if new_counter == size-1:
                    new_counter = 0
                new_counter += 1
                if np.all(test_inputs_y[counter] == test_targets_y[new_counter]):
                    test_targets_y[counter] = test_targets_y[new_counter]
                    test_targets_x[counter] = test_targets_x[new_counter]
                    break
        counter += 1

    train_index = idx[size:]
    train_inputs_x = inputs_x[train_index]
    train_inputs_y = inputs_y[train_index]
    train_index = np.append(idx[size:], np.arange(len(inputs_x), len(targets_x)))
    train_targets_x = targets_x[train_index]
    train_targets_y = targets_y[train_index]

    return test_inputs_x, test_inputs_y, test_targets_x, test_targets_y, train_inputs_x, train_inputs_y, train_targets_x, train_targets_y



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

    #test_inputs_x, test_inputs_y, test_targets_x, test_targets_y, train_inputs_x, train_inputs_y, train_targets_x, train_targets_y = separate_test_and_training(x_cyr, y_cyr, x_lat, y_lat, 500)


    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])


    minibatchsize = 300


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

    #------------Wird aktuell nicht verwendet------------------
    h_flat = tf.reshape(h_trans_conv1, [-1, 28 * 28])
    W_fc1 = weight_variable([28 * 28, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)


    W_fc2 = weight_variable([1024, 784])
    b_fc2 = bias_variable([784])
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
    #----------------------------------------------------------

    loss = tf.losses.mean_squared_error(labels=y_, predictions=h_trans_conv1)
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(1):
            print("Epoch: ", e)
            for i, batch in enumerate(minibatches(x_cyr, y_cyr, x_lat, y_lat, minibatchsize, True)):
                batch_cyr, batch_lat = batch
                batch_cyr = batch_cyr.reshape((minibatchsize, 28, 28, 1))
                batch_lat = batch_lat.reshape((minibatchsize, 28, 28, 1))
                train_step.run(feed_dict={x: batch_cyr, y_: batch_lat})
            print(sess.run(loss, feed_dict={x: batch_cyr, y_: batch_lat}))

        output = sess.run(h_trans_conv1, feed_dict={x: x_cyr[0:300].reshape((300, 28,28,1))})


    plt.figure(1)
    plt.subplot(311)
    plt.imshow(output[0].reshape((28, 28)), cmap='Greys')
    plt.subplot(312)
    plt.imshow(x_lat[0].reshape((28, 28)), cmap='Greys')
    plt.subplot(313)
    plt.imshow(x_cyr[0].reshape((28, 28)), cmap='Greys')
    plt.show()

    #plt.imshow(output[0].reshape((28, 28)), cmap='Greys')
    #plt.show()


if __name__ == "__main__":
    main()