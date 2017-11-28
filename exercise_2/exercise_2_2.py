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
    graphs = True
    num_classes = 10
    two_convs = True
    test_acc_list_one_conv = []
    test_acc_list_two_convs = []


    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    # for j in range(11):
    #     mbs = 2 ** j

    for i in range(10):

        train_acc_list_one_conv = []
        train_acc_list_two_convs = []

        eta = 1 / (10 ** i)

        w_conv1 = weight([5, 5, 1, 32])
        b_conv1 = bias([32])
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        for j in range(2):

            two_convs = j

            if two_convs:
                w_conv2 = weight([5, 5, 32, 64])
                b_conv2 = bias([64])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)
                w_fc = weight([7 * 7 * 64, 1024])
            else:
                w_fc = weight([14 * 14 * 32, 1024])
            b_fc = bias([1024])

            if two_convs:
                h_pool_fc = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            else:
                h_pool_fc = tf.reshape(h_pool1, [-1, 14 * 14 * 32])

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
                        if graphs:
                            train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y})
                            if two_convs:
                                train_acc_list_two_convs.append(train_acc * 100)
                            else:
                                train_acc_list_one_conv.append(train_acc * 100)
                        if i % 100 == 0:
                            train_acc = acc.eval(feed_dict={x: batch_x, y: batch_y})
                            logger.info('Epoch {}, step {} ({} / {} samples): accuracy: {:.2f}%'.format(e, i, mbs * i, x_tr.shape[0], train_acc * 100))
                        train.run(feed_dict={x: batch_x, y: batch_y})
                    end = time.time()
                    logger.info('epoch training took {:.3f}s'.format(end - begin))
                logger.info('test accuracy:: {:.2f}%'.format(acc.eval(feed_dict={x: x_te, y: y_te}) * 100))
                if graphs:
                    test_acc = acc.eval(feed_dict={x: x_te, y: y_te}) * 100
                    if two_convs:
                        test_acc_list_two_convs.append(test_acc)

                    else:
                        test_acc_list_one_conv.append(test_acc)

        if graphs:
            train_acc_arr = np.asarray([train_acc_list_one_conv, train_acc_list_two_convs])
            plt.title('eta: {} mbs: {}'.format(eta, mbs))
            plt.xlabel('training steps')
            plt.ylabel('accuracy')
            plt.ylim(0, 100)
            plt.plot(train_acc_arr[0], label='one conv')
            plt.plot(train_acc_arr[1], label='two convs')
            plt.legend()
            plt.savefig('training_data_eta_{}_mbs_{}.png'.format(eta, mbs))
            # plt.show()
            plt.close()

    if graphs:
        test_acc_arr = np.asarray([test_acc_list_one_conv, test_acc_list_two_convs])
        plt.title('accuracy on test set with mbs: {}'.format(mbs))
        plt.xlabel('eta in 10^-x')
        plt.ylabel('accuracy')
        plt.ylim(0, 100)
        plt.plot(test_acc_arr[0], label='one conv')
        plt.plot(test_acc_arr[1], label='two convs')
        plt.legend()
        plt.savefig('test_data_mbs_{}.png'.format(mbs))
        # plt.show()
        plt.close()


'''
Remove one convolution layer and describe by your own words how does it influence training and testing data.

- By removing one convolutional layer, in this case the second convolutional/deeper layer, our network is only able to learn filters
    of lower order like filters for edge detection. The more convolutional layers you have, the higher the order of
    learnt filters (and features) gets (e.g. corner detection filters within the second conv layer). As published in 
    'Visualizing and Understanding Convolutional Networks' by Matthew D. Zeiler and Rob Fergus.
    
    
Try different step sizes for the training. Explain what happens when the step-size is too large and too small. 
Draw graphs which how does the accuracy changes over iterations with different step sizes.

- If the step-size is too large global minima are likely to be missed out, because (you could say) you 'jump over' said minimum (and/or oscillate around a minimum). (As seen in 'training_data_eta_0.1_mbs_100.png')

- On the other hand, if your step-size is too small, you are likely to get stuck within a global minimum. And of course your training time
    increases.
'''
if __name__ == '__main__':

    logger = logging.getLogger('ex2_2')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main(file=_file)
