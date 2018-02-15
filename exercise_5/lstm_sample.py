""" Action recognition - Recurrent Neural Network

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This is pruned version of an original example https://github.com/aymericdamien/TensorFlow-Examples/ 
for MNIST letter classification

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Adapt this script for purpose of video sequence classification defined in exercise sheet

"""

from __future__ import print_function

import os
import time
import _pickle
import logging
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def transform_val_data(x, y, timesteps, num_classes):
    x_val = [np.reshape(np.transpose(x[i][..., 0:timesteps], (2, 0, 1)), (timesteps, -1)) for i in range(len(x))]
    y_val = np.eye(num_classes)[y.astype(np.uint8)]
    return np.asarray(x_val), y_val


def next_batch(x, labels, timesteps, batch_size, num_classes):
    n = len(x)
    ind = np.random.randint(0, n, batch_size)
    batch_x = [np.reshape(np.transpose(x[i][..., 0:timesteps], (2, 0, 1)), (timesteps, -1)) for i in ind]
    batch_y = np.eye(num_classes)[labels[ind].astype(np.uint8)]
    return np.asarray(batch_x), batch_y


def RNN(x, weights, biases, timesteps, num_hidden):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell1 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.)
    lstm_cell2 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.)
    lstm_cells = rnn.MultiRNNCell([lstm_cell1, lstm_cell2])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cells, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def lfg(args):

    # Training Parameters
    learning_rate = 0.0001
    training_steps = args.steps
    batch_size = 50
    display_step = 50

    # Network Parameters
    num_input = 67 * 27  # 67 x 27 is size of each frame
    timesteps = 28  # number of timesteps used for classification
    if args.mnh:
        _num_hidden = [10, 20, 32, 50, 64, 100, 128, 150, 200, 256, 500, 512]
        all_avg_val_accs = []
    else:
        _num_hidden = [64]  # hidden layer num of features
    num_classes = 10  # 10 actions

    if args.graph:
        import matplotlib.pyplot as plt
        if not os.path.isdir(os.path.join(os.path.abspath(''), 'plots')):
            os.makedirs(os.path.join(os.path.abspath(''), 'plots'))

    for num_hidden in _num_hidden:

        tf.reset_default_graph()

        if args.log:
            if not os.path.isdir(os.path.join(os.path.abspath(''), 'log')):
                os.makedirs(os.path.join(os.path.abspath(''), 'log'))

            fh = logging.FileHandler(
                './log/ex_5_num_hidden_{}_{}.log'.format(num_hidden, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')))
            logger.addHandler(fh)

        logger.info(args)
        logger.info('Hidden layer number of features: {}'.format(num_hidden))

        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        logits = RNN(X, weights, biases, timesteps, num_hidden)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        with open('./data/X.pickle', 'rb') as f:
            x = _pickle.load(f, encoding='bytes')
        _y = np.load('./data/l.npy')

        _x = np.asarray([i[:, :, :28] for i in x])

        k = 5

        if args.mnh:
            per_fold_val_accs = []

        for i in range(k):

            num_val = len(x) // k
            start = i * k
            mask_val = np.zeros(len(_x))
            mask_val[start: start + num_val] = True

            idx = np.arange(len(_x))
            np.random.shuffle(idx)
            x = _x[idx]
            y = _y[idx]

            x_tr = x[mask_val == False]
            y_tr = y[mask_val == False]

            x_val = x[mask_val == True]
            y_val = y[mask_val == True]

            logger.info('This fold\'s validation data contains examples from following classes: {}'.format(y_val))

            x_val, y_val = transform_val_data(x_val, y_val, timesteps, num_classes)

            if args.graph:
                train_loss = []
                train_acc = []
                val_loss = []
                val_acc = []

            # Start training
            with tf.Session() as sess:

                # Run the initializer
                sess.run(init)

                for step in range(1, training_steps + 1):
                    batch_x, batch_y = next_batch(x_tr, y_tr, timesteps, batch_size, num_classes)
                    # define the optimization procedure
                    train_op.run(feed_dict={X: batch_x, Y: batch_y})

                    if args.graph:
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                        loss_val, acc_val = sess.run([loss_op, accuracy], feed_dict={X: x_val, Y: y_val})
                        train_acc.append(acc * 100)
                        train_loss.append(loss)
                        val_acc.append(acc_val * 100)
                        val_loss.append(loss_val)


                    if step % display_step == 0 or step == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                             Y: batch_y})
                        logger.info("Training: Step " + str(step) + ", Minibatch Loss= " + \
                                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                                    "{:.3f}".format(acc * 100))

                        loss_val, acc_val = sess.run([loss_op, accuracy], feed_dict={X: x_val, Y: y_val})
                        logger.info("Validation: Step " + str(step) + ", Minibatch Loss= " + \
                                    "{:.4f}".format(loss_val) + ", Training Accuracy= " + \
                                    "{:.3f}".format(acc_val * 100))

                logger.info("Optimization Finished!")

            if args.graph:
                plot_loss = np.asarray([train_loss, val_loss])
                plt.title('loss - number of hidden units {} fold {}'.format(num_hidden, i))
                plt.xlabel('training steps')
                plt.ylabel('loss')
                plt.plot(plot_loss[0], label='training loss')
                plt.plot(plot_loss[1], label='validation loss')
                plt.legend()
                plt.savefig('./plots/loss_num_hidden_{}_fold_{}_steps_{}.png'.format(num_hidden, i, training_steps))
                # plt.show()
                plt.close()

                plot_acc = np.asarray([train_acc, val_acc])
                plt.title('accuracy - number of hidden units {} fold {}'.format(num_hidden, i))
                plt.xlabel('training steps')
                plt.ylabel('accuracy')
                plt.ylim(0, 110)
                plt.plot(plot_acc[0], label='training acc')
                plt.plot(plot_acc[1], label='validation acc')
                plt.legend()
                plt.savefig('./plots/acc_num_hidden_{}_fold_{}_steps_{}.png'.format(num_hidden, i, training_steps))
                # plt.show()
                plt.close()

            if args.graph and args.mnh:
                per_fold_val_accs.append(np.mean(np.asarray(val_acc)))

        if args.graph and args.mnh:
            all_avg_val_accs.append(np.mean(np.asarray(per_fold_val_accs)))

    logger.info('{} hidden features performed best on average'.format(np.asarray(_num_hidden[np.argmax(all_avg_val_accs)])))


'''
150 hidden features performed best on average
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store_true', help='wirte log to file')
    parser.add_argument('--mnh', action='store_true', help='try different num_hidden')
    parser.add_argument('--graph', action='store_true', help='draw some nice graphs')
    parser.add_argument('--steps', type=int, metavar='', help='number of training steps', default=10000)

    args = parser.parse_args()

    logger = logging.getLogger('ex_5')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    seed = 1337
    np.random.seed(seed=seed)
    lfg(args)
