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

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle

# Training Parameters
learning_rate = 0.0001
training_steps = 10000
batch_size = 50
display_step = 50

# Network Parameters
num_input = 67*27 # 67 x 27 is size of each frame
timesteps = 28    # number of timesteps used for classification
num_hidden = None  # hidden layer num of features
num_classes = 10  # 10 actions

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



def next_batch(x,labels,timesteps,batch_size):
    n = len(x)
    ind = np.random.randint(0,n,batch_size)
    batch_x = [np.reshape(np.transpose(x[i][...,0:timesteps],(2,0,1)),(timesteps,-1)) for i in ind]
    batch_y = np.eye(num_classes)[labels[ind].astype(np.uint8)]
    return np.asarray(batch_x), batch_y



def RNN(x, weights, biases):

    # here is a space to define your LSTM machine
    return None

logits = RNN(X, weights, biases)
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

x = pickle.load(open('./data/X.pickle'))
labels = np.load('./data/l.npy')

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = next_batch(x,labels,timesteps,batch_size)
        # define the optimization procedure


    print("Optimization Finished!")
