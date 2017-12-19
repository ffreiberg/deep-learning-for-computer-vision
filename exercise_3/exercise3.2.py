import numpy as np
import tensorflow as tf


def main():
    x = tf.placeholder(tf.float32, [3, 1])

    y = tf.placeholder(tf.float32, [1])

    # definiere Gewichtsmatrix für layer 1
    weights_l1 = np.zeros((3, 3))
    weights_l1[:] = 0.1

    W_layer1 = tf.Variable(weights_l1, dtype=tf.float32)

    # definiere Gewichtsmatrix für layer 2
    weights_l2 = np.zeros((3, 1))
    weights_l2[:] = 0.2

    W_layer2 = tf.Variable(weights_l2, dtype=tf.float32)

    # Forward path
    fc1 = tf.matmul(W_layer1, x)
    fc2 = tf.matmul(tf.transpose(fc1), W_layer2)

    loss = 0.5 * tf.pow((y - fc2), 2)

    var_gradient = tf.gradients(xs=[W_layer1, W_layer2], ys=loss)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(var_gradient, feed_dict={y: [10.0], x: [[1], [2], [3]]}))
        print(sess.run(loss, {y: [10.0], x: [[1], [2], [3]]}))
        #for i in range(10):
        #    train_step.run(feed_dict={y: [10.0], x: [[1], [2], [3]]})
        #    print(sess.run(loss, {y: [10.0], x: [[1], [2], [3]]}))



    #init = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init)
    #print(sess.run(loss, {y: [10.0], x: [[1], [2], [3]]}))

if __name__ == main():
    main()