import logging
import tensorflow as tf

from load_data import load_data, _file


def main(file):

    epochs = 1000
    num_classes = 10

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, num_classes])
    w = tf.Variable(tf.zeros([784, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    pred = tf.matmul(x, w) + b

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.nn.softmax(pred)), reduction_indices=[1]))

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

    logger = logging.getLogger('ex2_1')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main(file=_file)