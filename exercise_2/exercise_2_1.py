import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from load_data import load_data, _file


def main(file):

    epochs = 1000
    num_classes = 10

    logger.info('loading data from file  {}'.format(file))
    x_tr, y_tr, x_te, y_te = load_data(file, num_classes, flatten=True)
    logger.info('loading finished')

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, num_classes])
    w = tf.Variable(tf.zeros([784, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    pred = tf.matmul(x, w) + b

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.nn.softmax(pred)), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

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

    _w = sess.run(w, feed_dict={x: x_tr, y: y_tr})
    print(_w.shape)

    idx = 1
    for i, weights in enumerate(_w.T):
        plt.subplot(num_classes, num_classes, idx)
        idx += 1
        img = np.reshape(weights, (int(np.sqrt(len(weights))), int(np.sqrt(len(weights)))))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        tmp = x_tr[y_tr[:, i] == 1][:9]
        for j, pic in enumerate(tmp):
            plt.subplot(num_classes, num_classes, idx)
            idx += 1
            img = np.reshape(pic, (int(np.sqrt(len(pic))), int(np.sqrt(len(pic)))))
            plt.imshow(img, cmap='gray')
            plt.axis('off')

    plt.savefig('a_nice_table_of_images.png')
    plt.show()


if __name__ == '__main__':

    logger = logging.getLogger('ex2_1')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main(file=_file)