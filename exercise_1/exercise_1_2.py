import time
import logging
import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


# def read_imgs(start=1, end=3):
#
#     X = []
#     y = []
#
#     for i in range(start, end + 1):
#
#         img_vignetted = misc.imread('cat_0{}_vignetted.jpg'.format(i))
#         img = misc.imread('cat_0{}.jpg'.format(i))
#
#         if img_vignetted.shape != (500, 313, 3):
#             img_vignetted = np.rot90(img_vignetted)
#         if img.shape != (500, 313, 3):
#             img = np.rot90(img)
#
#         X.append(img_vignetted)
#         y.append(img)
#
#     return X, y

def read_imgs(start=1, end=3):

    X = []
    y = []
    r = []

    for i in range(start, end + 1):

        img_vignetted = misc.imread('cat_0{}_vignetted.jpg'.format(i))
        img = misc.imread('cat_0{}.jpg'.format(i))

        W = img.shape[1]
        H = img.shape[0]
        wc = W / 2
        hc = H / 2

        xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)

        _r = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)
        _r = np.dstack((_r, _r, _r))

        X.append(img.flatten())
        y.append(img_vignetted.flatten())
        r.append(_r.flatten())

    return np.asarray(X).flatten().astype(np.float32), np.asarray(y).flatten().astype(np.float32), np.asarray(r).flatten().astype(np.float32)


def exercise_1_2a():

    seed = 12345
    np.random.seed(seed)

    z = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    logger.info('Reading images...')
    X_data, y_data, r = read_imgs()
    logger.info('Finished reading images')

    k = 5
    max_degree = 10
    train_steps = 1000

    for n in range(1, max_degree):

        a = tf.Variable(np.random.uniform(-1, 1, n + 1), dtype=tf.float32)
        s = a[0]

        for i in range(1, n + 1):
            s += a[i] * (z ** i)

        model = tf.multiply(x, s)

        # tf.losses.mean_squared_error(y, model)
        loss = tf.reduce_mean(tf.pow(tf.subtract(model, y), 2))
        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(loss)

        val_loss = np.zeros((k, train_steps))
        train_loss = np.zeros((k, train_steps))

        for j in range(k):

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            num_val = len(X_data) // k
            start = j * num_val
            mask_val = np.zeros(len(X_data))
            mask_val[start : start + num_val] = True

            x_train = X_data[mask_val == False]
            x_val = X_data[mask_val == True]

            y_train = y_data[mask_val == False]
            y_val = y_data[mask_val == True]

            r_train = r[mask_val == False]
            r_val = r[mask_val == True]

            a_before, loss_before = sess.run([a, loss], {x : x_train, y: y_train, z: r_train})

            logger.info('params and loss before training on training data: \n\t{}\t{}'.format(a_before, loss_before))

            begin = time.time()

            for i in range(train_steps):
                sess.run(train, {x : x_train, y: y_train, z: r_train})
                loss_train = sess.run(loss, {x : x_train, y: y_train, z: r_train})
                loss_val = sess.run(loss, {x : x_val, y: y_val, z: r_val})
                train_loss[j, i] = loss_train
                val_loss[j,i] = loss_val
                if (i % 100) == 0:
                    logger.info('{}: {} \t {}'.format(i, loss_train, loss_val))

            end = time.time()

            a_after, loss_after = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
            logger.info('params and loss after training on validation data: \n\t{}\t{}'.format(a_after, loss_after))
            logger.info('training took {0:.3f}s with {} training steos'.format(end-begin, train_steps))

        logger.info('Saving loss arrays...')
        np.savez('train_loss_degree_{}'.format(n), train_loss)
        np.savez('val_loss_degree_{}'.format(n), val_loss)
        logger.info('Finished saving loss arrays')

        # plt.plot(val_loss.mean(axis=0))
        # plt.show()
            # s_ = a_[0]
            #
            # for i in range(1, n):
            #     s_ += a_[i] * (r_ ** i)
            #
            #
            # J = np.zeros(x_tr.shape, np.float32)
            # J[:, :, 0] = y_tr[:, :, 0] / s_
            # J[:, :, 1] = y_tr[:, :, 1] / s_
            # J[:, :, 2] = y_tr[:, :, 2] / s_
            #
            # plt.imshow(np.uint8(J))
            # # plt.imshow(J)
            # plt.show()
            # r = r.flatten()
            # r = np.dstack((r,r,r)).flatten()
            # y_train = y_tr.flatten()
            # x_train = x_tr.flatten()
            # r_ = r
            # r = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)
            #
            # xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)
            #
            # hc = H / 2
            # wc = W / 2
            # H = x_tr.shape[0]
            # W = x_tr.shape[1]
            # y_tr = misc.imread('cat_01_vignetted.jpg')
            # x_tr = misc.imread('cat_01.jpg')

def nice_graphs():

    for i in range(1,10):
        loss_train = np.load('train_loss_degree_{}.npz'.format(i))['arr_0']
        loss_val = np.load('train_loss_degree_{}.npz'.format(i))['arr_0']

        plt.plot(loss_train.mean(axis=0), color='b', label='training error')
        plt.plot(loss_val.mean(axis=0), color='g', label='generalization error')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

#
# def test():
#
#     def model_fn(features, labels, mode):
#         a = tf.Variable(np.random.uniform(-1, 1, 5 + 1), dtype=tf.float32)
#         s = a[0]
#
#         for i in range(1, 6):
#             s += (a[i] * r) ** i
#
#         pred = tf.multiply(features['x'], s)
#         loss = tf.reduce_mean(tf.pow(tf.subtract(pred, labels), 2))
#         global_step = tf.train.get_global_step()
#         optimizer = tf.train.AdamOptimizer(0.01)
#         train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
#
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=pred, loss=loss, train_op=train )
#
#
#     X_data, y_data, r = read_imgs()
#
#     j = 0
#
#     num_val = len(X_data) // 5
#     start = j * num_val
#     mask_val = np.zeros(len(X_data))
#     mask_val[start: start + num_val] = True
#
#     x_train = X_data[mask_val == False]
#     x_val = X_data[mask_val == True]
#
#     y_train = y_data[mask_val == False]
#     y_val = y_data[mask_val == True]
#
#     r_train = r[mask_val == False]
#     r_val = r[mask_val == True]
#
#     estimator = tf.estimator.Estimator(model_fn=model_fn)
#
#     input_fn = tf.estimator.inputs.numpy_input_fn({'x' : x_train, 'r': r_train}, y_train, batch_size=len(X_data)-num_val, num_epochs=None, shuffle=False)
#     train_input_fn = tf.estimator.inputs.numpy_input_fn({'x' : x_train, 'r': r_train}, y_train, batch_size=len(X_data)-num_val, num_epochs=500, shuffle=False)
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x' : x_val, 'r': r_val}, y_val, batch_size=num_val, num_epochs=500, shuffle=False)
#
#     estimator.train(input_fn=input_fn, steps=500)
#
#     train_metrics = estimator.evaluate(input_fn=train_input_fn)
#     eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
#
#     print(train_metrics)
#     print(eval_metrics)


if __name__ == '__main__':

    logger = logging.getLogger('ex1')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # exercise_1_2a()
    nice_graphs()
    # test()