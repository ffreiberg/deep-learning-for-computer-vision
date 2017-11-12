import time
import logging
import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


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


def exercise_1_2a(regularization=False):

    seed = 12345
    np.random.seed(seed)

    z = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    logger.info('Reading images...')
    X_data, y_data, r = read_imgs()
    logger.info('Finished reading images')

    k = 5
    min_degree = 1
    max_degree = 10
    train_steps = 1000

    for n in range(min_degree, max_degree):

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
        best_params = np.zeros(a.shape)

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

            best_val_loss = np.finfo(np.float32).max

            a_before, loss_before = sess.run([a, loss], {x : x_train, y: y_train, z: r_train})
            best_params = a_before

            logger.info('params and loss before training on training data: \n\t{}\t{}'.format(a_before, loss_before))

            begin = time.time()

            for i in range(train_steps):
                sess.run(train, {x : x_train, y: y_train, z: r_train})
                loss_train = sess.run(loss, {x : x_train, y: y_train, z: r_train})
                params, loss_val = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
                train_loss[j, i] = loss_train
                val_loss[j,i] = loss_val
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_params = params
                if (i % 100) == 0:
                    logger.info('{}: {} \t {}'.format(i, loss_train, loss_val))
            end = time.time()

            a_after, loss_after = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
            logger.info('params and loss after training on validation data: \n\t{}\t{}'.format(a_after, loss_after))
            logger.info('training took {0:.3f}s with {ts} training steps'.format(end-begin, ts=train_steps))

        logger.info('Saving loss arrays...')
        np.savez('train_loss_degree_{}{}'.format(n, str), train_loss)
        np.savez('val_loss_degree_{}{}'.format(n, str), val_loss)
        # np.savez('best_params_degree_{}{}'.format(n, str), best_params)
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


def exercise_1_2c():

    seed = 12345
    np.random.seed(seed)

    z = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    logger.info('Reading images...')
    X_data, y_data, r = read_imgs()
    logger.info('Finished reading images')

    k = 5
    min_degree = 5
    max_degree = 10
    train_steps = 1000

    for n in range(min_degree, max_degree):

        for m in range(0, 5):
            weight_decay = 1 / (10 ** (m))

            a = tf.Variable(np.random.uniform(-1, 1, n + 1), dtype=tf.float32)
            s = a[0]

            for i in range(1, n + 1):
                s += a[i] * (z ** i)

            model = tf.multiply(x, s)

            # tf.losses.mean_squared_error(y, model)
            loss = tf.reduce_mean(tf.pow(tf.subtract(model, y), 2)) + (weight_decay * 0.5 * tf.reduce_sum(tf.pow(a, 2)))
            optimizer = tf.train.AdamOptimizer(0.01)
            train = optimizer.minimize(loss)

            val_loss = np.zeros((k, train_steps))
            train_loss = np.zeros((k, train_steps))
            best_params = np.zeros(a.shape)

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

                best_val_loss = np.finfo(np.float32).max

                a_before, loss_before = sess.run([a, loss], {x : x_train, y: y_train, z: r_train})
                best_params = a_before

                logger.info('params and loss before training on training data: \n\t{}\t{}'.format(a_before, loss_before))

                begin = time.time()

                for i in range(train_steps):
                    sess.run(train, {x : x_train, y: y_train, z: r_train})
                    loss_train = sess.run(loss, {x : x_train, y: y_train, z: r_train})
                    params, loss_val = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
                    train_loss[j, i] = loss_train
                    val_loss[j,i] = loss_val
                    if loss_val < best_val_loss:
                        best_val_loss = loss_val
                        best_params = params
                    if (i % 100) == 0:
                        logger.info('{}: {} \t {}'.format(i, loss_train, loss_val))
                end = time.time()

                a_after, loss_after = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
                logger.info('params and loss after training on validation data: \n\t{}\t{}'.format(a_after, loss_after))
                logger.info('training took {0:.3f}s with {ts} training steps'.format(end-begin, ts=train_steps))

            logger.info('Saving loss arrays...')
            np.savez('train_loss_degree_{}_weight_decay_{}'.format(n, weight_decay), train_loss)
            np.savez('val_loss_degree_{}_weight_decay_{}'.format(n, weight_decay), val_loss)
            # np.savez('best_params_degree_{}_weight_decay_{}'.format(n, weight_decay), best_params)
            logger.info('Finished saving loss arrays')


def draw_some_nice_graphs():

    for i in range(1,10):
        loss_train = np.load('train_loss_degree_{}.npz'.format(i))['arr_0']
        loss_val = np.load('val_loss_degree_{}.npz'.format(i))['arr_0']

        plt.title('')
        plt.plot(loss_train.mean(axis=0), color='b', label='training error')
        plt.plot(loss_val.mean(axis=0), color='g', label='generalization error')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()


def find_optimum_degree(max_degree=10):

    optimum_loss_per_degree = np.zeros(max_degree-1)

    for n in range(1, max_degree):
        loss_val = np.load('val_loss_degree_{}.npz'.format(n))['arr_0']
        optimum_loss_per_degree[n - 1] = np.amin(loss_val.mean(axis=0))

    optimum_degree = optimum_loss_per_degree.argmin() + 1

    logger.info('Best degree: n = {} with loss of {}'.format(optimum_degree,
                                                         optimum_loss_per_degree[optimum_loss_per_degree.argmin()]))

    return optimum_degree


def find_optimum_degree_and_lambda(lambdas=5, min_degree=5, max_degree=10):

    optimum_loss_per_degree_per_lambda = np.zeros((max_degree-min_degree, lambdas))

    for n in range(min_degree, max_degree):
        for l in range(lambdas):
            loss_val = np.load('val_loss_degree_{}_weight_decay_{}.npz'.format(n, 1 / (10 ** l)))['arr_0']
            optimum_loss_per_degree_per_lambda[n - min_degree, l] = np.amin(loss_val.mean(axis=0))

    idx = np.unravel_index(optimum_loss_per_degree_per_lambda.argmin(), optimum_loss_per_degree_per_lambda.shape)

    optimum_degree = idx[0] + min_degree
    optimum_lambda = 1 / (10 ** idx[1])

    logger.info('Best degree and lambda: n = {} and lambda = {} with loss of {}'.format(optimum_degree,
                                                                                        optimum_lambda,
                                                                                        optimum_loss_per_degree_per_lambda[idx]))

    return optimum_degree, optimum_lambda


def devignetting(img, a, n):
    W = img.shape[1]
    H = img.shape[0]
    wc = W / 2
    hc = H / 2

    xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)

    r = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)

    s = a[0]

    for i in range(1, n):
        s += a[i] * (r ** i)

    J = np.zeros(img.shape, np.float32)
    J[:, :, 0] = img[:, :, 0] / s
    J[:, :, 1] = img[:, :, 1] / s
    J[:, :, 2] = img[:, :, 2] / s

    plt.imshow(np.uint8(J))
    plt.show()

    return J


if __name__ == '__main__':

    logger = logging.getLogger('ex1')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # exercise_1_2a()
    # nice_graphs()
    # exercise_1_2c()
    # find_optimum_degree()
    find_optimum_degree_and_lambda()