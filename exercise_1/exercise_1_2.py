import tensorflow as tf
from scipy import misc
import numpy as np
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

    return np.asarray(X).flatten(), np.asarray(y).flatten(), np.asarray(r).flatten()


def exercise_1_2a():

    seed = 12345
    np.random.seed(seed)

    z = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    for n in range(1,10):

        a = tf.Variable(np.random.uniform(-1, 1, n + 1), dtype=tf.float32)
        s = a[0]

        for i in range(1, n + 1):
            s += (a[i] * z) ** i

        model = tf.multiply(x, s)

        # tf.losses.mean_squared_error(y, model)
        loss = tf.reduce_mean(tf.pow(tf.subtract(model, y), 2))
        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        X_data, y_data, r = read_imgs()

        # x_tr = misc.imread('cat_01.jpg')
        # y_tr = misc.imread('cat_01_vignetted.jpg')

        # W = x_tr.shape[1]
        # H = x_tr.shape[0]
        # wc = W / 2
        # hc = H / 2
        #
        # xv, yv = np.meshgrid(np.arange(W) - wc, np.arange(H) - hc)
        #
        # r = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(wc ** 2 + hc ** 2)

        # r_ = r

        # x_train = x_tr.flatten()
        # y_train = y_tr.flatten()

        # r = np.dstack((r,r,r)).flatten()
        # r = r.flatten()

        for j in range(5):
            num_val = len(X_data) // 5
            start = j * num_val
            mask_val = np.zeros(len(X_data))
            mask_val[start : start + num_val] = True

            x_train = X_data[mask_val == False]
            x_val = X_data[mask_val == True]

            y_train = y_data[mask_val == False]
            y_val = y_data[mask_val == True]

            r_train = r[mask_val == False]
            r_val = r[mask_val == True]

            # a_before, loss_before, _x, _y, _z = sess.run([a, loss, x, y, z], {x : x_train, y: y_train, z: r})
            #
            # print(a_before, loss_before, _x, _y, _z)

            for i in range(500):
                sess.run(train, {x : x_train, y: y_train, z: r_train})
                # aa, ll = sess.run([a, loss], {x : x_train, y: y_train, z: r})
                # print('{}: {} \t {}'.format(i, aa, ll))

            a_, loss_after = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
            print(n, ': ', a_, loss_after)

    # s_ = a_[0]
    #
    # for i in range(1, n):
    #     s_ += (a_[i] * r_) ** i
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


if __name__ == '__main__':
    exercise_1_2a()