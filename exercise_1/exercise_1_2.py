import time
import logging
import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


'''read images and return data, labels and r'''
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


'''do exercise 2a and b'''
def exercise_1_2a():

    # placeholders for x, y and z
    z = tf.placeholder(tf.float32)
    # training data, in this case original images
    x = tf.placeholder(tf.float32)
    # 'labels', in this case vignetted and noisy images
    y = tf.placeholder(tf.float32)

    # one could swap values of x and y, therefore model had to be changed to tf.divide(x, s)

    logger.info('Reading images...')
    X_data, y_data, r = read_imgs()
    logger.info('Finished reading images')

    # k: number of folds for cross validation
    k = 5
    # minimum degree to check in cross validation
    min_degree = 1
    # maximum degree - 1 to check in cross validation
    max_degree = 10
    # number of trainings steps
    train_steps = 1000

    # check degrees from min_degree to max_degrees - 1
    for n in range(min_degree, max_degree):

        # 'randomly' initialize n+1 values for a
        a = tf.Variable(np.random.uniform(-1, 1, n + 1), dtype=tf.float32)
        s = a[0]

        # calculate value of polynomial
        for i in range(1, n + 1):
            s += a[i] * (z ** i)

        # calculate prediction of model
        model = tf.multiply(x, s)

        # MSE of predictions and true labels, MSE because it's a solid error measure
        # tf.losses.mean_squared_error(y, model)
        loss = tf.reduce_mean(tf.pow(tf.subtract(model, y), 2))
        # using adam because (quote mr. goldlücke) 'give any learning rate, and adam will do the rest for you'
        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(loss)

        # zero-filled matrices to store values for later comparison
        val_loss = np.zeros((k, train_steps))
        train_loss = np.zeros((k, train_steps))
        best_params = np.zeros(a.shape)

        # loop over folds of cross validation
        for j in range(k):

            # tensorflow initialisation stuff
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            # divide data into training and validation data
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

            # set best loss of validation to maximum value, so everything will be less
            best_val_loss = np.finfo(np.float32).max

            # get parameters and loss before training
            a_before, loss_before = sess.run([a, loss], {x : x_train, y: y_train, z: r_train})
            best_params = a_before

            logger.info('params and loss before training on training data: \n\t{}\t{}'.format(a_before, loss_before))

            begin = time.time()

            # do training
            for i in range(train_steps):
                # do training
                sess.run(train, {x : x_train, y: y_train, z: r_train})
                #get losses after each training step
                loss_train = sess.run(loss, {x : x_train, y: y_train, z: r_train})
                params, loss_val = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
                train_loss[j, i] = loss_train
                val_loss[j,i] = loss_val
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_params = params
                # logging output all 100 'epochs'
                if (i % 100) == 0:
                    logger.info('{}: {} \t {}'.format(i, loss_train, loss_val))
            end = time.time()

            a_after, loss_after = sess.run([a, loss], {x : x_val, y: y_val, z: r_val})
            logger.info('params and loss after training on validation data: \n\t{}\t{}'.format(a_after, loss_after))
            logger.info('training took {0:.3f}s with {ts} training steps'.format(end-begin, ts=train_steps))

        logger.info('Saving loss arrays...')
        # save values for later comparison
        np.savez('train_loss_degree_{}'.format(n), train_loss)
        np.savez('val_loss_degree_{}'.format(n), val_loss)
        # np.savez('best_params_degree_{}{}'.format(n, str), best_params)
        logger.info('Finished saving loss arrays')


'''do exercise 2c, in doubt see comments of exercise_1_2a()'''
def exercise_1_2c():

    z = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    logger.info('Reading images...')
    X_data, y_data, r = read_imgs()
    logger.info('Finished reading images')

    k = 5
    # minimum degree to check in cross validation (nice graphs of 2b showed degree 4 resulted in the best loss)
    min_degree = 5
    max_degree = 10
    train_steps = 1000

    for n in range(min_degree, max_degree):

        # loop over different weight decay values, in our case from 10^0 to 10^-4
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


'''draw those nice graphs'''
def draw_some_nice_graphs():

    for i in range(1,10):
        # read data from file
        loss_train = np.load('train_loss_degree_{}.npz'.format(i))['arr_0']
        # read data from file
        loss_val = np.load('val_loss_degree_{}.npz'.format(i))['arr_0']

        plt.title('Degree {}'.format(i))
        # mean of training loss
        plt.plot(loss_train.mean(axis=0), color='b', label='training error')
        # mean of validation loss
        plt.plot(loss_val.mean(axis=0), color='g', label='generalization error')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('degree_{}.png'.format(i))
        plt.show()
        plt.close()


'''find optimum degree'''
def find_optimum_degree(max_degree=10):

    # zero-filled matrix to store optimum losses
    optimum_loss_per_degree = np.zeros(max_degree-1)

    for n in range(1, max_degree):
        loss_val = np.load('val_loss_degree_{}.npz'.format(n))['arr_0']
        # store optimum loss per degree
        optimum_loss_per_degree[n - 1] = np.amin(loss_val.mean(axis=0))

    # store optimum over all optimum losses
    optimum_degree = optimum_loss_per_degree.argmin() + 1

    logger.info('Best degree: n = {} with loss of {}'.format(optimum_degree,
                                                         optimum_loss_per_degree[optimum_loss_per_degree.argmin()]))

    return optimum_degree


'''find optimum degree and lambda'''
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


'''retrain with optimum degree and weight decay to obtain parameters for devignetting'''
def retrain_with_optimum(degree, weight_decay):

    z = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    logger.info('Reading images...')
    X_data, y_data, r = read_imgs()
    logger.info('Finished reading images')

    train_steps = 1000

    a = tf.Variable(np.random.uniform(-1, 1, degree + 1), dtype=tf.float32)
    s = a[0]

    for i in range(1, degree + 1):
        s += a[i] * (z ** i)

    model = tf.multiply(x, s)

    # tf.losses.mean_squared_error(y, model)
    loss = tf.reduce_mean(tf.pow(tf.subtract(model, y), 2)) + (weight_decay * 0.5 * tf.reduce_sum(tf.pow(a, 2)))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    x_train = X_data
    y_train = y_data
    r_train = r

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    a_before, loss_before = sess.run([a, loss], {x : x_train, y: y_train, z: r})

    logger.info('params and loss before training on training data: \n\t{}\t{}'.format(a_before, loss_before))

    begin = time.time()

    for i in range(train_steps):
        sess.run(train, {x : x_train, y: y_train, z: r_train})
        if (i % 100) == 0:
            loss_train = sess.run(loss, {x : x_train, y: y_train, z: r_train})
            logger.info('{}: {}'.format(i, loss_train))
    end = time.time()

    a_after, loss_after = sess.run([a, loss], {x : x_train, y: y_train, z: r})
    logger.info('params and loss after training on validation data: \n\t{}\t{}'.format(a_after, loss_after))
    logger.info('training took {0:.3f}s with {ts} training steps'.format(end-begin, ts=train_steps))

    return a_after


'''perform devignetting'''
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

    # same as vignetting example, but in this case you just divide by s instead of multiplying
    J = np.zeros(img.shape, np.float32)
    J[:, :, 0] = img[:, :, 0] / s
    J[:, :, 1] = img[:, :, 1] / s
    J[:, :, 2] = img[:, :, 2] / s

    plt.imshow(np.uint8(J))
    plt.show()

    return J


'''devignetting for all images'''
def devignetting_all_imgs(a, n):

    for i in range(1,7):
        img = misc.imread('cat_0{}_vignetted.jpg'.format(i))
        img_devignetted = devignetting(img, a, n)
        misc.imsave('cat_0{}_devignetted.jpg'.format(i), img_devignetted)



""" The model doesn't help to remove the noise in the image because it describes only the vignetting of the image. To 
    get the noise we have to subtract the original image from the devignetted image. Then we can compute the empirical 
    mean and with this also the standard deviation. The formula you can see in the code. Then for denoising we subtract
    the noise from the devignetted image. Furthermore you can denoise the image with the given sigma and a filter 
    operation
    Otherwise one could apply a gaussian filter with an adaptable sigma (variable for tensorflow to optimize) 
    to the image after vignetting and thereby getting 
    model = tf.multiply(x, s) + gauss_filter(sigma) (because gauss filter is additive)
"""
def exercise_1_2_e():
    for i in range(1, 4, 1):
        img = misc.imread('cat_0{}.jpg'.format(i))
        img_devignetted = misc.imread('cat_0{}_devignetted.jpg'.format(i))
        noise = img_devignetted - img
        empirical_mean = np.sum(noise) / img.flatten().shape[0]
        sig_2 = np.sum(np.power(noise - empirical_mean, 2)) / (img.flatten().shape[0] - 1)
        logger.info('standard deviation image {}:\t{}'.format(i, np.sqrt(sig_2)))

        plt.imshow(np.uint8(img_devignetted- noise))
        plt.show()




def main():

    # set seed for reproducible outcomes
    seed = 12345
    np.random.seed(seed)

    exercise_1_2a()
    draw_some_nice_graphs()
    find_optimum_degree()
    exercise_1_2c()
    n, l = find_optimum_degree_and_lambda()
    a = retrain_with_optimum(n, l)
    devignetting_all_imgs(a, n)
    exercise_1_2_e()

# sidenode: code might be a bit repetitive but it's sunday night, so there might be some quick'n'dirty solutions :D
if __name__ == '__main__':

    logger = logging.getLogger('ex1')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main()