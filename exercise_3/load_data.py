import h5py
import numpy as np

te_x = 'test_x'
te_y = 'test_y'
tr_x = 'train_x'
tr_y = 'train_y'

global _file
_file = 'pmjt_sample_20161116/train_test_file_list.h5'

def load_data(file, num_classes, flatten=False):

    with h5py.File(file) as hf:
        x_train = normalize_data(np.array(hf.get(tr_x)).astype(np.float32))
        y_train = np.array(hf.get(tr_y))
        x_test = normalize_data(np.array(hf.get(te_x)).astype(np.float32))
        y_test = np.array(hf.get(te_y))

    if flatten:
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    else:
        x_train = x_train[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]

    y_train = one_hot(y_train, num_classes)
    y_test = one_hot(y_test, num_classes)


    return x_train, y_train, x_test, y_test


# def normalize_data(data):
#
#     data /= data.max()
#     data -= data.mean()
#
#     return data

def normalize_data(X):
    for i in range(X.shape[0]):
        # zero mean
        X[i, ...] = X[i, ...] - np.mean(X[i, ...].ravel())
        # X[i, ...] = X[i, ...] / np.std(X[i, ...].ravel())
        X[i,...] = X[i,...] / np.maximum(np.finfo(np.float32).eps, np.sqrt(np.sum(X[i,...].ravel() ** 2)))
    return X


def one_hot(data, num_classes):

    data = np.eye(num_classes)[data.flatten()]

    return data