import numpy as np


input = np.array([1., 2., 3.]).astype('float32')

# definiere Gewichtsmatrix für layer 1
weights_l1 = np.zeros((3, 3)).astype('float32')
weights_l1[:] = 0.1

# definiere Gewichtsmatrix für layer 2
weights_l2 = np.zeros((3, 1)).astype('float32')
weights_l2[:] = 0.2

f_l1 = np.dot(weights_l1, input)
f_l2 = np.dot(f_l1, weights_l2)

loss = 0.5 * np.power((10.0 - f_l2), 2)
print(loss)

ableitung_w_20_10 = 0.1 * 1. + 0.1 * 2. + 0.1 * 3.
ableitung_w_20_11 = 0.1 * 1. + 0.1 * 2. + 0.1 * 3.
ableitung_w_20_12 = 0.1 * 1. + 0.1 * 2. + 0.1 * 3.

df_l2_w = np.array([ableitung_w_20_10, ableitung_w_20_11, ableitung_w_20_12]).astype('float32')

d1 = (f_l2 - 10.0)
print(d1 * df_l2_w)

dx = np.ones((3)).astype('float32') * (0.2)
dt_dx = d1 * dx

dx2 = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]).astype('float32')

erg = dx2 * dt_dx
print(erg)