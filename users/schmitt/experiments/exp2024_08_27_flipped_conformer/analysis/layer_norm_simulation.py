# Simulate what happens when a vector with small L2 norm is added to a vector with large L2 norm and then
# layer norm is applied to the result.
# in the paper, we argue that the layer norm will remove the information from the vector with small L2 norm.

import numpy as np
import matplotlib.pyplot as plt


def cos_sim(a: np.ndarray, b: np.ndarray):
  """

  Args:
    a: array of shape [N, D]
    b: array of shape [N, D]

  Returns:
    array of shape [N]
  """

  dot = np.einsum('nd,nd->n', a, b)
  norm_a = np.linalg.norm(a, axis=-1)
  norm_b = np.linalg.norm(b, axis=-1)
  return dot / (norm_a * norm_b)


# factor by which the magnitude of the second vector is increased compared to the first vector
MAGNITUDE_FACTOR = 3

# visualization using 2 2D vectors
rand = np.random.rand(2, 2)
norm = np.linalg.norm(rand, axis=1)[:, None]
rand /= norm
x, y = np.vsplit(rand, 2)
x = x[0]
y = y[0]
y *= MAGNITUDE_FACTOR
xpy = x + y
xpy_norm = (xpy - xpy.mean()) / xpy.std()
print(f"x: {x}, y: {y}, xpy: {xpy}")
print(f"xpy_norm: {xpy_norm}, xpy_norm_mean: {xpy_norm.mean()}, xpy_norm_std: {xpy_norm.std()}")
plt.scatter(*x, label='x')
plt.plot([0, x[0]], [0, x[1]], label='x')
plt.scatter(*y, label='y')
plt.plot([0, y[0]], [0, y[1]], label='x')
plt.scatter(*xpy, label='x+y')
plt.plot([0, xpy[0]], [0, xpy[1]], label='x')
plt.scatter(*xpy_norm, label='x+y layer normed')
plt.plot([0, xpy_norm[0]], [0, xpy_norm[1]], label='x')
plt.legend()
lim = 5
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.show()

# simulation using N pairs 512D vectors
D = 512
N = 10_000
rand = np.random.rand(N, 2, D)  # [N, 2, D]
norm = np.linalg.norm(rand, axis=-1)[:, :, None]  # [N, 2, 1]
rand /= norm
x, y = np.split(rand, 2, axis=1)  # [N, 1, D]
x = x[:, 0]  # [N, D]
y = y[:, 0]  # [N, D]
y *= MAGNITUDE_FACTOR
xpy = x + y
xpy_norm = (xpy - xpy.mean(axis=-1)[:, None]) / xpy.std(axis=-1)[:, None]

cos_x_xpy = cos_sim(x, xpy).mean()
cos_y_xpy = cos_sim(y, xpy).mean()
cos_x_xpy_norm = cos_sim(x, xpy_norm).mean()
cos_y_xpy_norm = cos_sim(y, xpy_norm).mean()

print(f"cos_x_xpy: {cos_x_xpy}, cos_y_xpy: {cos_y_xpy}")
print(f"cos_x_xpy_norm: {cos_x_xpy_norm}, cos_y_xpy_norm: {cos_y_xpy_norm}")
