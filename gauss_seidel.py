# https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method#An_example_using_Python_and_NumPy
import numpy as np

ITERATION_LIMIT = 1000

# initialize the matrix
A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])
# initialize the RHS vector
b = np.array([6., 25., -11., 15.])

print("System of equations:")
for i in range(A.shape[0]):  # pylint: disable=E1136
    row = ["{0:3g}*x{1}".format(A[i, j], j + 1)
           for j in range(A.shape[1])]  # pylint: disable=E1136
    print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

x = np.zeros_like(b)
for it_count in range(1, ITERATION_LIMIT):
    x_new = np.zeros_like(x)
    #print("Iteration {0}: {1}".format(it_count, x))
    for i in range(A.shape[0]):  # pylint: disable=E1136
        s1 = np.dot(A[i, :i], x_new[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
    if np.allclose(x, x_new, rtol=1e-8):
        break
    x = x_new

print("Solution: {0}".format(x))
error = np.dot(A, x) - b
print("Error: {0}".format(error))
