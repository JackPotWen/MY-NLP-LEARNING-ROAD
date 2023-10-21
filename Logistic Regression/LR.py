import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def gradient(x, y, theta, alpha, num_iters):
    m = x.shape[0]

    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1. / m * (np.dot(y.transpose(), np.log(h)) + np.dot((1 - y).transpose(), np.log(1 - h)))
        theta = theta - (alpha / m) * np.dot(x.transpose(), (h - y))

    J = float(J)

    return J, theta
"""
# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Apply gradient descent
tmp_J, tmp_theta = gradient(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")
"""
