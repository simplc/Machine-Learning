import numpy as np


def gen_rand_matrix(shape, scope=(0, 1)):
    """
    generating random matrix with certain shape and scope
    :param shape: matrix shape (a, b)
    :param scope: random value scope (min_val, max_val), initialized as (0, 1)
    :return: randomized matrix
    """
    return np.random.uniform(scope[0], scope[1], shape)


def sigmoid(x):
    """
    sigmoid function
    :param x: input np array
    :return:
    """
    return 1 / (1 + np.exp(-x))


class NeuralNetwork(object):
    def __init__(self, d, q, l):
        """
        :param d: # input layer
        :param q: # hidden layer
        :param l: # output layer
        """

        self.d, self.q, self.l = d, q, l

        # randomization
        self.in2hid = gen_rand_matrix((d, q), (-1, 1))
        self.hid2out = gen_rand_matrix((q, l), (-1, 1))
        self.hid_thres = gen_rand_matrix((1, q), (-1, 1))
        self.out_thres = gen_rand_matrix((1, l), (-1, 1))

    def estimate(self, x):
        """
        forward-propagation to estimate the outputs for current parameters
        :param x: inputs with shape (1, d)
        :return y_est: outputs with shape (1, l)
        """
        hid_est = sigmoid(np.dot(x, self.in2hid) - self.hid_thres)  # shape: (1, q)
        y_est = sigmoid(np.dot(hid_est, self.hid2out) - self.out_thres)

        return hid_est, y_est

    def back_propagate(self, hid_est, y_est, x, y_truth, rate):
        """
        back-propagation which take one training example a time
        :param hid_est: estimation of hidden layer
        :param y_est:   estimation of output layer
        :param x:       inputs
        :param y_truth: ground truth of outputs
        :param rate:    learning rate
        :return:        error after each iteration
        """
        g = (-(y_est - y_truth) * y_est * (1 - y_est))
        e = np.dot(g, self.hid2out.T) * hid_est * (1 - hid_est)

        # upgrading weights and thresholds
        self.hid2out += rate * np.dot(hid_est.T, g)
        self.out_thres += rate * g
        self.in2hid += rate * np.dot(x.T, e)
        self.hid_thres += (-rate * e)

        # calculating error
        error = np.dot(y_est - y_truth, (y_est - y_truth).T) / 2

        return error

    def train(self, x, y, itr, rate):
        """
        training
        :param x:    inputs for training
        :param y:    outputs for training
        :param itr:  max training iterations
        :param rate: learning rate
        :return:
        """

        for i in range(1, itr + 1):
            # using one training example a time
            for x_, y_ in zip(x, y):
                x_ = x_.reshape((1, self.d))
                hid_est, y_est = self.estimate(x_)
                error = self.back_propagate(hid_est, y_est, x_, y_, rate)
            if i % 50 == 0:
                print("  Epoch %d/%d - loss: %.8f" % (i, itr,error))

    def test(self, x_test):
        """
        testing
        :param x_test: inputs for testing
        :return y_est: outputs(estimations)
        """
        hid_est, y_est = self.estimate(x_test)
        return y_est
