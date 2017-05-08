import numpy as np


class LinearRegression(object):

    def __init__(self, array_x, array_y, alpha):
        self.x = np.concatenate((np.ones((np.array(array_x).shape[0], 1)), np.array(array_x)), axis=1)
        self.y = np.array(array_y)
        self.alpha = alpha
        self.theta = []

    def __hyp_func(self):
        return np.dot(self.x, self.theta)

    def __cost_func(self):
        tmp = np.dot(self.x, self.theta) - self.y
        return np.sum(np.multiply(tmp, tmp)) / 2

    def batch_gradient_descent(self):
        n = self.x.shape[1]
        delta = 1e-6              # error checker
        itr = 0                   # iteration counter
        self.theta = np.zeros(n)  # initialization

        cost0 = self.__cost_func()
        while 1:
            self.theta -= self.alpha * np.dot(np.transpose(self.x), 
                                              self.__hyp_func() - self.y)
            cost = self.__cost_func()
            if abs(cost0 - cost) < delta:
                print("Convergent!")
                break
            else:
                cost0 = cost
            itr += 1

        return cost, itr

    def normal_equ(self):
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.x), self.x)), np.transpose(self.x)), self.y)
        cost = self.__cost_func()
        return cost

    def newton_method(self):
        n = self.x.shape[1]
        delta = 1e-8              # error checker
        itr = 0                   # iteration counter
        self.theta = np.zeros(n)  # initialization

        cost0 = self.__cost_func()
        while 1:
            gradient = np.dot(np.dot(np.transpose(self.theta),
                                     np.transpose(self.x)),
                              self.x) - np.dot(np.transpose(self.y), self.x)
            hessian = np.dot(np.transpose(self.x), self.x)
            self.theta -= np.dot(np.linalg.inv(hessian),
                                 np.transpose(gradient))
            cost = self.__cost_func()
            if abs(cost0 - cost) < delta:
                print("Convergent!")
                break
            else:
                cost0 = cost
            itr += 1

        return cost, itr

    def predict(self, qx):
        return np.dot(qx, self.theta)
