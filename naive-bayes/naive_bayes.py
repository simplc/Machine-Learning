import numpy as np


class NaiveBayes(object):
    # =================================================================== #
    # ARGUMENTS                                                           #
    #   x:      training inputs (n*d matrix)                              #
    #   y:      training outputs (n*1 vector)                             #
    #   test_x: testing inputs                                            #
    #   x_dim:  range of x (d*1 vector)                                   #
    #             x_dim[i] = p means the i-th feature's range is [0, p-1] #
    #             may vary from one feature to another                    #
    #   y_dim:  range of y (scalar)                                       #
    #             y_dim = q means the output's range is [0, q-1]          #
    #   prior:  prior distribution P(y) (y_dim*1 vector)                  #
    #   llh:    likelihood P(x|y) (d*y_dim*x_dim[i] vector)               #
    #             llh[i][j][k] means P(x_i=k|y=j)                         #
    #             x_i: the i-th feature of x                              #
    #             e.g. llh = [ [[0.1, 0.9]      [0.5, 0.5]]               #
    #                          [[0.2, 0.3, 0.5] [0.3, 0.4, 0.3]]          #
    #                          [[0.45, 0.55]    [0.7, 0.3]] ]             #
    # =================================================================== #
    def __init__(self, x, y, test_x, x_dim, y_dim):
        self.__x, self.__y = np.array(x), np.array(y)
        self.__test_x = np.array(test_x)
        self.__prdt_y = 0
        self.__x_dim, self.__y_dim = np.array(x_dim), y_dim
        self.__num_train, self.__num_feat = self.__x.shape
        self.__prior = np.zeros(self.__y_dim)
        self.__llh = np.zeros((self.__num_feat, self.__y_dim, np.max(self.__x_dim)))
        # for convenience, just take the maximal of :x_dim:

    # calc prior
    def __calc_prior(self):
        for i in range(self.__y_dim):
            self.__prior[i] = float(np.bincount(self.__y)[i]) / self.__num_train

    # calc likelihood
    def __calc_llh(self):
        for i in range(self.__num_feat):
            for j in range(self.__y_dim):
                for k in range(self.__x_dim[i]):
                    """
                    To count the number of (x_i==k, y==j)
                        Method: take subtraction, and find the number where (x_i-k==0) and (y-j==0)
                            i.e. just find the number of zeros in vector (x_i-k)|(y-j)
                    """
                    tmp = np.bincount(np.abs((self.__x[:, i] - k) | (self.__y - j)))[0]
                    # Laplace smoothing method applied
                    self.__llh[i][j][k] = (tmp + 1) / \
                                          (np.bincount(self.__y)[j] + self.__x_dim[i])

    def __calc_llh_2(self):
        for i in range(self.__num_feat):
            for j in range(self.__y_dim):
                for k in range(self.__x_dim[i]):
                    tmp = np.column_stack((self.__x[:, i], self.__y))
                    self.__llh[i][j][k] = (tmp.tolist().count([k, j]) + 1) / \
                                          (np.bincount(self.__y)[j] + self.__x_dim[i])

    # predict the output given :test_x:
    def predict(self):
        self.__calc_prior()
        self.__calc_llh()
        prdt_llh = []
        for j in range(self.__y_dim):
            tmp = [self.__llh[:, j][i][self.__test_x[i]] for i in range(self.__num_feat)]
            prdt_llh.append(np.prod(tmp))

        self.__prdt_y = np.argmax(np.multiply(prdt_llh, self.__prior))

        return self.__prdt_y
