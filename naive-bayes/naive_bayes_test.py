import csv
import numpy as np
from naive_bayes import NaiveBayes


"""
An application of naive bayes: Spam Email Classifying
"""


class SpamClassify(object):
    def __init__(self, train_file, test_file):
        self.__train_file = train_file
        self.__test_file = test_file
        self.__x, self.__y, self.__test_x, self.__test_y = [], [], [], []
        self.__x_dim, self.__y_dim = (None, None)
        self.__prdt_accuracy = 0.0
        self.__label = []

    # To classify the array into 3 classes according its value
    def __trisection(self, val, row):
        if (row == -4 and val >= 3.112) or \
           (row == -3 and val >= 28.0) or \
           (row == -2 and val >= 174.0):
            return 2
        elif (row == -4 and val >= 1.794) or \
             (row == -3 and val >= 10.0) or \
             (row == -2 and val >= 52.0):
            return 1
        else:
            return 0

    def __quasection(self, val, row):
        if (row == -4 and val >= 3.714) or \
           (row == -3 and val >= 44.0) or \
           (row == -2 and val >= 264.0):
            return 3
        elif (row == -4 and val >= 2.278) or \
             (row == -3 and val >= 15.0) or \
             (row == -2 and val >= 95.0):
            return 2
        elif (row == -4 and val >= 1.592) or \
             (row == -3 and val >= 6.0) or \
             (row == -2 and val >= 35.0):
            return 1
        else:
            return 0

    def __preprocess(self, file):
        tmp_x, tmp_y = [], []
        with open(file, 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                if rd.line_num == 1:
                    self.__label = row
                    continue
                # The last three features: 'average', 'longest', 'total'
                # Through testing, only using the first or the last one would generates better result
                xi = [1 if float(xi) > 0 else 0 for xi in row[:-4]]
                # xi.append(self.__quasection(float(row[-4]), -4))
                # xi.append(self.__quasection(float(row[-3]), -3))
                xi.append(self.__quasection(float(row[-2]), -2))
                tmp_x.append(xi)
                tmp_y.append(int(row[-1]))
        return tmp_x, tmp_y

    def calc_accuracy(self):
        self.__x, self.__y = self.__preprocess(self.__train_file)
        self.__test_x, self.__test_y = self.__preprocess(self.__test_file)
        self.__x_dim = [2 for i in range(54)] + [4 for i in range(1)]
        self.__y_dim = 2

        yes = 0
        for i in range(len(self.__test_x)):
            nb_obj = NaiveBayes(self.__x, self.__y, self.__test_x[i],
                                self.__x_dim, self.__y_dim)
            if nb_obj.predict() == self.__test_y[i]:
                yes += 1
            if (i + 1) % 20 == 0:
                print('%d tests Done. ' % (i + 1), end='')
                print('%d/%d are right' % (yes, i + 1))
        self.__prdt_accuracy = yes / len(self.__test_x)
        return self.__prdt_accuracy

    # just use one feature to calculate the prediction accuracy
    def calc_k_accuracy(self, kth):
        self.__x, self.__y = self.__preprocess(self.__train_file)
        self.__x = np.array(self.__x)[:, kth].reshape((4000, 1))
        self.__test_x, self.__test_y = self.__preprocess(self.__test_file)
        self.__test_x = np.array(self.__test_x)[:, kth].reshape((600, 1))
        self.__x_dim = [2]
        self.__y_dim = 2

        yes = 0
        for i in range(len(self.__test_x)):
            nb_obj = NaiveBayes(self.__x, self.__y, self.__test_x[i],
                                self.__x_dim, self.__y_dim)
            if nb_obj.predict() == self.__test_y[i]:
                yes += 1

        self.__prdt_accuracy = yes / len(self.__test_x)
        return self.__label[kth], self.__prdt_accuracy


def main():
    train_file = 'data_nb\\train.csv'
    test_file = 'data_nb\\test.csv'
    spam_obj = SpamClassify(train_file, test_file)
    print(spam_obj.calc_accuracy())

    # Accuracy: 0.87833

    """
    for i in range(54):
        spam_obj = SpamClassify(train_file, test_file)
        print("%s : %f" % spam_obj.calc_k_accuracy(i))
    """

main()
