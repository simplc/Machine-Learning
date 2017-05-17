from neural_network import NeuralNetwork
import numpy as np


def init_data(ch1, ch2):
    data_file = "data\\letter-recognition.data"
    fp = open(data_file, "r+")

    x, y = [], []
    while True:
        line = fp.readline().strip().split(',')
        if not line[0]:
            break
        char = line[0]

        if char == ch1 or char == ch2:
            y.append(0 if char == ch1 else 1)  # binary choice
            x.append([int(ch) for ch in line[1:]])

    total = len(y)

    # x = np.array((x - np.min(x)) / (np.max(x) - np.min(x)))  # regularize to (0, 1)
    # with regularization: 0.9935

    # x = np.array((np.sign(x - np.mean(x)) + 1) / 2, dtype=int)  # classified into 0 or 1
    # with classifying: 0.9891

    x = np.array(x)
    # without any operation: 0.9956

    y = np.array(y).reshape((total, 1))
    x_train, x_test = x[:int(total * 0.7)], x[int(total * 0.7):]
    y_train, y_test = y[:int(total * 0.7)], y[int(total * 0.7):]

    return x_train, y_train, x_test, y_test


def bin_classifier(ch1, ch2, itr, rate):

    print("Classifier for %c and %c..." % (ch1, ch2))

    x_train, y_train, x_test, y_test = init_data(ch1, ch2)
    nn_obj = NeuralNetwork(16, 16, 1)
    nn_obj.train(x_train, y_train, itr, rate)
    y_est = nn_obj.test(x_test)

    right = 0
    for est, test in zip(y_est, y_test):
        est = np.round(est)
        if est == test:
            right += 1

    print("\tPrediction accuracy: %.8f" % (right / y_test.shape[0]))


if __name__ == '__main__':
    bin_classifier('O', 'X', itr=500, rate=0.2)
    bin_classifier('O', 'D', itr=500, rate=0.2)

