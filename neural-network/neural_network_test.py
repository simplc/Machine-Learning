from neural_network import NeuralNetwork
import numpy as np


def init():
    data_file = "data_nn\\letter-recognition.data"
    fp = open(data_file, "r+")

    x, y = [], []
    while True:
        try:
            line = fp.readline().strip().split(',')
            char = ord(line[0]) - ord('A') + 1

            # O or X
            if char == 15 or char == 24:
                y.append(0 if char == 15 else 1)
                x.append([int(ch) for ch in line[1:]])

        except:
            break

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


if __name__ == '__main__':
    itr = 100
    rate = 0.2
    x_train, y_train, x_test, y_test = init()
    nn_obj = NeuralNetwork(16, 10, 1)
    nn_obj.train(x_train, y_train, itr, rate)
    y_est = nn_obj.test(x_test)

    right = 0
    for est, test in zip(y_est, y_test):
        est = np.round(est)
        if est == test:
            right += 1

    print(right / y_test.shape[0])

