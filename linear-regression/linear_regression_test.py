from linear_regression import LinearRegression
import numpy as np
import csv
import matplotlib.pyplot as plt


# initialize the training and test dataset
def init():
    train_x, train_y, test_x, test_y = [], [], [], []
    with open('data_lin_reg\\train.csv', 'r') as train_file:
        reader = csv.DictReader(train_file)
        for row in reader:
            train_y.append(int(row['price'])/1e7)
            train_x.append([int(row['sqft_living'])/1e4])

    with open('data_lin_reg\\test.csv', 'r') as test_file:
        reader = csv.DictReader(test_file)
        for row in reader:
            test_y.append(int(row['price'])/1e7)
            test_x.append([int(row['sqft_living'])/1e4])

    return train_x, train_y, test_x, test_y


def fig_plot(train_x, train_y, test_x, test_y, theta, lbl):
    plt.title('House Price vs Living Area', fontweight='bold')
    plt.xlabel('Living Area/$10^4foot^2$')
    plt.ylabel('Price/$10^7$')
    plt.plot(train_x, train_y, 'r.', label='training data')
    plt.plot(test_x, test_y, 'g.', label='test data')

    # plot the prediction line parametered by theta
    x = np.linspace(0, 1.4)
    y = x * theta[1] + theta[0]
    plt.plot(x, y, label=lbl)
    plt.legend(loc='upper left', numpoints=1)

    plt.show()


# calculate the root-mean-square-error
def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean()) * 1e7


def main():
    train_x, train_y, test_x, test_y = init()
    alpha = 0.0001
    LRobj = LinearRegression(train_x, train_y, alpha)
    qx = np.array(test_x)
    qx = np.concatenate((np.ones((qx.shape[0], 1)), qx), axis = 1)
    tar = np.array(test_y)

    print("=== Gradient descent ===")
    J, itr = LRobj.batch_gradient_descent()
    prediction = LRobj.predict(qx)
    print("itr:", itr)
    print("theta:", LRobj.theta, "error:", J)
    print("prediction:", prediction)
    print("target:", tar)
    print("RMSE:", rmse(prediction, tar))
    fig_plot(train_x, train_y, test_x, test_y, LRobj.theta, "Gradient Descent")

    print("=== Normal equation ===")
    J = LRobj.normal_equ()
    prediction = LRobj.predict(qx)
    print("theta:", LRobj.theta, "error:", J)
    print("prediction:", prediction)
    print("target:", tar)
    print("RMSE:", rmse(prediction, tar))
    fig_plot(train_x, train_y, test_x, test_y, LRobj.theta, "Normal Equation")

    print("=== Newton method ===")
    J, itr = LRobj.newton_method()
    prediction = LRobj.predict(qx)
    print("itr:", itr)
    print("theta:", LRobj.theta, "error:", J)
    print("prediction:", prediction)
    print("target:", tar)
    print("RMSE:", rmse(prediction, tar))
    fig_plot(train_x, train_y, test_x, test_y, LRobj.theta, "Newton Method")

main()
