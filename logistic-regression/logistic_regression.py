import numpy as np

# initialize input x and target y
def init_traning_set():
    array_x = [[1], [2], [1.8], [3], [4], [5], [6], [6.5], [7], [7.2]]
    array_y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    x = np.array(array_x)
    y = np.array(array_y)
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return x, y


# initialize query qx
def init_query():
    qx = [8]
    qx = np.array(qx)
    qx = np.concatenate((np.ones(1), qx), axis = 0)
    return qx

# define the Hypothesis Function as Sigmoid Function
def hyp_func(x, theta):
    z = np.dot(x, theta)
    return 1 / (1 + np.exp(-z))
    ##########################################
    ##  Below is 'Perception Function', but ##
    ##   divide-by-zero error will occur.   ##
    ##########################################
    #return np.array([int(i >= 0) for i in z])

# calculate logL(theta): Likelihood of theta
# ========================== #
#    TO MAXIMIZE L(theta)    #
# ========================== #
def calc_likelihood(theta, x, y):
    tmp = hyp_func(x, theta)
    l_theta = np.dot(y, np.log(tmp)) + np.dot((1 - y), np.log(1 - tmp))
    return l_theta

# Batch gradient descent algorithm
def batch_gradient_descent(x, y):
    m = y.shape[0]
    n = x.shape[1]
    delta = 0.00001 # error checker
    alpha = 0.001   # learning rate
    itr = 1         # itertion counter

    # initialization of theta
    #theta = np.random.random(n)
    theta = np.zeros(n)

    # calc init likelihood 'L0'
    L0 = calc_likelihood(theta, x, y)

    while 1:
        # almost the same with the formula in linear regression
        theta += alpha * np.dot(np.transpose(x), y - hyp_func(x, theta))

        # calc l(theta) after each iteration
        L = calc_likelihood(theta, x, y)
        if abs(L0 - L) < delta:
            print("Convergent!")
            break
        else:
            L0 = L

        itr += 1

    return theta, L, itr

def test():
    x, y = init_traning_set()
    qx = init_query()

    theta, L, itr = batch_gradient_descent(x, y)
    result = hyp_func(qx, theta)
    print("--- LOGISTIC REGRESSION RESULT ---")
    print("=== Gradient descent ===")
    print("itr:", itr)
    print("theta:", theta, "l(theta):", L)
    print("result:", result)

test()
