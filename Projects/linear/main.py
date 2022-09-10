import random
import numpy as np
import matplotlib.pyplot as plt

this_size = 1000
this_w = [3, 4]
this_b = 2


def makedata(size, w_known, b_known):
    noise = np.random.normal(loc=0.0, scale=0.5, size=(size, 1))
    untwisted = np.random.normal(loc=0.0, scale=1, size=(size, 2))

    data = np.array([np.array([0, 0])])
    labels = np.array([b_known])

    for i in range(1000):
        data_unit = np.array([untwisted[i][0], untwisted[i][1]])
        data = np.concatenate((data, np.array([data_unit])))
        label_unit = untwisted[i][0] * w_known[0] + untwisted[i][1] * w_known[1] + b_known + noise[i][0]
        labels = np.append(labels, label_unit)
    return data, labels


'''
        Now we create data as 
        w^T*x_i + b + e_i
        with x_i ~ N(0,1), e_i ~ N(0, 0.5)
'''

data, labels = makedata(this_size, this_w, this_b)
print("training data is")
print(data)
print(labels)
print("now the result is")

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(data[:, 0], data[:, 1], labels, marker='o')
# plt.show()
print("-------------------------------------------")

'''
        Define loss function for each object 
        
        Define the function that takes one point and computes its [anti-]gradient
        
        Define loss function for the first initialization
'''


def loss_i(params, weights, bias, result):
    return ((np.dot(params, weights) + bias - result) ** 2 ) /2     # L_i for i-th object


def get_minibatch_return_update(current_weights, current_bias):

    ind = random.randint(0, this_size - 1)
    x_i = data[ind]

    obj_1 = x_i[0]
    obj_2 = x_i[1]

    ans = labels[ind]

    w_1 = current_weights[0]
    w_2 = current_weights[1]

    derivative_1 = 2*(w_1*obj_1 + w_2*obj_2 + current_bias - ans)*obj_1
    derivative_2 = 2*(w_1*obj_1 + w_2*obj_2 + current_bias - ans)*obj_2
    derivative_3 = 2*(w_1*obj_1 + w_2*obj_2 + current_bias - ans)

    updweight1 = w_1 - derivative_1
    updweight2 = w_2 - derivative_2
    updbias = current_bias - derivative_3

    return [updweight1, updweight2], updbias


def loss(weights, bias):
    s = 0
    for ind, params in enumerate(data):
        s += loss_i(params, weights, bias, labels[ind])
    return s


'''
        The function that iterates minibatch updates and goes to minima
'''


def optimize():
    weight = np.random.normal(loc=0.0, scale=0.5, size=(2, 1))
    bias = np.random.normal(loc=0.0, scale=1)

    q_over = loss(weight, bias)


    for i in range(10000):
        weight, bias  = get_minibatch_return_update(weight, bias)


    print(weight, bias)


# optimize()
weight1 = np.random.normal(loc=0.0, scale=0.5)
weight2 = np.random.normal(loc=0.0, scale=0.5)
bias = np.random.normal(loc=0.0, scale=0.5)
weights = [weight1, weight2]
print(weights)

ind = 5

((np.dot(data[ind], weights) + bias - labels[ind]) ** 2 ) /2

