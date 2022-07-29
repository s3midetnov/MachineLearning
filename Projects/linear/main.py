import random
import numpy as np
import matplotlib.pyplot as plt

this_size = 1000
this_w = [3, 4]
this_b = 2

def makedata(size, w_known, b_known):
    noise = np.random.normal(loc=0.0, scale=0.01, size=(size, 1))
    untwisted = np.random.normal(loc=0.0, scale=1, size=(size, 2))

    data = np.array([np.array([0, 0])])
    labels = np.array([b_known])

    for i in range(1000):
        data_unit = np.array([untwisted[i][0], untwisted[i][1] ])
        data = np.concatenate((data, np.array([data_unit])))
        label_unit = untwisted[i][0] * w_known[0] + untwisted[i][1] * w_known[1] + b_known + noise[i][0]
        labels = np.append(labels,  label_unit)
    return data, labels


data, labels = makedata(this_size, this_w, this_b)
print(data)
print(labels)


def loss_i(params, weights, bias, result):
    return (np.mul(params, weights) + bias - result) ** 2  # the prediction for i-th object

def grad_loss_i(params, weights, bias, result):
    return 0


def loss(weights, bias):
    s = 0
    for params in data:
        s += loss_i(params, weights, bias)
    return s



def optimize(lam, tempo):
    weight = np.random.normal(loc=0.0, scale=1, size=(2, 1))
    bias = np.random.normal(loc=0.0, scale=1)
    q_over = loss(weight, bias)
    for i in range(10000):
        ind = random.randint(0, len(data) - 1)
        x_i = data[ind]
        ei = loss_i(x_i, weight, bias, labels[ind])

        weight = weight - tempo*grad_loss_i(x_i, weight, bias, labels[ind])
        q_over = q_over*(1-lam) + lam*ei
