import numpy as np
import matplotlib.pyplot as plt

data = []
teacher = []
for_print = np.linspace(0, 20, 300)  # for drawing pictures later

f = open("GT_samples_timeseries.txt", "r")
for line in f:
    whatss = line.split()
    num1 = int(whatss[0])
    num2 = int(whatss[1])
    data.append(num1)
    teacher.append(num2)
print(data)

plt.plot(data, teacher, 'o-', lw=1.5, alpha=0.6, label='initial data')

sum = 0
for x in teacher:
    sum += x
print("expectancy prediction is", sum / len(data))
plt.plot(20, sum / len(data), 'o-', 'r-', label='expectancy prediction')

'''
linear model taught by the last two points
fetching relevant data 
solving directly
'''

features = np.array([[1, data[len(data) - 2]], [1, data[len(data) - 1]]])
tech = np.array([[teacher[18]], [teacher[19]]])

print(features.shape)
print(features)
print(tech.shape)
print(tech)
w_ML = (np.linalg.inv((features.transpose() @ features)) @ features.transpose()) @ tech
print(w_ML)

plt.plot(for_print, w_ML[0] + w_ML[1] * for_print, 'r-', lw=0.9, alpha=0.6, label='linear model')
plt.plot(20, w_ML[0] + w_ML[1] * 20, 'g-', alpha=0.6, label='linear model')

'''
quadratic model taught by the last three points
solving directly as well
'''
q_features = np.array(
    [[1, data[len(data) - 3], data[len(data) - 3] ** 2], [1, data[len(data) - 2], data[len(data) - 2] ** 2],
     [1, data[len(data) - 1], data[len(data) - 1] ** 2]])
q_tech = [[teacher[17]], [teacher[18]], [teacher[19]]]

w_q_ML = (np.linalg.inv((q_features.transpose() @ q_features)) @ q_features.transpose()) @ q_tech
print(w_q_ML)
plt.plot(for_print, w_q_ML[0] + w_q_ML[1] * for_print + w_q_ML[2] * for_print * for_print, 'r-', lw=0.9, alpha=0.6,
         label='quadratic model')
plt.plot(20, w_q_ML[0] + w_q_ML[1] * 20 + w_q_ML[2] * 20, 'o-', 'm-', alpha=0.6)

'''third solution, Lasso regression for 20-power polynomial'''

weights = np.array([1 for i in range(20)])


def features_for(x):
    return np.array([x ** n for n in range(20)])


def loss_function():
    s = 0
    for i in range(20):
        s += (teacher[i] - features_for(data[i]).transpose() @ weights) ** 2
    return s + np.linalg.norm(weights)


def optimize(w):
    while True:
        minus_grad = np.zeroes(20)
        for i in range(20):
            minus_grad[i] = 2 * data[i] * (teacher[i] - data[i])


print("answers for: \n")
print("linear regression: ")
print(int(w_ML[0] + w_ML[1] * 20))
print("quadratic regression: ")
print(int(w_q_ML[0] + w_q_ML[1] * 20 + w_q_ML[2] * 20 * 20))

# plt.legend()
plt.show()
