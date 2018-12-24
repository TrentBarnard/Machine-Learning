import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1/(1+numpy.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

data=[[3, 1.5, 1],
[2, 1, 0],
[4, 1.5, 1],
[3, 1, 0],
[3.5, .5, 1],
[2, .5, 0],
[5.5, 1, 1],
[1, 1, 0]]

mystery_flower = [4.5, 1]

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

T = np.linspace(-6,6,100)
Y = sigmoid_p(T)

# plt.plot(T, sigmoid(T), c='r')
# plt.plot(T, sigmoid_p(T), c='g')

# plot scatter graph
for i in range(len(data)):
    point = data[i]
    color = 'r'
    if point[2] == 0:
        color = 'b'
    plt.scatter(point[0], point[1], c=color)


# training loop
a = 0.9
costs = []

for i in range(100000):
    ri = np.random.randint(len(data))
    point = data[ri]

    z = point[0] * w1 + point[1] * w2 + b
    h = sigmoid(z)

    target = point[2]
    cost = (target - h)**2
    costs.append(cost)

    dcost_pred = 2 * (h - target)
    dpred_dz = sigmoid_p(z)

    dcost_dz = dcost_pred * dpred_dz

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - a * dcost_dw1
    w2 = w2 - a * dcost_dw2
    b = b - a * dcost_db

for i in range(len(data)):
    z = data[i][0] * w1 + data[i][1] * w2 +b
    prediction = sigmoid(z)
    print(data[i])
    print('Prediction is {}'.format(prediction))

import os
os.system("say hi")

def whitch_flower(length, width):
    z = length * w1 + width * w2 + b
    prediction = sigmoid(z)
    if prediction < 0.5:
        os.system("say blue")
    else:
        os.system("say red")

for x in np.linspace(0,6,20):
    for y in np.linspace(0,5,20):
        pred = sigmoid(w1 * x + w2 * y + b)
        c = 'b'
        if pred > 0.5:
            c = 'r'
        plt.scatter([x], [y], c=c, alpha = 0.2)

plt.show()

whitch_flower(1,1)
