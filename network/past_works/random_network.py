import numpy as np
from common.functions import identity_function, sigmoid


def init_network():
    network = {
        "W1": np.random.randn(2, 3),
        "W2": np.random.randn(3, 2),
        "W3": np.random.randn(2, 2),
        "b1": np.random.randn(3),
        "b2": np.random.randn(2),
        "b3": np.random.randn(2),
    }
    return network


def forward(network, activation_function, x):
    a1 = np.dot(x, network["W1"]) + network["b1"]
    a1 = activation_function(a1)
    a2 = np.dot(a1, network["W2"]) + network["b2"]
    a2 = activation_function(a2)
    a3 = np.dot(a2, network["W3"]) + network["b3"]
    y = identity_function(a3)   # 마지막 layer는 활성화함수를 사용하는 규칙을 맞추기 위해서 항등함수를 활성화함수로 사용
    return y


network = init_network()

x = np.array([0, 0])
predict = forward(network, sigmoid, x)

print(predict)