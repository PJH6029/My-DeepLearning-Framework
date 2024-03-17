import sys, os
import numpy as np
from common.functions import softmax, sigmoid
from common.functions import cross_entropy_error
from common.gradient import numerical_gradient
sys.path.append(os.pardir)


class Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): # 기본적인 파라미터, 하이퍼 파라미터 설정
        self.parameter = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),   # 784 x n 사이즈 가중치 설정
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),  # n x 10(0~9) 사이즈 가중치 설정
            "b2": np.zeros(output_size),
        }

    def predict(self, x):   # 가중치를 불러오고 계산 실행
        W1, W2 = self.parameter["W1"], self.parameter["W2"]
        b1, b2 = self.parameter["b1"], self.parameter["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):   # 결과값을 기반으로 loss 계산
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)    # 각 행에 대해서 최대값들을 뽑아 옴
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])   # 일치하는 수를 전체의 수로 나눠줌

    def numerical_gradient(self, x, t):
        loss = lambda W: self.loss(x, t)    # loss함수를 W에 대한 함수로 변환

        grads = {
            "W1": numerical_gradient(loss, self.parameter["W1"]),  # loss함수에 대한 기울기를 구함(각각의 가중치에 대해서 따로 기울기를 구함)
            "b1": numerical_gradient(loss, self.parameter["b1"]),
            "W2": numerical_gradient(loss, self.parameter["W2"]),
            "b2": numerical_gradient(loss, self.parameter["b2"]),
        }
        return grads


if __name__ == "__main__":
    net = Network(2, 20, 3)
    x = np.array([0.6, 0.9])
    y = net.predict(x)
    print(y)
    print(net.loss(x, np.array([1, 0, 0])))
