import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import *

class Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.parameter = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),  # 784 x n 사이즈 가중치 설정
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),  # n x 10(0~9) 사이즈 가중치 설정
            "b2": np.zeros(output_size),
        }
        self.layers = OrderedDict()  # 순서대로 신경망이 실행되어야하므로 순서가 있는 dict 사용
        self.layers["Affine1"] = Affine(self.parameter["W1"], self.parameter["b1"])     # 각각의 layer 생성
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.parameter["W2"], self.parameter["b2"])
        self.softmax_layer = SoftmaxWithLoss()

    def predict(self, x):   # 모든 layer에 대해서 순전파 실행
        for layer in self.layers.values():
            x = layer.forward(x)
        return x    # 아직 softmax 함수는 거치지 않은 상태

    def loss(self, x, t):   # softmax함수와 loss함수 동시에 순전파
        y = self.predict(x)
        return self.softmax_layer.forward(y, t)

    def accuracy(self, x, t):   # 정확도를 구하는 함수
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):   # backpropagation 기울기 구하기
        y = self.loss(x, t)
        d_out = 1
        d_out = self.softmax_layer.backward(d_out)  # loss함수와 softmax 함수에 대한 d 구함

        layers = list(self.layers.values())     # 모든 layer들을 list로 변환
        layers.reverse()    # 역전파니까 반대로 실행해야 함(Affine2 -> Relu1 -> Affine1)순으로 backward
        for layer in layers:    # 각 층에대해 역전파 실행
            d_out = layer.backward(d_out)

        grads = {
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
        }
        return grads

    def numerical_gradient(self, x, t): # 수치미분 기울기 구하기
        loss = lambda W: self.loss(x, t)    # loss함수를 W에 대한 함수로 변환

        grads = {
            "W1": numerical_gradient(loss, self.parameter["W1"]),   # loss함수에 대한 기울기를 구함(각각의 가중치에 대해서 따로 기울기를 구함)
            "b1": numerical_gradient(loss, self.parameter["b1"]),
            "W2": numerical_gradient(loss, self.parameter["W2"]),
            "b2": numerical_gradient(loss, self.parameter["b2"]),
        }
        return grads
