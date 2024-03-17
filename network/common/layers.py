from network.common.functions import *


class Affine:    # 일반적인 y = xw + b 네트워크
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None # for handling tensors

        self.dW = None
        self.db = None

    def forward(self, x):   # 계산한 결과를 return(순전파)
        # tensor 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1) # flatten except for batch dimension
        self.x = x

        y = np.dot(x, self.W) + self.b
        return y

    def backward(self, d_out):  # 위 층으로부터 온 기울기를 받아서 다시 x에 대한 기울기를 return
        self.dW = np.dot(self.x.T, d_out)   # Affine layer의 가중치 기울기
        self.db = np.sum(d_out, axis=0)     # Affine layer의 편향 기울기

        dx = np.dot(d_out, self.W.T)
        dx = dx.reshape(*self.original_x_shape) # tensor 대응
        return dx


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):   # 배치를 위한 relu 순전파
        out = x.copy()
        self.mask = out <= 0
        out[self.mask] = 0
        return out

    def backward(self, d_out):  # relu 역전파
        d_out[self.mask] = 0    # x가 0보다 작으면 0, 0이상이면 1*d_out
        dx = d_out
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):   # sigmoid 순전파
        self.out = sigmoid(x)
        return self.out

    def backward(self, d_out):  # sigmoid 역전파
        return self.out * (1 - self.out) * d_out    # sigmoid 미분한 식은 (1 - a) * a, a는 sigmoid(x)값


class SoftmaxWithLoss:  # 항등함수는 오차제곱합을 손실함수로, softmax는 cross_entropy를 손실함수로 사용
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, d_out=1):    # softmax와 cross_entropy함수를 하나의 함수로 보고 미분하면 ((예측값 - 정답) / 정답 개수) 가 나옴
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size: # ont-hot label
            return (d_out * (self.y - self.t)) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1 # 정답인 위치마다 1(=t의 value)만큼 빼줌
            return dx / batch_size

