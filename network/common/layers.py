import numpy as np
import network.common.initialization as init
from network.common.functions import sigmoid, softmax, cross_entropy_error


class Affine:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

        self.weight = np.zeros((in_size, out_size))
        self.bias = np.zeros(out_size)

        self.x = None
        self.original_x_shape = None  # for handling tensors

        self.d_weight = None
        self.d_bias = None

        self.init_params()

    def init_params(self, distrib='kaiming'):
        fan_in, fan_out = init.calculate_fans(self.weight.shape)
        if distrib.lower() in ['kaiming']:
            scale = np.sqrt(2.0 / fan_in) # He 분포
        elif distrib.lower() in ['xavier']:
            scale = np.sqrt(1.0 / fan_in) # Xavier 분포
        else:
            scale = 0.01

        self.weight = scale * np.random.randn(*self.weight.shape)
        self.bias = np.zeros(*self.bias.shape) # TODO 나중을 위해 일단 둠

    def forward(self, x):  # 계산한 결과를 return(순전파)
        # tensor 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # flatten except for batch dimension
        self.x = x

        y = np.dot(x, self.weight) + self.bias
        return y

    def backward(self, d_out):  # 위 층으로부터 온 기울기를 받아서 다시 x에 대한 기울기를 return
        self.d_weight = np.dot(self.x.T, d_out)   # Affine layer의 가중치 기울기
        self.d_bias = np.sum(d_out, axis=0)     # Affine layer의 편향 기울기

        d_x = np.dot(d_out, self.d_weight.T)
        d_x = d_x.reshape(*self.original_x_shape) # tensor 대응
        return d_x

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
        d_x = d_out
        return d_x

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
            d_x = self.y.copy()
            d_x[np.arange(batch_size), self.t] -= 1 # 정답인 위치마다 1(=t의 value)만큼 빼줌
            return d_x / batch_size

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True):
        if train_flag:
            # 학습 시에는 dropout_ratio만큼 뉴런을 off
            # rand: [0, 1)에서 uniformly / randn: N(0, 1)에서

            # mask: x.shape과 같은 T/F array
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 학습 시에 dout_ratio만큼 꺼져 있었다는 것을 보정해주기 위해, 값을 scaling
            return x * (1.0 - self.dropout_ratio)

    def backward(self, d_out):
        # 꺼진 node는 전파하지 않고, 켜져있던 node는 그대로 전파
        return d_out * self.mask

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.1, running_mean=None, running_var=None):
        # parameters
        self.gamma = gamma
        self.beta = beta

        self.momentum = momentum
        self.input_shape = None # conv는 4차원, fc는 2차원

        # test 단계에서 사용되는 변수
        # test 시에는, input으로 들어온 평균과 분산 대신, 학습 단계에서 사용한 지표들의 이동평균을 이용하여 정규화
        self.running_mean = running_mean
        self.running_var = running_var

        # backward에 사용할 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.d_gamma = None
        self.d_beta = None

    def forward(self, x, train_flag=True):
        self.input_shape = x.shape

        if x.ndim != 2: # conv layer input
            N, C, H, W = x.shape
            x = x.reshape(N, -1) # flatten

        out = self.__forward(x, train_flag) # __는 python에서 method를 private하게 만듦
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flag):
        # x: flattened array (N, D)
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flag:
            mu = x.mean(axis=0) # batch의 평균
            xc = x - mu
            var = np.mean(np.square(xc), axis=0) # batch의 분산
            std = np.sqrt(var + 10e-7) # division error 방지를 위한 delta
            xn = xc / std # 정규화

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            # test에 사용할 평균/분산 업데이트
            # see https://pytorch.org/docs/master/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 이동 평균/분산으로 정규화
            xc = x - self.running_mean
            std = np.sqrt(self.running_var + 10e-7)
            xn = xc / std

        out = self.gamma * xn + self.beta # scale 후 shift
        return out

    def backward(self, d_out):
        if d_out.ndim != 2:
            # conv layer
            N, C, H, W = d_out.shape
            d_out = d_out.reshape(N, -1)

        d_x = self.__backward(d_out)

        return d_x.reshape(*self.input_shape)

    def __backward(self, d_out):
        # 논문에 있는거 그냥 거의 배껴옴
        d_beta = d_out.sum(axis=0)
        d_gamma = np.sum(self.xn * d_out, axis=0)

        d_xn = self.gamma * d_out
        d_xc = d_xn / self.std

        d_std = -np.sum((d_xn * self.xc) / (self.std * self.std), axis=0)
        d_var = 0.5 * d_std / self.std

        d_xc += (2.0 / self.batch_size) * self.xc * d_var

        d_mu = np.sum(d_xc, axis=0)
        d_x = d_xc - d_mu / self.batch_size

        # GD update용
        self.d_gamma = d_gamma
        self.d_beta = d_beta

        # backward pass용
        return d_x

class Convolution:
    pass # TODO