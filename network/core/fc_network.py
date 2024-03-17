from collections import OrderedDict

import numpy as np

from network.common.gradient import numerical_gradient
from network.common.layers_old import Sigmoid, ReLU, Affine, SoftmaxWithLoss


class FCNetwork:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', loss='softmax'):
        self.params = {}
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        self.__init_weight(activation)

        self.layers = None
        self.loss_layer = None
        self.build_layers(activation, loss)

    def __init_weight(self, activation):
        all_sizes = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_sizes)):
            if activation.lower() in ["relu"]:
                scale = np.sqrt(2.0 / all_sizes[idx - 1]) # relu 시 초깃값
            elif activation.lower() in ["sigmoid"]:
                scale = np.sqrt(1.0 / all_sizes[idx - 1]) # sigmoid 사용 시 초깃값
            else:
                scale = 0.01

            self.params[f'W{idx}'] = scale * np.random.randn(all_sizes[idx-1], all_sizes[idx])
            self.params[f'b{idx}'] = np.zeros(all_sizes[idx])

    def build_layers(self, activation, loss):
        activations = { 'sigmoid': Sigmoid, 'relu': ReLU }
        self.layers = OrderedDict()

        # hidden layers
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])
            self.layers[f'ActivationFunction{idx}'] = activations[activation]()

        # output layer
        idx = self.hidden_layer_num + 1
        self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])

        # loss layer
        if loss.lower() in ['softmax']:
            self.loss_layer = SoftmaxWithLoss()
        else:
            self.loss_layer = SoftmaxWithLoss() # 확장성을 위해 남겨둠

    def predict(self, x):   # 모든 layer에 대해서 순전파 실행
        for layer in self.layers.values():
            x = layer.forward(x)
        return x    # 아직 softmax & loss 함수는 거치지 않은 상태

    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1) # one-hot인 경우

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads[f"W{idx}"] = numerical_gradient(loss, self.params[f"W{idx}"])
            grads[f"b{idx}"] = numerical_gradient(loss, self.params[f"b{idx}"])

        return grads

    def gradient(self, x, t):
        # use backpropagation

        # forward
        self.loss(x, t) # forward해서 loss를 구해둠. 전파 자체가 목적이기 때문에, 따로 변수에 저장할 필요 x

        # backward
        d_out = 1
        d_out = self.loss_layer.backward(d_out)  # loss함수와 softmax 함수에 대한 d 구함

        layers = list(self.layers.values())     # 모든 layer들을 list로 변환
        layers.reverse()    # 역전파니까 반대로 실행해야 함(Affine2 -> Relu1 -> Affine1)순으로 backward
        for layer in layers:    # 각 층에대해 역전파 실행
            d_out = layer.backward(d_out)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db

        return grads




