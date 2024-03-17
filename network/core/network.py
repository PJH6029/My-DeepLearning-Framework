from collections import OrderedDict

import numpy as np
from common.layers import *


class Network:
    def __init__(self, layers, loss='softmax'):
        self.layers = None
        self.loss_layer = None
        self.use_batch_norm = False
        self.use_dropout = False


        self.build_layers(layers, loss)

    def build_layers(self, layers, loss):
        self.layers = OrderedDict()
        for i, layer in enumerate(layers):
            layer_name = layer.__class__.__name__
            self.layers[f'{layer_name}{i}'] = layer

            if layer_name == BatchNormalization.__name__:
                self.use_batch_norm = True

            if layer_name == Dropout.__name__:
                self.use_dropout = True

        # loss layer
        if loss.lower() in ['softmax']:
            self.loss_layer = SoftmaxWithLoss()
        else:
            self.loss_layer = SoftmaxWithLoss()  # 확장성을 위해 남겨둠

    def eval(self):
        pass # batch norm, dropout train flag 조절

    def predict(self, x):
        # does not include softmax & loss
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)  # one-hot인 경우

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward (including the loss layer)
        self.loss(x, t)

        # backward
        d_out = 1
        d_out = self.loss_layer.backward(d_out)
        for layer in reversed(self.layers.values()):
            d_out = layer.backward(d_out)

        # configure grads
        grads = {}
        for layer_key in self.layers.keys():
            if hasattr(self.layers[layer_key], "d_weight"):
                # Affine or Conv layer
                grads[layer_key] = {
                    'd_weight': self.layers[layer_key].d_weight,
                    'd_bias': self.layers[layer_key].d_bias,
                }

            elif self.use_batch_norm and hasattr(self.layers[layer_key], "d_gamma"):
                # Batch norm
                grads[layer_key] = {
                    'd_gamma': self.layers[layer_key].d_gamma,
                    'd_beta': self.layers[layer_key].d_beta,
                }
        return grads