import numpy as np

class Optimizer:

    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        pass

class SGD(Optimizer):
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum(Optimizer):
    pass

class Adam(Optimizer):
    pass

class RMSprop(Optimizer):
    pass

class AdaGrad(Optimizer):
    pass

class Nesterov(Optimizer):
    pass

