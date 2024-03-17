import numpy as np

class Optimizer:

    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        pass

class SGD(Optimizer):
    def update(self, network, grads):
        for layer_key in grads:
            grad = grads[layer_key]
            for grad_key in grad:
                val = getattr(network.layers[layer_key], grad_key)
                setattr(network.layers[layer_key], grad_key, val - self.lr * grad[grad_key])
                # network.layers[layer_key][grad_key] -= self.lr * grad[grad_key]

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

