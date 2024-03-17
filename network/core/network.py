import numpy as np

class Network:
    def __init__(self, layers, activation='relu', loss='softmax'):
        self.__layers = layers
        self.layers = None # build layers에서
        self.loss_layer = None



    def eval(self):
        pass

    def predict(self, x):
        # does not include softmax & loss
        for layer in self.layers:
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
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

        # configure grads
        # TODO
        return None