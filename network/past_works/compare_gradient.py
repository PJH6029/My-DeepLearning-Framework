import numpy as np
from dataset.mnist import load_mnist
from network.past_works.backprop_simplenet import Network

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

network = Network(input_size=784, hidden_size=64, output_size=10)

x_batch = x_train[:2]
t_batch = t_train[:2]

numerical = network.numerical_gradient(x_batch, t_batch)
backprop = network.gradient(x_batch, t_batch)

for key in numerical.keys():
    print(np.average(numerical[key] - backprop[key]))   # 역전파와 수치미분으로 구한 가중치들의 차이를 print
