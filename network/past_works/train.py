import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from network.past_works.backprop_simplenet import Network

# 데이터들 불러옴
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

# hyper파라미터 지정
hyper_params = {
    "iters_num": 10000,
    "train_size": x_train.shape[0],
    "batch_size": 128,
    "learning_rate": 1e-2,
}
iter_per_epoch = max(hyper_params["train_size"] // hyper_params["batch_size"], 1)   # 한 에폭당 몇번 학습이 돌아가는지
# epoch 수 = 전체 데이터를 사용한 수 / 학습 데이터 수

# net 생성
network = Network(input_size=784, hidden_size=64, output_size=10)

# 학습시작, 에폭이 몇번인지 print
print(f"train start, num_epochs: {hyper_params['iters_num'] // iter_per_epoch}")

train_loss_list = []
for i in range(hyper_params["iters_num"]):  # iter_nums만큼 학습을 돌린다
    # 배치 생성(random으로 indexing)
    mask = np.random.choice(hyper_params["train_size"], hyper_params["batch_size"])
    x_batch = x_train[mask]
    t_batch = t_train[mask]
    # grad 구하고 파라미터 갱신
    grad = network.gradient(x_batch, t_batch)
    for key in grad.keys():     # grad.keys = W1, b1, W2, b2
        network.parameter[key] -= hyper_params["learning_rate"] * grad[key]     # 각각의 파라미터들에 대해서 갱신
    # loss 계산
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 한 epoch마다 정확도 구함
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(f"epoch: {i // iter_per_epoch}, train_acc: {train_acc}, test_acc: {test_acc}")

plt.plot(train_loss_list)
plt.show()

