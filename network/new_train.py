import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from network.common.trainer import Trainer
from network.core.fc_network import FCNetwork

# 데이터들 불러옴
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

# hyper파라미터 지정
hyper_params = {
    "epochs": 5,
    "train_size": x_train.shape[0],
    "batch_size": 256,
    "learning_rate": 1e-2,
}
9
def train(hyper_params):
    # net = Network(layers=[
    #     Affine(28*28, 128),
    #     ReLU(),
    #     Affine(128, 64),
    #     ReLU(),
    #     Affine(64, 10),
    # ])

    network = FCNetwork(input_size=784, hidden_size_list=[128, 64], output_size=10)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=hyper_params['epochs'], batch_size=hyper_params['batch_size'],
                      optimizer='sgd', optimizer_params={'lr': hyper_params['learning_rate']})
    trainer.train()

    return (
        trainer.train_loss_list,
        trainer.train_acc_list,
        trainer.test_acc_list
    )

def plot(train_loss_list, train_acc_list, test_acc_list):
    f, axes = plt.subplots(1, 2)
    axes[0].set_title('Train Loss')
    axes[0].plot(train_loss_list)

    axes[1].set_title('Accuracy')
    axes[1].plot(train_acc_list, label='train acc')
    axes[1].plot(test_acc_list, label='test acc')
    axes[1].legend()
    plt.show()

train_loss_list, train_acc_list, test_acc_list = train(hyper_params)
plot(train_loss_list, train_acc_list, test_acc_list)