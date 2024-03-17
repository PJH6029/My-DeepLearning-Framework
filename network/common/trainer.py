from network.common.optimizer import *

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, batch_size=128,
                 optimizer='SGD', optimizer_params={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.verbose = verbose

        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size

        # epoch마다 test 시에 몇개나 test할지. 안넣으면 모든 데이터셋 다 들어감
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_dict = {
            'sgd': SGD, 'momentum': Momentum, 'nesterov':Nesterov,
            'adagrad': AdaGrad, 'adam': Adam, 'rmsprop': RMSprop,
        }
        self.optimizer = optimizer_dict[optimizer.lower()](**optimizer_params)

        # train size
        self.train_size = x_train.shape[0]
        self.iters_per_epoch = max(self.train_size // batch_size, 1)
        self.max_iter = int(self.epochs * self.iters_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        # monitor result
        self.train_loss_list = []
        self.train_acc_list = []
        # self.test_loss_list = []
        self.test_acc_list = []

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print(f"iter {self.current_iter} / Train loss: {loss}")

        if self.current_iter % self.iters_per_epoch == 0:
            # increase epoch
            self.current_epoch += 1

            # evaluate accuracy
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = x_train_sample[:t], t_train_sample[:t]
                x_test_sample, t_test_sample = x_test_sample[:t], t_test_sample[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f"=== epoch: {self.current_epoch}, train_acc: {train_acc}, test_acc: {test_acc}")
        self.current_iter += 1


