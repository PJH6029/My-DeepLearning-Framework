import sys, os
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import softmax, sigmoid

sys.path.append(os.pardir)  # 부모의 디렉터리에서 파일을 가져올 수 있도록 허용


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
    # normalize = 정규화(0~255 -> 0~1), flatten = 1차원 배열로 폄, one_hot_label = 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    return x_train, t_train, x_test, t_test


def init_network():
    with open("../../dataset/sample_weight.pkl", "rb") as f:  # 파일을 읽고 자동으로 닫아주는 코드(rb는 읽을 포맷중 하나)
        network = pickle.load(f)    # pickle은 파이썬이 데이터를 저장하는 모듈

    return network


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1     # (k, 784) x (784, 50)
    a1 = sigmoid(a1)
    a2 = np.dot(a1, W2) + b2     # (k, 50) x (50, 100)
    a2 = sigmoid(a2)
    a3 = np.dot(a2, W3) + b3    # (k, 100) x (100, 10)
    y = softmax(a3)

    return y


def get_batch(batch_size):
    train_size = x_train.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    return x_batch, t_batch


# forward 실행
x_train, t_train, x_test, t_test = get_data()
network = init_network()    # 네트워크 지정
print(np.argmax(forward(network, x_test[1])))   # 예측값을 출력(np.argmax: one_hot_label을 숫자로 변환)
print(np.argmax(t_test[1]))    # 정닶을 출력


# 정확도 구하는 함수
def evaluate_accuracy(batch_size):
    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_size):  # 데이터를 batch_size만큼씩 띄워서 자름
        x_batch = x_test[i:i+batch_size]     # 이미지들을 batch_size단위로 자름
        y_batch = forward(network, x_batch)
        predict = np.argmax(y_batch, axis=1)    # 행에 대해 확률이 가장 높은 원소 = 추론한 정답을 가져옴
        answer = np.argmax(t_test[i:i+batch_size], axis=1)

        accuracy_cnt += np.sum(predict == answer)  # 각 원소가 같다는 조건의 boolean 리스트의 값들을 전부 합침

    print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))


evaluate_accuracy(100)


# # 이거 작동 안됨
# def img_show():
#     x_test, t_test = get_data()
#     print(t_test[0])
#     image = x_test[0]
#
#     image = image.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
#     pil_img = Image.fromarray(np.uint8(image))    # 이미지를 pil용 데이터 객체로 변환
#     pil_img.show()
#
# img_show()