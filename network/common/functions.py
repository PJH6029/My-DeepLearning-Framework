import numpy as np


def step_function(x):
    y = x > 0   # = [true, false, true, true, ...]
    return y.astype(np.int32)


def sigmoid(x):  # for batch & single
    return 1 / (1 + np.exp(-x))    # 브로드캐스팅 가능


def relu(x):     # for single
    return np.maximum(x, 0)


def identity_function(x):
    return x


def softmax(x):     # for batch
    if x.ndim == 1:
        m = np.max(x)
        exp_x = np.exp(x - m)  # 위 식에 임의의 값을 빼거나 더해도 결과값은 같다를 이용해서 값을 줄임(e^100 -> e^2)
        return exp_x / np.sum(exp_x)  # 이렇게 하면 합이 1이 되어서 결과값이 확률이 된다

    x = x.T
    m = np.max(x, axis=0)   # 배치로 들어오면 각 데이터에 대해서 최대값을 구해준다
    x = x - m
    exp_y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return exp_y.T

def sum_squares_error(y, t):    # 오차제곱합 for batch & single
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):  # for batch & single
    if y.ndim == 1:
        # single data handling
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # one-hot이 아닌 label 처리
    if t.size != y.size:
        num_classes = y.shape[1]
        t = np.eye(num_classes)[t]

    delta = 1e-7    # log0은 음의 무한대이므로 오류를 방지하기 위한 아주 작은 값

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + delta)) / batch_size

# 정확도 대신 손실함수를 사용하는 이유 = 학습에서 미분을 해야하는데 정확도를 쓰면 대부분 미분값이 0이 나오기때문에 경사하강법이 적용이 안됨



if __name__ == "__main__":      # 이 파일에서만 실행시켜야 작동하는 조건문
    # a = softmax(np.array([-1, 2, 4]))
    # print(sum(a))
    _y = np.array([[0.1, 0.2, 0.7], [0.0, 0.9, 0.1]])
    _t = np.array([1, 0])
    print(cross_entropy_error(_y, _t))