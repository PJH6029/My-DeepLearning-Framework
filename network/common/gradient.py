import numpy as np

def numerical_diff(f, x):   # 수치미분(근사치를 구함)
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)


# 기울기 구함
def numerical_gradient(f, x):   # 편미분
    h = 1e-4
    grad = np.zeros_like(x)     # X와 같은 형상의 배열을 생성

    # nd array iteration을 위한 iterator
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        init_x = float(x[idx])
        x[idx] = init_x + h
        fxh1 = f(x) # f(x+h)

        x[idx] = init_x - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = init_x # 복원
        it.iternext()

    return grad
