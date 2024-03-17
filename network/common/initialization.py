import numpy as np

def calculate_fans(shape):
    if len(shape) < 2:
        raise ValueError("Fans cannot be computed for array with fewer than 2 dimensions")

    # len == 2인 경우(fc) : (in_features, out_features)
    # len > 2인 경우(conv): (c_out, c_in, *kernel_size)
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) ==2 else shape[0]
    return fan_in, fan_out