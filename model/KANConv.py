import torch
import numpy as np
import math
import time
from functools import wraps

from KANLinear import *

def timing_decorator(func):
    """
    A decorator to measure the execution time of a function.
    :param func: The function to measure.
    :return: Wrapped function with timing functionality.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 使用更精确的计时
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
        return result
    return wrapper

def calc_out_dims_1d(vector, kernel_size, stride, padding, dilation):
    """
    Calculate the output dimensions for 1D convolution.
    :param vector: Input tensor with shape (batch_size, n_channels, length)
    :param kernel_size: Size of the kernel (int)
    :param stride: Stride of the convolution (int)
    :param dilation: Dilation of the kernel (int)
    :param padding: Padding added to both sides of the input (int)
    :return: Tuple (length_out, batch_size, n_channels)
    """
    batch_size, in_channels, length = vector.shape
    # print(batch_size, in_channels, length)

    # Calculate the output length
    length_out = np.floor(
        (length + 2 * padding - (dilation*(kernel_size-1)) - 1) / stride
    ).astype(int) + 1

    return length_out, batch_size, in_channels


@timing_decorator
def KConv1d(input_tensor, kernel, out_channels, conv_core, stride=1, padding=0, dilation=1):
    """
    Perform 1D convolution using PyTorch.
    :param conv_core:
    :param out_channels:
    :param input_tensor: 1D input tensor.
    :param kernel: 1D kernel tensor.
    :param stride: Stride of the convolution.
    :param padding: Padding added to both sides of the input.
    :return: Convolved output.
    """
    # 计算维度
    length_out, batch_size, in_channels = calc_out_dims_1d(input_tensor, kernel,
                                                          stride=stride, padding=padding, dilation=dilation)

    # Add padding to the input tensor
    if padding > 0:
        input_tensor = torch.nn.functional.pad(input_tensor, (padding, padding))
    # print(input_tensor)

    # print(length_out, batch_size, in_channels)
    # Perform convolution
    output = torch.zeros((batch_size, out_channels, length_out))
    for b in range(batch_size):
        batch_tmp = torch.zeros((out_channels, length_out))
        for out in range(out_channels):
            out_tmp = torch.zeros(length_out)
            for i in range(length_out):
                start_time = time.perf_counter()
                for input in range(in_channels):
                    index = i * stride
                    out_tmp[i] += conv_core[out][input](input_tensor[b, input, index:index+kernel]).item()
            batch_tmp[out, :] = out_tmp
        output[b, :, :] = batch_tmp

    # Squeeze the output to remove batch and channel dimensions
    return output


def func1():
    # 示例
    input_1d = torch.Tensor(np.random.rand(1, 30, 256))  # Input tensor [channel, features]
    model = KAN_Convolution(2, 3, 2, stride=1, padding=0, dilation=1)
    output_1d = model(input_1d)
    print("1D Convolution Output:", output_1d)


class KAN_Convolution(torch.nn.Module):
    def __init__(self, in_channels, output_channels, kernel, stride=1, padding=0, dilation=1,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 scale_noise: float = 0.1,
                 scale_base: float = 1.0,
                 scale_spline: float = 1.0,
                 base_activation=torch.nn.SiLU,
                 grid_eps: float = 0.02,
                 grid_range: tuple = [-1, 1]
                 ):
        super(KAN_Convolution, self).__init__()

        self.in_channels = in_channels
        self.output_channels = output_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv_core = torch.nn.ModuleList()  # [out_channels, in_channels]
        for o in range(self.output_channels):
            tmp_list = torch.nn.ModuleList()
            for i in range(self.in_channels):
                tmp_list.append(KANLinear(
                    in_features=kernel,
                    out_features=1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range
                ))
            self.conv_core.append(tmp_list)

    def forward(self, x):
        print('start conv1d')
        return KConv1d(x, self.kernel, self.output_channels, self.conv_core,
                       stride=self.stride, padding=self.padding, dilation=self.dilation)


if __name__ == '__main__':
    func1()
