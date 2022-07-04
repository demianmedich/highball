# coding=utf-8
import time
from typing import Literal

from torch import (
    nn,
    Tensor
)

INIT_METHOD = Literal[
    'zeros',
    'ones',
    'uniform',
    'normal',
    'trunc_normal',
    'xavier_uniform',
    'xavier_normal',
    'kaiming_uniform',
    'kaiming_normal',
    'he_uniform',
    'he_normal'
]


def init_tensor_(method: INIT_METHOD, tensor: Tensor, **kwargs) -> None:
    if method == 'zeros':
        nn.init.zeros_(tensor)
    elif method == 'ones':
        nn.init.ones_(tensor)
    elif method == 'uniform':
        a = kwargs.get('a', 0.0)
        b = kwargs.get('b', 0.0)
        nn.init.uniform_(tensor, a, b)
    elif method == 'normal':
        mean = kwargs.get('mean', 1.0)
        std = kwargs.get('std', 0.)
        nn.init.normal_(tensor, mean, std)
    elif method == 'trunc_normal':
        mean = kwargs.get('mean', 1.0)
        std = kwargs.get('std', 0.)
        a = kwargs.get('a', 0.0)
        b = kwargs.get('b', 0.0)
        nn.init.trunc_normal_(tensor, mean, std, a, b)
    elif method == 'xavier_uniform':
        gain = kwargs.get('gain', 1.0)
        nn.init.xavier_uniform_(tensor, gain)
    elif method == 'xavier_normal':
        gain = kwargs.get('gain', 1.0)
        nn.init.xavier_normal_(tensor, gain)
    elif method in ('he_uniform', 'kaiming_uniform'):
        a = kwargs.get('a', 0.0)
        mode = kwargs.get('mode', 'fan_in')
        nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
        nn.init.kaiming_uniform_(tensor, a, mode, nonlinearity)
    elif method in ('he_normal', 'kaiming_normal'):
        a = kwargs.get('a', 0.0)
        mode = kwargs.get('mode', 'fan_in')
        nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
        nn.init.kaiming_normal_(tensor, a, mode, nonlinearity)
    else:
        pass


class PerfCounter:

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self.start_time
