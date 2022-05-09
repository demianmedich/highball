# coding=utf-8
from typing import (
    Literal,
    Union,
    Callable
)

import torch.nn as nn
from torch import Tensor

ACTIVATION_LAYERS = Literal[
    'identity',
    'step',
    'relu',
    'gelu'
]


class Step:
    """Step activation function"""

    def __call__(self, x: Tensor):
        return (x > 0).to(dtype=x.dtype, device=x.device)


def activation_layer(name: ACTIVATION_LAYERS, **kwargs) -> Union[nn.Module, Callable]:
    if name == 'relu':
        inplace = kwargs.get('inplace', False)
        return nn.ReLU(inplace=inplace)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'step':
        return Step()
    else:
        return nn.Identity()
