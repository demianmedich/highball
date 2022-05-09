# coding=utf-8
import dataclasses
from abc import (
    ABCMeta,
    abstractmethod
)
from typing import (
    Tuple,
    Iterator
)

from torch import Tensor
from torch.nn import Parameter
from torch.optim import (
    SGD,
    Adam
)
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


@dataclasses.dataclass
class OptimizerConfig(metaclass=ABCMeta):
    lr: float

    @abstractmethod
    def instantiate(self, params: Iterator[Parameter]) -> Optimizer:
        raise NotImplementedError()


@dataclasses.dataclass
class LrSchedulerConfig(metaclass=ABCMeta):

    @abstractmethod
    def instantiate(self, optimizer: Optimizer) -> _LRScheduler:
        raise NotImplementedError()


@dataclasses.dataclass
class SgdOptimizerConfig(OptimizerConfig):
    momentum: float = 0.
    dampening: float = 0.
    weight_decay: float = 0.
    nesterov: bool = False

    def instantiate(self, params: Tensor) -> Optimizer:
        return SGD(
            params,
            self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov
        )


@dataclasses.dataclass
class AdamOptimizerConfig(OptimizerConfig):
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.
    amsgrad: bool = False

    def instantiate(self, params: Tensor) -> Optimizer:
        return Adam(
            params,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )


@dataclasses.dataclass
class CosineWarmupSchedulerConfig(OptimizerConfig):

    def instantiate(self, params: Tensor) -> Optimizer:
        pass
