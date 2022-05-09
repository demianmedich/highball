# coding=utf-8
import dataclasses
from abc import (
    ABCMeta,
    abstractmethod
)
from typing import Optional

from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from highball.optim_config import (
    OptimizerConfig,
    LrSchedulerConfig
)


@dataclasses.dataclass
class TrainingConfig:
    num_epochs: int = 1
    batch_size: int = 1
    num_workers: int = 0
    clip_grad_norm: float = 0.
    use_lr_monitor: bool = False


@dataclasses.dataclass
class DatasetConfig:

    @abstractmethod
    def instantiate(self, *args, **kwargs) -> Dataset:
        raise NotImplementedError()


@dataclasses.dataclass
class LightningModuleConfig(metaclass=ABCMeta):
    training_cfg: Optional[TrainingConfig]
    optimizer_cfg: Optional[OptimizerConfig]
    lr_scheduler_cfg: Optional[LrSchedulerConfig]
    train_data_cfg: Optional[DatasetConfig]
    val_data_cfg: Optional[DatasetConfig]
    test_data_cfg: Optional[DatasetConfig]

    @abstractmethod
    def instantiate(self) -> LightningModule:
        raise NotImplementedError()
