# coding=utf-8
import dataclasses
from abc import (
    ABCMeta,
    abstractmethod
)
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from highball.optim_config import (
    OptimizerConfig,
    LrSchedulerConfig
)


@dataclasses.dataclass
class TrainingConfig:
    accelerator: Optional[str] = 'gpu' if torch.cuda.is_available() else None
    devices: Optional[int] = 1 if torch.cuda.is_available() else None
    num_epochs: int = 1
    batch_size: int = 1
    num_workers: int = 6
    clip_grad_norm: float = 0.
    use_lr_monitor: bool = False


@dataclasses.dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool

    @abstractmethod
    def instantiate(self, *args, **kwargs) -> DataLoader:
        raise NotImplementedError()


@dataclasses.dataclass
class LightningModuleConfig(metaclass=ABCMeta):
    training_cfg: Optional[TrainingConfig]
    optimizer_cfg: Optional[OptimizerConfig]
    lr_scheduler_cfg: Optional[LrSchedulerConfig]
    train_dataloader_cfg: Optional[DataLoaderConfig]
    val_dataloader_cfg: Optional[DataLoaderConfig]
    test_dataloader_cfg: Optional[DataLoaderConfig]

    @abstractmethod
    def instantiate(self) -> LightningModule:
        raise NotImplementedError()
