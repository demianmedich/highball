# coding=utf-8
import dataclasses
from abc import (
    ABCMeta,
    abstractmethod
)
from datetime import timedelta
from typing import (
    Optional,
    Union,
    List
)

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from highball.optim_config import (
    OptimizerConfig,
    LrSchedulerConfig
)

CHECKPOINTING_CONFIG_TYPE = Union["CheckpointingConfig", List["CheckpointingConfig"]]


@dataclasses.dataclass
class TrainingConfig:
    accelerator: Optional[str] = 'gpu' if torch.cuda.is_available() else None
    devices: Optional[int] = 1 if torch.cuda.is_available() else None
    num_epochs: int = 1
    clip_grad_norm: float = 0.
    use_lr_monitor: bool = False
    num_sanity_val_steps: int = 0
    checkpointing_cfg: Optional[CHECKPOINTING_CONFIG_TYPE] = None
    early_stopping_cfg: Optional["EarlyStoppingConfig"] = None


@dataclasses.dataclass
class CheckpointingConfig:
    dirpath: Optional[str] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    mode: str = 'min'
    save_last: Optional[bool] = None
    save_top_k: int = 1
    save_weights_only: bool = False
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[timedelta] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None


@dataclasses.dataclass
class EarlyStoppingConfig:
    monitor: str
    mode: str = 'min'
    patience: int = 3


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
