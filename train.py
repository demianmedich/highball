# coding=utf-8
import argparse
import importlib
from copy import deepcopy
from typing import List

import pytorch_lightning
from pytorch_lightning import (
    Callback,
    Trainer,
    LightningModule,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping
)

from highball.config import (
    LightningModuleConfig,
    TrainingConfig,
    CHECKPOINTING_CONFIG_TYPE,
    EarlyStoppingConfig
)


def main(args: argparse.Namespace) -> None:
    print(f'Start train.py\n'
          f'args: {args}')

    args.cfg = args.cfg.replace('/', '.').replace('.py', '')
    cfg: LightningModuleConfig = importlib.import_module(args.cfg).CFG
    model: LightningModule = cfg.instantiate()

    pytorch_lightning.seed_everything(args.seed)

    trainer = make_trainer(cfg.training_cfg)
    trainer.fit(model)

    # Do test
    if cfg.test_dataloader_cfg is not None:
        trainer.test(model)
    else:
        print('test_dataloader_cfg is None. skipped test step')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Path of configuration python file')
    parser.add_argument('--seed', type=int, default=1988,
                        help='seed to generate random values. '
                             'if not specified, default value will be used. (1988)')
    args = parser.parse_args()
    return args


def make_trainer(cfg: TrainingConfig) -> Trainer:
    assert cfg, 'training_cfg should not be None'

    cfg = deepcopy(cfg)
    callbacks = add_callbacks(cfg)

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.clip_grad_norm,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        deterministic=cfg.deterministic,
        reload_dataloaders_every_n_epochs=cfg.reload_dataloader_every_n_epochs
    )
    return trainer


def add_callbacks(cfg: TrainingConfig) -> List[Callback]:
    callbacks = []
    if cfg.use_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    add_checkpointing_callbacks_(cfg.checkpointing_cfg, callbacks)
    add_early_stopping_callbacks_(cfg.early_stopping_cfg, callbacks)
    return callbacks


def add_checkpointing_callbacks_(cfg: CHECKPOINTING_CONFIG_TYPE, callbacks: List[Callback]):
    if type(cfg) == list:
        for _cfg in cfg:
            callbacks.append(
                ModelCheckpoint(dirpath=_cfg.dirpath,
                                filename=_cfg.filename,
                                monitor=_cfg.monitor,
                                mode=_cfg.mode,
                                save_last=_cfg.save_last,
                                save_top_k=_cfg.save_top_k,
                                save_weights_only=_cfg.save_weights_only,
                                every_n_train_steps=_cfg.every_n_train_steps,
                                train_time_interval=_cfg.train_time_interval,
                                every_n_epochs=_cfg.every_n_epochs,
                                save_on_train_epoch_end=_cfg.save_on_train_epoch_end)
            )
    else:
        callbacks.append(
            ModelCheckpoint(dirpath=cfg.dirpath,
                            filename=cfg.filename,
                            monitor=cfg.monitor,
                            mode=cfg.mode,
                            save_last=cfg.save_last,
                            save_top_k=cfg.save_top_k,
                            save_weights_only=cfg.save_weights_only,
                            every_n_train_steps=cfg.every_n_train_steps,
                            train_time_interval=cfg.train_time_interval,
                            every_n_epochs=cfg.every_n_epochs,
                            save_on_train_epoch_end=cfg.save_on_train_epoch_end)
        )


def add_early_stopping_callbacks_(cfg: EarlyStoppingConfig, callbacks: List[Callback]):
    callbacks.append(
        EarlyStopping(monitor=cfg.monitor,
                      mode=cfg.mode,
                      patience=cfg.patience)
    )


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
