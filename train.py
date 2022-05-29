# coding=utf-8
import argparse
import importlib
from typing import List

from pytorch_lightning import (
    Callback,
    Trainer,
    LightningModule,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint
)

from highball.config import (
    LightningModuleConfig,
    TrainingConfig,
    CHECKPOINTING_CONFIG_TYPE
)


def main(args: argparse.Namespace) -> None:
    print('Start train.py')

    args.cfg = args.cfg.replace('/', '.').replace('.py', '')
    cfg: LightningModuleConfig = importlib.import_module(args.cfg).CFG
    model: LightningModule = cfg.instantiate()

    assert cfg.training_cfg, 'training_cfg should not be None'
    callbacks = add_callbacks(cfg.training_cfg)

    trainer = Trainer(
        max_epochs=cfg.training_cfg.num_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.training_cfg.clip_grad_norm,
        accelerator=cfg.training_cfg.accelerator,
        devices=cfg.training_cfg.devices,
        num_sanity_val_steps=cfg.training_cfg.num_sanity_val_steps
    )
    trainer.fit(model)

    # TODO: Call trainer.test()
    # trainer.test(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Path of configuration python file')
    args = parser.parse_args()
    return args


def add_callbacks(cfg: TrainingConfig) -> List[Callback]:
    callbacks = []
    if cfg.use_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    add_checkpointing_callbacks_(cfg.checkpointing_cfg, callbacks)
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


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
