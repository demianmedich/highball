# coding=utf-8
import argparse
import importlib
from typing import List

from pytorch_lightning import (
    Callback,
    Trainer,
    LightningModule,
)
from pytorch_lightning.callbacks import LearningRateMonitor

from highball.config import (
    LightningModuleConfig,
    TrainingConfig
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
        devices=cfg.training_cfg.num_gpus
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
    return callbacks


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
