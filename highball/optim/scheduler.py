# coding=utf-8
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
        optimizer: Optimizer,
        warmup_steps: int,
        total_training_steps: int,
        last_epoch: int = -1,
) -> LambdaLR:
    def _schedule(current_step: int):
        if warmup_steps > current_step:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0., float(total_training_steps - current_step) / float(
                max(1, total_training_steps - warmup_steps)))

    return LambdaLR(optimizer, _schedule, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        warmup_steps: int,
        total_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
