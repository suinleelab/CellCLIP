"""CLIP scheduler related functions"""
import math

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def assign_learning_rate(optimizer, new_lr):
    """Assign learning rate"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    """Warm-up learning rate"""
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, step):
    """Constant learning rate scheduler"""

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(
    optimizer,
    base_lr,
    warmup_length,
    steps,
    cooldown_steps,
    cooldown_power=1.0,
    cooldown_end_lr=0.0,
):
    """Constant learning rate scheduler"""

    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """Cosine learning rate scheduler"""

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
    ----
        optimizer : torch.optim.Optimizer
            The optimizer for which to schedule the learning rate.
        warmup:  int
            The number of steps for the warmup phase.
        num_training_steps : int
            The total number of training steps.
        num_cycles : float
            The number of waves in the cosine schedule
            (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch : int
            The index of the last epoch when resuming training.

    Return:
    ------
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, num_training_steps - warmup))
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    warmup: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following
    the values of the cosine function between the initial lr set in the optimizer to 0,
    with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
    ----
        optimizer : torch.optim.Optimizer
            The optimizer for which to schedule the learning rate.
        warmup : int
            The number of steps for the warmup phase.
        num_training_steps : int
            The total number of training steps.
        num_cycles : int
            The number of hard restarts to use.
        last_epoch : int
            The index of the last epoch when resuming training.

    Return:
    ------
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, num_training_steps - warmup))
        if progress >= 1.0:
            return 0.0
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
