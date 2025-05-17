
'''
* @name: scheduler.py
* @description: Warm up and cosine annealing functions for learning rate scheduling.
'''


import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually increases the learning rate from a small initial value to the target base learning rate
    over a specified number of warm-up epochs. After warm-up, optionally delegates to another scheduler.

    Based on:
    "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" — Facebook AI Research.

    Args:
        optimizer (Optimizer): Optimizer to apply the schedule to.
        multiplier (float): 
            - If > 1.0, final LR = base_lr * multiplier.
            - If == 1.0, LR starts at 0 and linearly increases to base_lr.
        total_epoch (int): Number of warm-up epochs.
        after_scheduler (_LRScheduler, optional): 
            Scheduler to apply after warm-up is finished (e.g., CosineAnnealingLR).
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """
        Compute the learning rate at each step.
        """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """
        Special handling for ReduceLROnPlateau scheduler, which is called at the end of epoch.

        Args:
            metrics (float): Validation metric to monitor
            epoch (int, optional): Current epoch number
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        """
        Update learning rate — delegates to appropriate scheduler based on context.

        Args:
            epoch (int, optional): Current epoch
            metrics (float, optional): Metric for ReduceLROnPlateau (if used)
        """
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            
            
def get_scheduler(model, optimizer, args):
    """
    Utility function to construct a learning rate scheduler with warm-up and cosine annealing.

    Args:
        model (nn.Module): The model being trained (unused here, but may be extended)
        optimizer (Optimizer): Optimizer instance
        args: Argument namespace that contains training configuration, including `args.base.n_epochs`

    Returns:
        GradualWarmupScheduler: Learning rate scheduler with warm-up followed by cosine decay
    """
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.9 * args.base.n_epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=0.1 * args.base.n_epochs, after_scheduler=scheduler_steplr)

    return scheduler_warmup
