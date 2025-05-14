
'''
* @name: scheduler.py
* @description: Warm up and cosine annealing functions.
'''


import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_scheduler(optimizer, args):
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4, weight_decay=1e-4)
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.9 * args.base.n_epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=0.1 * args.base.n_epochs, after_scheduler=scheduler_steplr)

    return scheduler_warmup