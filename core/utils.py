import gc
from types import SimpleNamespace
import torch
import os
import csv


def clear_memory():
    """
    Manually clears GPU and CPU memory to prevent memory overflow.
    Useful between validation loops or when dynamically allocating large tensors.
    """
    torch.cuda.empty_cache() 
    gc.collect()
    
def count_parameters(model):
    """
    Prints the total and trainable number of parameters in a model.

    Args:
        model (torch.nn.Module): Model instance
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    

def save_validation_metrics(epoch, metrics, save_path):
    """
    Saves validation metrics to a CSV file, appending each epoch's results.

    Args:
        epoch (int): Current epoch number
        metrics (dict): Dictionary of metric names and values (e.g., {'MAE': 0.1234})
        save_path (str): File path to save the CSV
    """
    file_exists = os.path.exists(save_path)
    header = ["epoch"] + list(metrics.keys())

    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([epoch] + [metrics[k] for k in header[1:]])

        

def save_best_model(model, mae, best_val_mae, save_path):
    """
    Saves the model if its MAE is better (lower) than the previous best.

    Args:
        model (torch.nn.Module): The model instance
        mae (float): Current validation MAE
        best_val_mae (float): Best MAE seen so far
        save_path (str): File path to save model weights

    Returns:
        float: Updated best MAE value
    """
    if mae < best_val_mae:
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved best model (MAE = {mae:.4f}) to {save_path}")
        return mae 
    else:
        return best_val_mae


class AverageMeter(object):
    """
    Tracks and computes the running average of a metric (e.g., loss or MAE).

    Usage:
        meter = AverageMeter()
        for batch in dataloader:
            meter.update(loss.item(), count=batch_size)
        print(meter.value_avg)
    """
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        """
        Resets all stored values.
        """
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        """
        Updates the average with new value(s).

        Args:
            value (float): New value to include
            count (int): Weight
        """
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count
        
def dict_to_namespace(d):
    """
    Recursively converts a dictionary to a SimpleNamespace,
    allowing attribute-style access.

    Args:
        d (dict): Dictionary to convert

    Returns:
        SimpleNamespace: Namespace representation
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)
