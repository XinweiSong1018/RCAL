import gc
from types import SimpleNamespace
import torch
import os
import csv


def clear_memory():
    torch.cuda.empty_cache() 
    gc.collect()
    
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    

def save_validation_metrics(epoch, metrics, save_path):
    file_exists = os.path.exists(save_path)
    header = ["epoch"] + list(metrics.keys())

    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([epoch] + [metrics[k] for k in header[1:]])

        

def save_best_model(model, mae, best_val_mae, save_path):
    if mae < best_val_mae:
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved best model (MAE = {mae:.4f}) to {save_path}")
        return mae 
    else:
        return best_val_mae


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count
        
def dict_to_namespace(d):
    """Recursively converts a dictionary and its nested dictionaries to a Namespace."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)
