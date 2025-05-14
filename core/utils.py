import gc

def clear_memory():
    torch.cuda.empty_cache()  # 释放 GPU 缓存
    gc.collect()  # 释放 CPU 内存
    
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    

def save_validation_metrics(epoch, metrics, save_path="metrics_log_chsims.csv"):
    """
    保存每轮验证的指标到 CSV 文件。
    
    Args:
        epoch (int): 当前 epoch
        metrics (dict): 包含各类验证指标的字典
        save_path (str): 保存路径
    """
    file_exists = os.path.exists(save_path)
    header = ["epoch"] + list(metrics.keys())

    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([epoch] + [metrics[k] for k in header[1:]])

        

def save_best_model(model, mae, best_val_mae, save_path='best_model.pt'):
    """Save the model if current mae is better than best_val_mae."""
    if mae < best_val_mae:
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved best model (MAE = {mae:.4f}) to {save_path}")
        return mae  # 返回更新后的 best_val_mae
    else:
        return best_val_mae  # 不更新


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