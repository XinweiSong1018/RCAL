import os
import torch
import argparse
from core.dataset import MMDataset
from torch.utils.data import DataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, dict_to_namespace, clear_memory, save_validation_metrics, save_best_model
from models.model import Model
from core.metric import MetricsTop
from core.loss import SupervisedContrastiveLoss, emo_loss_fn
import yaml
from tqdm import tqdm


parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='configs/mosi.yaml') 
parser.add_argument('--gpu_id', type=int, default=-1) 
opt = parser.parse_args()
print(opt)

with open(opt.config_file) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
print(args)


print('-----------------args-----------------')
print(args)
print('-------------------------------------')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

best_val_mae = float("inf")

def main():

    model = Model(args).to(device)

    train_dataset = MMDataset(args,mode='train')
    valid_dataset = MMDataset(args,mode='valid')
    train_loader = DataLoader(train_dataset, batch_size=args.base.batch_size, shuffle=True, num_workers=args.base.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.base.batch_size, shuffle=False, num_workers=args.base.num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.base.lr,
                                 weight_decay=args.base.weight_decay)

    scheduler_warmup = get_scheduler(model, optimizer, args)

    for epoch in range(1, args.base.n_epochs + 1):
        clear_memory()
        train(model, train_loader, optimizer, epoch, args)
        evaluate(model, val_loader, optimizer, epoch, args)  
        scheduler_warmup.step()

def train(model, train_loader, optimizer, epoch, args):
    train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
    losses = AverageMeter()
    l1s = AverageMeter()
    l2s = AverageMeter()
    rls = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    moving_baseline = args.train.moving_baseline
    alpha = args.train.alpha
    
    lambda_recon = args.train.lambda_recon    
    lambda_rl = args.train.lambda_rl

    for cur_iter, data in train_pbar:
        img = data['images'].to(device)
        audio = data['audio'].to(device)
        text = data['text'].to(device)
        label = data['labels'].to(device).view(-1, 1)
        batchsize = img.shape[0]

        optimizer.zero_grad()

        output, h_v_generated, log_probs, gates, probs = model(img, audio, text)

        l1 = emo_loss_fn(output, label)
        l2 = SupervisedContrastiveLoss(h_v_generated, label)

        # === REINFORCE ===
        mae_clamped = l1.detach().clamp(0, 2)
        reward = 1.0 - mae_clamped / 2.0
        advantage = reward - moving_baseline
        loss_rl = sum([-(lp.squeeze() * advantage).mean() for lp in log_probs])
        moving_baseline = alpha * moving_baseline + (1 - alpha) * reward.item()

       

        # === 总损失 ===
        loss = l1 + lambda_recon * l2 + lambda_rl * loss_rl
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batchsize)
        l1s.update(l1.item(), batchsize)
        l2s.update(l2.item(), batchsize)
        rls.update(loss_rl.item(), batchsize)

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description(f'train [Epoch {epoch}]')
        train_pbar.set_postfix({
            'loss': '{:.5f}'.format(losses.value_avg),
            'emo_loss': '{:.4f}'.format(l1s.value_avg),
            'visual_loss': '{:.4f}'.format(l2s.value_avg),
            'RL_loss': '{:.4f}'.format(rls.value_avg),
            'λ_recon': '{:.2f}'.format(lambda_recon),
            'λ_RL': '{:.2f}'.format(lambda_rl),
            'lr': '{:.2e}'.format(optimizer.param_groups[0]['lr'])
        }, refresh=False)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    mae = torch.mean(torch.abs(pred - true)).item()
    tqdm.write(f"train MAE: {mae:.4f}")




# Evaluate 函数
def evaluate(model, eval_loader, optimizer, epoch, args):
    global best_val_mae
    metric = MetricsTop()
    test_pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    losses = AverageMeter()
    l1s = AverageMeter()
    l2s = AverageMeter()

    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img = data['images'].to(device)
            audio = data['audio'].to(device)
            text = data['text'].to(device)
            label = data['labels'].to(device).view(-1, 1)
            batchsize = img.shape[0]

            output, h_v_generated, log_probs, gates, probs = model(img, audio, text)           

            l1 = emo_loss_fn(output, label)
            l2 = SupervisedContrastiveLoss(h_v_generated, label)

            
            lambda_recon = args.train.lambda_recon
            
            loss = l1 + lambda_recon * l2
            
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)
            l1s.update(l1.item(), batchsize)
            l2s.update(l2.item(), batchsize)


            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                'loss': '{:.5f}'.format(losses.value_avg),
                'emo_loss': '{:.4f}'.format(l1s.value_avg),
                'visual_loss': '{:.4f}'.format(l2s.value_avg),
                'lambda_recon': '{:.2f}'.format(lambda_recon),
                'lr': '{:.2e}'.format(optimizer.param_groups[0]['lr'])
            }, refresh=False)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_func = metric.getMetics(args.dataset.datasetName)
        eval_results = eval_func(pred, true)
        
        print(f"Evaluation Results:")
        for key,value in eval_results.items():
            print(f"{key}: {value:.4f}")
        
        gate_stats = torch.cat(gates, dim=1)  # [B, num_layers]
        gate_ratio = gate_stats.float().mean(dim=0).cpu().numpy()

        print('Gate Activation Ratio:', gate_ratio)
        
        metric_file = args.dataset.datasetName+'_metric.csv'
        model_file = args.dataset.datasetName+'_best_model.pt'
        
        save_validation_metrics(epoch,eval_results, metric_file)
        best_val_mae = save_best_model(model, eval_results['MAE'], best_val_mae, model_file)

if __name__ == '__main__':
    main()