import torch
import os, sys, argparse, time, datetime, warnings
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import data_class, data_in_channels, dataloader   # ← 新增 data_in_channels
from losses import (edl_mse_loss, edl_digamma_loss, edl_log_loss,
                    edl_HENN, edl_HENN_log, edl_HENN_mse,
                    get_henn_fc, relu_evidence)
from models import Conv1DBackbone
from data_utils import LOCAL_UCR_PATH                              # ← 新增

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
def build_model(args):
    """从 DATASET_CONFIG 读取 in_channels，不再写死。"""
    in_ch    = data_in_channels(args.dataset)          # ← 改动点
    feat_dim = args.feat_dim
    model = Conv1DBackbone(
        in_channels = in_ch,
        num_classes = args.num_classes,
        feat_dim    = feat_dim,
        loss        = args.loss,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(args, writer, model, criterion, data_loader, optimizer, epoch):
    model.train()
    last_loss = None
    for count, (inputs, targets) in enumerate(tqdm(data_loader, desc=f'Train E{epoch}')):
        inputs  = inputs.to(args.device)
        targets = targets.float().to(args.device)

        if args.loss in ('Softmax', 'dropout'):
            features, outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif args.loss in ('edl_HENN', 'edl_HENN_log', 'edl_HENN_mse'):
            features, outputs = model(inputs)
            loss = criterion(args.w, features, model.get_weight(),
                             targets, epoch, args.num_classes, 10,
                             args.device, outputs)
        else:
            features, outputs = model(inputs)
            loss = criterion(outputs, targets, epoch,
                             args.num_classes, 10, args.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer is not None and count % args.log_steps == 0:
            writer.add_scalar('Loss/train', loss.item(),
                              epoch * len(data_loader) + count)
        last_loss = loss

        if args.debug and count >= 1:
            break

    return last_loss


@torch.no_grad()
def evaluate(args, writer, model, criterion, data_loader, epoch):
    model.eval()
    running_loss, running_corrects, size = 0.0, 0, 0

    for count, (inputs, targets) in enumerate(tqdm(data_loader, desc=f'Valid E{epoch}')):
        inputs  = inputs.to(args.device)
        targets = targets.float().to(args.device)

        if args.loss in ('Softmax', 'dropout'):
            features, outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
        elif args.loss in ('edl_HENN', 'edl_HENN_log', 'edl_HENN_mse'):
            features, outputs = model(inputs)
            loss = criterion(args.w, features, model.get_weight(),
                             targets, epoch, args.num_classes, 10, args.device)
            belief = get_henn_fc(features, model.get_weight(),
                                 args.w, args.num_classes, args.device)
            _, preds = torch.max(belief, 1)
        else:
            features, outputs = model(inputs)
            loss = criterion(outputs, targets, epoch,
                             args.num_classes, 10, args.device)
            _, preds = torch.max(outputs, 1)

        running_loss     += loss.item() * inputs.size(0)
        label_idx         = targets.argmax(dim=1)
        running_corrects += (preds == label_idx).sum().item()
        size             += inputs.size(0)

        if args.debug and count >= 1:
            break

    epoch_loss = running_loss / size
    epoch_acc  = running_corrects / size

    if writer is not None:
        writer.add_scalar('Loss/valid',     epoch_loss, epoch)
        writer.add_scalar('Accuracy/valid', epoch_acc,  epoch)

    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='HEDL 1D Time-Series Training')
    parser.add_argument('--dataset',      type=str,   default='D1',
                        help='数据集代码，如 D1~D20（见 data_utils.DATASET_CONFIG）')
    parser.add_argument('--local-path',   type=str,   default=LOCAL_UCR_PATH,
                        help='UCR/UEA 数据集本地路径')
    parser.add_argument('--batch-size',   type=int,   default=64)
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--feat-dim',     type=int,   default=256,
                        help='Conv1D backbone 输出特征维度')
    parser.add_argument('--loss',         type=str,   default='edl_HENN',
                        help='Softmax/dropout/edl_digamma_loss/'
                             'edl_log_loss/edl_mse_loss/edl_HENN/'
                             'edl_HENN_log/edl_HENN_mse')
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--w',            type=float, default=2.0)
    parser.add_argument('--log-steps',    type=int,   default=50)
    parser.add_argument('--name',         type=str,   default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--debug',        action='store_true')
    parser.add_argument('--device',
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ── 数据 ────────────────────────────────────────────────────────────────
    args.num_classes = data_class(args.dataset)        # 从 DATASET_CONFIG 读取

    train_dl = dataloader(args.dataset, 'train',
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          local_path=args.local_path)
    valid_dl = dataloader(args.dataset, 'valid',
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          local_path=args.local_path)

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = build_model(args).to(args.device)
    print('参数量:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ── 损失 ────────────────────────────────────────────────────────────────
    loss_map = {
        'Softmax'         : nn.CrossEntropyLoss(),
        'dropout'         : nn.CrossEntropyLoss(),
        'edl_digamma_loss': edl_digamma_loss,
        'edl_log_loss'    : edl_log_loss,
        'edl_mse_loss'    : edl_mse_loss,
        'edl_HENN'        : edl_HENN,
        'edl_HENN_log'    : edl_HENN_log,
        'edl_HENN_mse'    : edl_HENN_mse,
    }
    if args.loss not in loss_map:
        parser.error(f'未知 loss: {args.loss}')
    criterion = loss_map[args.loss]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        print(f'已从 {args.checkpoint_path} 恢复权重')

    if args.name is None:
        args.name = args.loss

    # ── 日志与 checkpoint ────────────────────────────────────────────────────
    log_dir       = f'runs_{args.dataset}/{args.name}'
    writer        = SummaryWriter(log_dir=log_dir)
    ckpt_dir_acc  = Path(log_dir) / 'acc'  / 'checkpoints'
    ckpt_dir_loss = Path(log_dir) / 'loss' / 'checkpoints'
    ckpt_dir_acc.mkdir(parents=True, exist_ok=True)
    ckpt_dir_loss.mkdir(parents=True, exist_ok=True)

    best_acc, best_loss = 0.0, float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f'\n── Epoch {epoch} ──────────────────────────')

        train_loss = train_one_epoch(args, writer, model, criterion,
                                     train_dl, optimizer, epoch)
        valid_loss, valid_acc = evaluate(args, writer, model, criterion,
                                         valid_dl, epoch)

        scheduler.step()

        print(f'  train_loss={train_loss.item():.4f}  '
              f'valid_loss={valid_loss:.4f}  valid_acc={valid_acc:.4f}  '
              f'best_acc={best_acc:.4f}')

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),  # ← 新增
            'epoch': epoch,
        }

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(state, ckpt_dir_acc / 'checkpoint.pth')
            print('  ✓ 保存 best-acc checkpoint')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(state, ckpt_dir_loss / 'checkpoint.pth')
            print('  ✓ 保存 best-loss checkpoint')

        print(f'  epoch time: {datetime.timedelta(seconds=int(time.time()-t0))}')

    writer.close()