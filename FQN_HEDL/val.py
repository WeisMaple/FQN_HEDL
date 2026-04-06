import os, argparse, warnings, json
import torch
import numpy as np
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")

from dataloader import data_class, data_in_channels, data_fqn_config, dataloader, ood_dataloader
from losses import (edl_HENN, edl_HENN_log, edl_HENN_mse,
                    get_henn_fc, relu_evidence)
from evaluation import calculate_ood, calculate_socre
from models import Conv1DBackbone, FQNBackbone
from data_utils import LOCAL_UCR_PATH
from tqdm import tqdm


def build_model(args):
    in_ch = data_in_channels(args.dataset)
    if args.model == 'FQN':
        fqn_cfg = data_fqn_config(args.dataset)
        model = FQNBackbone(
            in_channels    = in_ch,
            num_classes    = args.num_classes,
            device         = str(args.device),
            dim            = fqn_cfg['dim'],
            depth          = fqn_cfg['depth'],
            input_window   = fqn_cfg['input_window'],
            input_scale    = fqn_cfg['input_scale'],
            hidden_window  = fqn_cfg['hidden_window'],
            loss           = args.loss,
        )
    else:
        model = Conv1DBackbone(
            in_channels = in_ch,
            num_classes = args.num_classes,
            feat_dim    = args.feat_dim,
            loss        = args.loss,
        )
    return model


@torch.no_grad()
def validate(args, valid_dl, ood_dl, model):
    model.eval()
    total_uncertainty, total_label = [], []
    running_corrects, size_k = 0, 0

    print('=== In-Distribution Testing ===')
    for inputs, labels in tqdm(valid_dl):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        if args.loss in ('Softmax', 'dropout'):
            features, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_uncertainty.append(outputs.cpu())

        elif args.loss in ('edl_HENN', 'edl_HENN_log', 'edl_HENN_mse'):
            features, outputs = model(inputs)
            belief = get_henn_fc(features, model.get_weight(),
                                 args.w, args.num_classes, args.device)
            _, preds = torch.max(belief, 1)
            # ← 修复：squeeze 到 (B,)，而非 (B,1)
            u = (args.w / torch.sum(belief, dim=1, keepdim=True)).squeeze(1)
            total_uncertainty.append(u.cpu())

        else:
            features, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            alpha = relu_evidence(outputs) + 1
            # ← 修复：squeeze 到 (B,)
            u = (args.num_classes / torch.sum(alpha, dim=1, keepdim=True)).squeeze(1)
            total_uncertainty.append(u.cpu())

        label_idx = labels.argmax(dim=1)
        running_corrects += (preds == label_idx).sum().item()
        size_k += inputs.size(0)
        total_label.extend([1] * inputs.size(0))

    print('=== OOD Testing ===')
    for inputs, labels in tqdm(ood_dl):
        inputs = inputs.to(args.device)

        if args.loss in ('Softmax', 'dropout'):
            features, outputs = model(inputs)
            total_uncertainty.append(outputs.cpu())

        elif args.loss in ('edl_HENN', 'edl_HENN_log', 'edl_HENN_mse'):
            features, outputs = model(inputs)
            belief = get_henn_fc(features, model.get_weight(),
                                 args.w, args.num_classes, args.device)
            u = (args.w / torch.sum(belief, dim=1, keepdim=True)).squeeze(1)
            total_uncertainty.append(u.cpu())

        else:
            features, outputs = model(inputs)
            alpha = relu_evidence(outputs) + 1
            u = (args.num_classes / torch.sum(alpha, dim=1, keepdim=True)).squeeze(1)
            total_uncertainty.append(u.cpu())

        total_label.extend([0] * inputs.size(0))

    total_uncertainty = torch.cat(total_uncertainty, dim=0)   # (N,) or (N, C)
    total_label       = np.array(total_label)
    acc               = running_corrects / size_k
    return acc, total_uncertainty, total_label


def main(args):
    args.num_classes = data_class(args.dataset)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    valid_dl = dataloader(args.dataset, 'test',
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          local_path=args.local_path)
    ood_dl = ood_dataloader(
        ood_dataset_code = args.ood_dataset,
        id_dataset_code  = args.dataset,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        local_path       = args.local_path,
    )

    model = build_model(args).to(args.device)
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'模型已加载: {args.checkpoint}  backbone={args.model}')

    acc, total_uncertainty, total_label = validate(args, valid_dl, ood_dl, model)
    ood_res = {'acc': acc, 'err': 1.0 - acc}
    print(f'Accuracy: {acc:.4f}')

    if args.loss in ('edl_digamma_loss', 'edl_log_loss',
                     'edl_HENN', 'edl_HENN_log', 'edl_HENN_mse'):
        fpr95, auroc, auprc = calculate_ood(
            calculate_socre(total_uncertainty, 'uncertainty'), total_label
        )
        ood_res.update({'fpr95': fpr95, 'auroc': auroc, 'auprc': auprc})
    else:
        for method in ('msp', 'entropy', 'energy'):
            fpr95, auroc, auprc = calculate_ood(
                calculate_socre(total_uncertainty, method), total_label
            )
            ood_res.update({f'fpr95_{method}': fpr95,
                            f'auroc_{method}': auroc,
                            f'auprc_{method}': auprc})

    print(ood_res)
    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'ood_res.json'), 'w') as f:
        json.dump(ood_res, f, indent=2)
    print(f'结果已保存: {out_dir}/ood_res.json')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     type=str,   default='D1')
    parser.add_argument('--ood-dataset', type=str,   default='D2')
    parser.add_argument('--local-path',  type=str,   default=LOCAL_UCR_PATH)
    parser.add_argument('--checkpoint',  type=str,   required=True)
    parser.add_argument('--model',       type=str,   default='FQN',
                        help='FQN / Conv1D')
    parser.add_argument('--loss',        type=str,   default='edl_HENN')
    parser.add_argument('--feat-dim',    type=int,   default=256,
                        help='仅 Conv1D backbone 使用')
    parser.add_argument('--w',           type=float, default=2.0)
    parser.add_argument('--batch-size',  type=int,   default=64)
    parser.add_argument('--num-workers', type=int,   default=4)
    parser.add_argument('--output-dir',  type=str,   default='results')
    parser.add_argument('--name',        type=str,   default='hedl_fqn')
    parser.add_argument('--device',
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    main(args)