"""
统一数据加载入口。
时序数据集通过 data_utils.load_and_split_dataset 一次性完成
加载 / 标准化 / 分割，结果缓存在模块级字典中避免重复 IO。
"""

import sys
import torch
from torch.utils.data import DataLoader

from timeseries import TimeSeriesDataset
from data_utils import (
    load_and_split_dataset,
    prepare_ood_data,
    get_dataset_config,
    LOCAL_UCR_PATH,
    DATASET_CONFIG,
)


# ── 模块级缓存，同一 dataset_code 只加载一次 ──────────────────────────────────
_DATASET_CACHE: dict = {}


def _get_cached_data(dataset_code: str, local_path: str = LOCAL_UCR_PATH) -> dict:
    """加载并缓存已切分好的数据字典。"""
    if dataset_code not in _DATASET_CACHE:
        print(f'[dataloader] 首次加载 {dataset_code}，正在读取并切分...')
        _DATASET_CACHE[dataset_code] = load_and_split_dataset(
            dataset_code, local_path=local_path
        )
    return _DATASET_CACHE[dataset_code]


# ── 查询类别数 ────────────────────────────────────────────────────────────────
def data_class(dataset_code: str) -> int:
    """根据 dataset_code 返回类别数（读自 DATASET_CONFIG）。"""
    if dataset_code not in DATASET_CONFIG:
        print(f'[dataloader] 未知数据集代码: {dataset_code}')
        print(f'  可用代码: {list(DATASET_CONFIG.keys())}')
        sys.exit(1)
    _, _, num_classes, *_ = get_dataset_config(dataset_code)
    return num_classes


def data_in_channels(dataset_code: str) -> int:
    """根据 dataset_code 返回输入通道数。"""
    if dataset_code not in DATASET_CONFIG:
        print(f'[dataloader] 未知数据集代码: {dataset_code}')
        sys.exit(1)
    _, in_channels, *_ = get_dataset_config(dataset_code)
    return in_channels


# ── 主加载函数 ────────────────────────────────────────────────────────────────
def dataloader(
    dataset_code: str,
    data_mode: str = 'train',       # 'train' | 'valid' | 'test'
    batch_size: int = 32,
    num_workers: int = 4,
    local_path: str = LOCAL_UCR_PATH,
    **kwargs,                        # 兼容旧调用（如 image_size）
) -> DataLoader:
    """
    返回对应 split 的 DataLoader。
    dataset_code : 'D1' ~ 'D20'（见 data_utils.DATASET_CONFIG）
    data_mode    : 'train' / 'valid' / 'test'
    """
    data = _get_cached_data(dataset_code, local_path=local_path)

    split_map = {
        'train': ('X_train', 'y_train'),
        'valid': ('X_val',   'y_val'),
        'test' : ('X_test',  'y_test'),
    }
    if data_mode not in split_map:
        print(f'[dataloader] 未知 data_mode: {data_mode}，应为 train/valid/test')
        sys.exit(1)

    xk, yk = split_map[data_mode]
    ds = TimeSeriesDataset(
        X           = data[xk],
        y           = data[yk],
        num_classes = data['num_classes'],
    )

    shuffle = (data_mode == 'train')
    dl = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
    )
    print(f'[dataloader] dataset={data["name"]}({dataset_code})  '
          f'mode={data_mode}  samples={len(ds)}  '
          f'channels={data["in_channels"]}  classes={data["num_classes"]}')
    return dl


def ood_dataloader(
    ood_dataset_code: str,
    id_dataset_code: str,
    batch_size: int = 32,
    num_workers: int = 4,
    local_path: str = LOCAL_UCR_PATH,
) -> DataLoader:
    id_data    = _get_cached_data(id_dataset_code, local_path=local_path)
    ood_name, ood_in_channels, ood_num_classes, *_ = get_dataset_config(ood_dataset_code)

    target_len   = id_data['X_train'].shape[2]
    global_means = id_data['global_means']
    global_stds  = id_data['global_stds']

    X_ood, y_ood = prepare_ood_data(
        ood_name     = ood_name,
        global_means = global_means,
        global_stds  = global_stds,
        target_len   = target_len,
        local_path   = local_path,
    )

    # ← 关键修复：使用 OOD 数据集自身的 num_classes，而非 ID 的
    ds = TimeSeriesDataset(
        X           = X_ood,
        y           = y_ood,
        num_classes = ood_num_classes,
    )
    dl = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    print(f'[dataloader] OOD dataset={ood_name}({ood_dataset_code})  '
          f'samples={len(ds)}  ood_classes={ood_num_classes}')
    return dl