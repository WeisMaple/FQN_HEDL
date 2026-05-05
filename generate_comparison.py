"""
生成统一四模型对比表：HEDL vs REEDL vs RSNN vs FQN+HEDL

数据来源：
  - hedh/OOD_results.txt: HEDL per-dataset averages (19 datasets, no D3)
  - reedl/.../benchmark_results.json: REEDL per-dataset + per-OOD detail (20 datasets)
  - rsnn/.../benchmark_results_resnet18.json: RSNN per-dataset + per-OOD detail (20 datasets)
  - FQN_HEDL/.../benchmark_results_fqn_hedl.json: FQN+HEDL per-dataset + per-OOD detail

JSON 格式统一：
  - avg_fpr95: percentage (0-100)
  - avg_auroc: raw (0-1)  → 表格中 ×100 显示为 %
  - avg_auprc: raw (0-1)  → 表格中 ×100 显示为 %

使用：
  - 正常： python generate_comparison.py
  - 仅含已有数据（FQN+HEDL 未跑完时）：自动处理缺失

输出：comparison_all_four.md
"""
import json
import re
from pathlib import Path
import numpy as np

_BASE = Path(__file__).resolve().parent
_PROJECT = _BASE.parent  # f:/QML/Papers/CODES

# Data source paths
HEDL_TXT = _PROJECT / "hedl" / "OOD_results.txt"
REEDL_JSON = _PROJECT / "reedl" / "Re-EDL-main" / "code_classical" / "saved_models" / "ood_benchmark" / "benchmark_results.json"
RSNN_JSON = _PROJECT / "rsnn" / "saved_models" / "benchmark_results_resnet18.json"
FQN_JSON = _BASE / "saved_models" / "benchmark_results_fqn_hedl.json"

# Channel counts from DATASET_CONFIG (matches FQN_HEDL/data_utils.py)
CHANNEL_MAP = {
    'D1': 1, 'D2': 1, 'D3': 1, 'D4': 1, 'D5': 1, 'D6': 1, 'D7': 1,
    'D8': 1, 'D9': 1, 'D10': 1, 'D11': 1, 'D12': 1, 'D13': 1,
    'D14': 3, 'D15': 3, 'D16': 4, 'D17': 7, 'D18': 10, 'D19': 28, 'D20': 64,
}

_ALL_IDS = [f"D{i}" for i in range(1, 21)]


def parse_hedl_txt(path):
    """Parse HEDL OOD_results.txt → {code: {fpr95%, auroc%, auprc%}}"""
    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r'(D\d+)\s+\S+\s+\d+\s+\d+\s+'
                r'([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%',
                line
            )
            if m:
                results[m.group(1)] = {
                    "fpr95": float(m.group(2)),
                    "auroc": float(m.group(3)),
                    "auprc": float(m.group(4)),
                }
    return results


def load_json_avg(path):
    """Load JSON benchmark results → {code: {fpr95%, auroc_raw, auprc_raw}}"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = {}
    for code, entry in data.items():
        results[code] = {
            "fpr95": entry["avg_fpr95"],           # already %
            "auroc": entry["avg_auroc"],            # raw
            "auprc": entry["avg_auprc"],            # raw
            "pairs": len(entry.get("per_ood", {})),
        }
    return results


def build_main_table(hedl, reedl, rsnn, fqn):
    """Build the main 4-model comparison markdown table."""
    lines = []
    lines.append("# HEDL vs REEDL vs RSNN vs FQN+HEDL — 统一协议 OOD 检测对比")
    lines.append("")
    lines.append("## 实验协议")
    lines.append("")
    lines.append("| 项目 | HEDL | REEDL | RSNN | FQN+HEDL |")
    lines.append("|------|------|-------|------|----------|")
    lines.append("| Backbone | ResNet18 | ResNet18 | ResNet18 | FQN Quanv1d |")
    lines.append("| 输入格式 | 伪图像 (3×H×W) | 伪图像 (3×H×W) | 伪图像 (3×H×W) | 原始时间序列 (C×T) |")
    lines.append("| OOD Score | Vacuity | Vacuity | Vacuity (mass) | Vacuity (K/Σα) |")
    lines.append("| Split | 60/20/20 (seed=42) | 60/20/20 (seed=42) | 60/20/20 (seed=42) | 60/20/20 (seed=42) |")
    lines.append("| 跨通道 OOD | 支持 (统一伪图像尺寸) | 支持 | 支持 | 不支持 (物理限制，仅同通道) |")
    lines.append("")
    lines.append("## 全量对比 (Full Comparison)")
    lines.append("")
    lines.append("> **注意：** FQN+HEDL 仅在同通道数的 ID/OOD 对之间评估（原始时间序列的物理维度限制），")
    lines.append("> 每个 ID 的 OOD 对数量 = 同通道数据集数 - 1（标记在 `Pairs` 列）。")
    lines.append("> HEDL/REEDL/RSNN 通过伪图像转换消除了通道差异，每个 ID 均与 **19** 个 OOD 数据集比较。")
    lines.append("> 因此，FQN+HEDL 的每行平均与 ResNet18 系列的每行平均评估范围不同，数值不完全可比。")
    lines.append("> 同范围比较请参考下方的**通道对齐分组统计**。")
    lines.append("")

    # Header
    header = (
        "| ID | Dataset | "
        "HEDL FPR95↓ | HEDL AUROC↑ | HEDL AUPRC↑ | "
        "REEDL FPR95↓ | REEDL AUROC↑ | REEDL AUPRC↑ | "
        "RSNN FPR95↓ | RSNN AUROC↑ | RSNN AUPRC↑ | "
        "FQN+H FPR95↓ | FQN+H AUROC↑ | FQN+H AUPRC↑ | FQN Pairs |"
    )
    sep = (
        "|----|---------|"
        "-----------|-----------|-----------|"
        "------------|------------|------------|"
        "------------|------------|------------|"
        "------------|------------|------------|-----------|"
    )
    lines.append(header)
    lines.append(sep)

    acc = {"hedl": [], "reedl": [], "rsnn": [], "fqn": []}

    for code in _ALL_IDS:
        h = hedl.get(code)
        r = reedl.get(code)
        s = rsnn.get(code)
        fq = fqn.get(code)

        # Determine display name
        name = None
        for src in [h, r, s, fq]:
            if src and src.get("name"):
                name = src["name"]
                break
        if name is None:
            name = code

        row = f"| {code} | {name} |"

        for src, key in [(h, "hedl"), (r, "reedl"), (s, "rsnn"), (fq, "fqn")]:
            if src:
                row += f" {src['fpr95']:.2f} | {src['auroc']*100:.2f} | {src['auprc']*100:.2f} |"
                acc[key].append(src)
            else:
                row += " — | — | — |"

        # Pairs column (FQN only)
        if fq:
            row += f" {fq.get('pairs', '?')} |"
        else:
            row += " — |"

        lines.append(row)

    # Averages
    def avg(vals, field, mult=1.0):
        if not vals:
            return " — "
        return f"{np.mean([v[field]*mult for v in vals]):.2f}"

    avg_row = "| **AVG** | |"
    for key in ["hedl", "reedl", "rsnn", "fqn"]:
        vals = acc[key]
        avg_row += f" **{avg(vals, 'fpr95')}** | **{avg(vals, 'auroc', 100)}** | **{avg(vals, 'auprc', 100)}** |"
    avg_row += " |"
    lines.append(avg_row)

    # Counts
    lines.append("")
    lines.append(f"| **Count** | | {len(acc['hedl'])}/20 | | | {len(acc['reedl'])}/20 | | | {len(acc['rsnn'])}/20 | | | {len(acc['fqn'])}/20 | | | |")

    return lines


def build_channel_aligned_table(fqn_json_path, reedl_json_path, rsnn_json_path):
    """Build channel-aligned comparison: all models on same (ID, OOD) pairs.

    For each channel group, compute per-model average over only those OOD pairs
    that FQN+HEDL can evaluate (same-channel).
    """
    with open(fqn_json_path, 'r') as f:
        fqn_data = json.load(f)
    with open(reedl_json_path, 'r') as f:
        reedl_data = json.load(f)
    with open(rsnn_json_path, 'r') as f:
        rsnn_data = json.load(f)

    # Group datasets by channel
    groups = {}
    for code in _ALL_IDS:
        ch = CHANNEL_MAP.get(code, 1)
        groups.setdefault(ch, []).append(code)

    lines = []
    lines.append("## 通道对齐分组统计 (Channel-Aligned Group Statistics)")
    lines.append("")
    lines.append("以下统计确保所有模型在**完全相同的 (ID, OOD) 对**上评估：")
    lines.append("每个 ID 仅与其同通道的 OOD 数据集比较。")
    lines.append("这消除了评估范围的差异，使得 FQN+HEDL 与 ResNet18 系列可以直接对比。")
    lines.append("")

    for ch in sorted(groups.keys()):
        ids = groups[ch]
        if len(ids) < 2:
            continue  # need at least 1 OOD pair

        lines.append(f"### ch={ch} ({len(ids)} datasets: {', '.join(ids)})")
        lines.append("")
        header = (
            "| ID | REEDL FPR95↓ | REEDL AUROC↑ | REEDL AUPRC↑ | "
            "RSNN FPR95↓ | RSNN AUROC↑ | RSNN AUPRC↑ | "
            "FQN+H FPR95↓ | FQN+H AUROC↑ | FQN+H AUPRC↑ |"
        )
        sep = (
            "|----|-------------|-------------|-------------|"
            "-------------|-------------|-------------|"
            "-------------|-------------|-------------|"
        )
        lines.append(header)
        lines.append(sep)

        for id_code in ids:
            re = reedl_data.get(id_code, {})
            rs = rsnn_data.get(id_code, {})
            fq = fqn_data.get(id_code, {})

            # Compute channel-aligned average for REEDL: only same-ch OOD pairs
            re_per = re.get("per_ood", {})
            re_ch_oods = {k: v for k, v in re_per.items() if k in ids and k != id_code}
            if re_ch_oods:
                re_fpr = np.mean([v["fpr95"] for v in re_ch_oods.values()])
                re_auc = np.mean([v["auroc"] for v in re_ch_oods.values()])
                re_apr = np.mean([v["auprc"] for v in re_ch_oods.values()])
            else:
                re_fpr = re.get("avg_fpr95", float('nan'))
                re_auc = re.get("avg_auroc", float('nan'))
                re_apr = re.get("avg_auprc", float('nan'))

            rs_per = rs.get("per_ood", {})
            rs_ch_oods = {k: v for k, v in rs_per.items() if k in ids and k != id_code}
            if rs_ch_oods:
                rs_fpr = np.mean([v["fpr95"] for v in rs_ch_oods.values()])
                rs_auc = np.mean([v["auroc"] for v in rs_ch_oods.values()])
                rs_apr = np.mean([v["auprc"] for v in rs_ch_oods.values()])
            else:
                rs_fpr = rs.get("avg_fpr95", float('nan'))
                rs_auc = rs.get("avg_auroc", float('nan'))
                rs_apr = rs.get("avg_auprc", float('nan'))

            fq_per = fq.get("per_ood", {})
            fq_ch_oods = {k: v for k, v in fq_per.items() if k in ids and k != id_code}
            if fq_ch_oods:
                fq_fpr = np.mean([v["fpr95"] for v in fq_ch_oods.values()])
                fq_auc = np.mean([v["auroc"] for v in fq_ch_oods.values()])
                fq_apr = np.mean([v["auprc"] for v in fq_ch_oods.values()])
            else:
                fq_fpr = fq.get("avg_fpr95", float('nan'))
                fq_auc = fq.get("avg_auroc", float('nan'))
                fq_apr = fq.get("avg_auprc", float('nan'))

            row = f"| {id_code} |"
            row += f" {re_fpr:.2f} | {re_auc*100:.2f} | {re_apr*100:.2f} |"
            row += f" {rs_fpr:.2f} | {rs_auc*100:.2f} | {rs_apr*100:.2f} |"
            row += f" {fq_fpr:.2f} | {fq_auc*100:.2f} | {fq_apr*100:.2f} |"
            lines.append(row)

        lines.append("")

    return lines


if __name__ == "__main__":
    print("Loading benchmark results...")

    hedl = {}
    if HEDL_TXT.exists():
        hedl = parse_hedl_txt(HEDL_TXT)
        print(f"  HEDL (txt): {len(hedl)} datasets")

    reedl = {}
    if REEDL_JSON.exists():
        reedl = load_json_avg(REEDL_JSON)
        print(f"  REEDL (json): {len(reedl)} datasets")

    rsnn = {}
    if RSNN_JSON.exists():
        rsnn = load_json_avg(RSNN_JSON)
        print(f"  RSNN (json): {len(rsnn)} datasets")

    fqn = {}
    if FQN_JSON.exists():
        fqn = load_json_avg(FQN_JSON)
        print(f"  FQN+HEDL (json): {len(fqn)} datasets")

    # Build tables
    main = build_main_table(hedl, reedl, rsnn, fqn)
    channel = []
    if FQN_JSON.exists() and REEDL_JSON.exists() and RSNN_JSON.exists():
        channel = build_channel_aligned_table(FQN_JSON, REEDL_JSON, RSNN_JSON)

    output_path = _BASE / "comparison_all_four.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(main))
        f.write('\n\n---\n\n')
        f.write('\n'.join(channel))

    print(f"\nWritten: {output_path}")
    print("Done. Open comparison_all_four.md to view.")
