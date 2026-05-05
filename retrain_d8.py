"""Retrain D8 with bias-fixed SetKernelPooling."""
import torch
from train_v2 import train_full_pipeline_v2
from eval_ood_v2 import evaluate_ood_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_full_pipeline_v2(
    "D8", device=device, stage1_epochs=200, stage2_epochs=50,
    skip_stage1_if_exists=True,
)
print("\n=== OOD Evaluation ===")
evaluate_ood_v2("D8", device=device)
