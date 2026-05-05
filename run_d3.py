"""Train FQN_HEDL_v2 on D3 (SyntheticControl, K=6) full pipeline."""
import torch
from train_v2 import train_full_pipeline_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_full_pipeline_v2("D3", device=device, stage1_epochs=200, stage2_epochs=50)
print("D3 done.")
