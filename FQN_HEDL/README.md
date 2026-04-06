# ReadMe

## FQN-HEDL

This is a model that integrates FQN with HEDL and can only process 1D data now.

To train the model, use

> python train.py --dataset D9 --model FQN --loss edl_HENN --w 2.0

where D9 is choosen from <D1~D20>(the trained model will be automatically saved), and valid as:

> python val.py --dataset D9 --ood-dataset D11 --checkpoint runs_D9/edl_HENN/acc/checkpoints/checkpoint.pth --loss edl_HENN

where D9,D11 is choosen from <D1~D20>.