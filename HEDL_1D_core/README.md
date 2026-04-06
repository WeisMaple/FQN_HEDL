# ReadMe

## Hyper-opinion Evidential Deep Learning (HEDL)

It's the code for Hyper-opinion Evidential Deep Learning (HEDL).

Please download every package in requirements.txt first, and download the cifar10, cifar100, flower, cub, SVHN, Texture, and Place365 datasets, then storage them in the directory named as '/mnt/Data/xx', e.g., '/mnt/Data/CIFAR10'.



To train an HEDL model from begining, please run the code follows:

First train the backbone with softmax for 90 epoches:

> python train.py --epoch 90 --name HEDL --dataset [dataset]

and then train in HEDL for 10 epoches:

> python train.py --epoch 10 --name HEDL --dataset [dataset] --loss edl_HENN --checkpoint_path [path of the backbone]

where [dataset] is choosen from <cifar10,cifar100,flower,cub>, and valid as:

> python val.py --loss edl_HENN --name HEDL --dataset [dataset] --ood [ood_dataset]

where [ood_dataset] is choosen from <SVHN/dtd/place365>, standing for SVHN/Texture/Place365 datasets.