import torch
import torchvision.models as models
from cnn_finetune import make_model
import torch.nn as nn
import timm
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, model, output_size, loss='Softmax'):
        super(SimpleCNN, self).__init__()
        if model=='ResNet34':
            # mynet = models.resnet34(pretrained=True)
            mynet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model=='ResNet18':
            # weights = models.ResNet50_Weights.IMAGENET1K_V2
            mynet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model=='ResNet50':
            # weights = models.ResNet50_Weights.IMAGENET1K_V2
            mynet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # mynet.fc = nn.Linear(mynet.fc.in_features, output_size)
        self.loss = loss
        mynet = mynet.module if isinstance(mynet, torch.nn.DataParallel) else mynet
        modules = list(mynet.children())[:-1]
        if loss == 'dropout':
            dropout_layer = nn.Dropout(p=0.2)
            modules.append(dropout_layer)
        self.mynet = nn.Sequential(*modules)
        self.fc = nn.Linear(mynet.fc.in_features, output_size)
        
        
    def forward(self, images):
        features = self.mynet(images)
        if self.loss == 'edl_HENN':
            features = F.relu(features)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return features,out


    def get_weight(self,features):
        henn_out = torch.zeros(len(features),len(self.fc.weight),len(features[0]))
        for i in range(len(features)):
            henn_out[i] = features[i]*self.fc.weight
        return henn_out.cuda()

    def get_weight(self):
        return self.fc.weight