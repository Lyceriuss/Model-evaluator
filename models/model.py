# models/model.py

import torch.nn as nn
from torchvision import models

class ClothingModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet50'):
        """
        Initializes the ClothingModel with the specified architecture.

        Args:
            num_classes (int): Number of output classes.
            model_name (str): Model architecture to use ('resnet50', 'efficientnet_b0', 'densenet121').
        """
        super(ClothingModel, self).__init__()
        self.model_name = model_name

        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
        elif model_name == 'densenet121':
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        elif model_name == 'vit-base-patch16-224':
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            num_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(num_features, num_classes)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def forward(self, x):
        return self.model(x)
