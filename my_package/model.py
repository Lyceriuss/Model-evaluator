# my_package/model.py

import torch.nn as nn
from torchvision import models

class ClothingModel(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the ResNet-50 model for classification.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ClothingModel, self).__init__()

        # Load pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)

        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
