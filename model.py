import torch
import torch.nn as nn
from torchvision import models


class PetResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze all layers first
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Unfreeze the deepest ResNet block
        for parameter in self.model.layer4.parameters():
            parameter.requires_grad = True

        # Replace the final classifier
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            37
        )

    def forward(self, x):
        return self.model(x)


model = PetResNet()
print(model)