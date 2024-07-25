import torch.nn as nn
import torchvision.models as models


class Resnet_50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x
