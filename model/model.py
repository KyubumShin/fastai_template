import torch
import torch.nn as nn
import timm
from munch import Munch


class BasicClassification(nn.Module):
    def __init__(self, **config: Munch):
        super(BasicClassification, self).__init__()
        self.config = config
        self.feature = timm.create_model(model_name=self.config.class_name,
                                         pretrained=True,
                                         num_classes=self.config.num_classes)

    def forward(self, x):
        x = self.feature(x)
        return x
