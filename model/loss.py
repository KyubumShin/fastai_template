import torch.nn as nn


def get_loss(config):
    loss = config.loss
    if loss == "CE":
        return nn.CrossEntropyLoss()
    if loss == "BCE":
        return nn.BCEWithLogitsLoss()