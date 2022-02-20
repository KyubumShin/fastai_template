import timm
from fastai.vision.all import *
from model.model import BasicClassification
from model.loss import get_loss
from dataloader.dataloader import get_data


def get_learner(config):
    data = get_data(config)
    loss = get_loss(config)
    model = timm.create_model()
    learner = Learner(dls=data, model=model, loss_func=loss, )
    return learner


def get_learner_with_custom_model(config):
    data = get_data(config)
    loss = get_loss(config)
    model = BasicClassification(config)
    learner = Learner(dls=data, model=model, loss_func=loss, )
    return learner
