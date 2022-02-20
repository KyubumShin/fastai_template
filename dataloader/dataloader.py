from fastai.vision.all import *


def get_data(config):
    dls = ImageDataLoaders.from_folder(config.data.train_path,
                                       test=config.data.test_path,
                                       )




