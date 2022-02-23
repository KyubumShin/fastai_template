from fastai.vision.all import *


def get_data(config, transpose):
    dls = ImageDataLoaders.from_folder(config.data.data_path,
                                       vaild_pct=1/config.data.num_fold,
                                       seed=config.seed,
                                       item_tfms=transpose,
                                       bs=config.data.batch_size,
                                       )
    return dls




