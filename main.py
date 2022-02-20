import os
import random
import argparse
import yaml

import numpy as np
import torch

from munch import Munch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed : {seed}")


def main(arg, config):
    seed_everything(config.seed)
    if arg.mode == "train":
        pass
    elif arg.mode == "test":
        pass
    else:
        raise Exception(f"Incorrect Mode {arg.mode}")


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='BoostCamp AI tech, Level 1 - P Stage, Image Classification Model')
    parse.add_argument('--mode', default="train", type=str, help="Set mode (train, test)")
    # parse.add_argument('-r', '--resume', default=None, type=str, help='')

    args = parse.parse_args()
    with open('config/config.yaml', 'r') as f:
        conf = yaml.safe_load(f)
    conf = Munch(conf)
    main(args, conf)
