from typing import Optional

import torch
from FastAutoAugment import cifar10_models

from FastAutoAugment.common.trainer import Trainer
from FastAutoAugment.common.config import Config
from FastAutoAugment.common.common import get_logger, common_init
from FastAutoAugment.common import data

def train_test(conf_eval:Config):
    logger = get_logger()

    # region conf vars
    conf_loader       = conf_eval['loader']
    conf_trainer = conf_eval['trainer']
    # endregion

    conf_trainer['epochs'] = 5
    conf_loader['batch'] = 128
    conf_loader['cutout'] = 0
    conf_trainer['drop_path_prob'] = 0.0
    conf_trainer['grad_clip'] = 0.0
    conf_trainer['aux_weight'] = 0.0

    device = torch.device(conf_eval['device'])
    Net = cifar10_models.resnet18
    model = Net().to(device)

    # get data
    train_dl, _, test_dl = data.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None

    trainer = Trainer(conf_trainer, model, device, None, False)
    trainer.fit(train_dl, test_dl)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/darts_cifar.yaml',
                       param_args=['--common.experiment_name', 'restnet_test'])

    conf_eval = conf['nas']['eval']

    # evaluate architecture using eval settings
    train_test(conf_eval)

    exit(0)

