from typing import Optional

import torch

from FastAutoAugment.common.trainer import Trainer
from FastAutoAugment.common.check_point import CheckPoint
from FastAutoAugment.common.config import Config
from FastAutoAugment.common.common import get_logger, common_init
from FastAutoAugment.nas import nas_utils
from dawnnet import DawnNet

def train_test(conf_eval:Config):
    logger = get_logger()

    # region conf vars
    conf_loader       = conf_eval['loader']
    save_filename    = conf_eval['save_filename']
    conf_checkpoint = conf_eval['checkpoint']
    resume = conf_eval['resume']
    conf_train = conf_eval['trainer']
    # endregion

    device = torch.device(conf_eval['device'])
    checkpoint = CheckPoint(conf_checkpoint, resume) if conf_checkpoint is not None else None
    model = DawnNet().to(device)

    # get data
    train_dl, _, test_dl = nas_utils.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None


    trainer = Trainer(conf_train, model, device, checkpoint)
    trainer.fit(train_dl, test_dl)

    # save metrics
    train_metrics, test_metrics = trainer.get_metrics()
    train_metrics.report_best()
    train_metrics.save('eval_train_metrics')
    if test_metrics:
        test_metrics.report_best()
        test_metrics.save('eval_test_metrics')

    # save model
    save_path = model.save(save_filename)
    if save_path:
        logger.info(f"Model saved in {save_path}")
    else:
        logger.info("Model is not saved because file path config not set")

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/darts_cifar.yaml',
                       param_args=['--common.experiment_name', 'dawnnet'])

    conf_eval = conf['nas']['eval']

    conf_eval['checkpoint'] = None
    conf_eval['resume'] = False
    conf_eval['trainer']['epochs'] = 35
    conf_eval['trainer']['drop_path_prob'] = 0.0
    conf_eval['loader']['cutout'] = 0
    conf_eval['trainer']['aux_weight'] = 0.0

    # evaluate architecture using eval settings
    train_test(conf_eval)

    exit(0)

