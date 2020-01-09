
from typing import Type

import torch
import yaml

from ..common.common import get_logger, logdir_abspath
from ..common.config import Config
from .model_desc import RunMode
from .micro_builder import MicroBuilder
from .arch_trainer import ArchTrainer
from . import nas_utils

def search_arch(conf_search:Config, micro_builder:MicroBuilder,
                trainer_class:Type[ArchTrainer])->None:
    logger = get_logger()

    conf_model_desc = conf_search['model_desc']
    model_desc_filename = conf_search['model_desc_file']
    conf_loader = conf_search['loader']
    conf_train = conf_search['trainer']

    device = torch.device(conf_search['device'])

    # create model
    model = nas_utils.create_model(conf_model_desc, device,
                                   run_mode=RunMode.Search,
                                   micro_builder=micro_builder)

    # get data
    train_dl, val_dl, _ = nas_utils.get_data(conf_loader)
    assert train_dl is not None

    # search arch
    arch_trainer = trainer_class(conf_train, model, device)
    arch_trainer.fit(train_dl, val_dl)

    # save metrics
    train_metrics, val_metrics = arch_trainer.get_metrics()
    train_metrics.report_best()
    train_metrics.save('search_train_metrics')
    if val_metrics:
       val_metrics.report_best()
       val_metrics.save('search_val_metrics')

    # save found model
    found_model_desc = arch_trainer.get_model_desc()
    save_path = found_model_desc.save(model_desc_filename)
    if save_path:
        logger.info(f"Best architecture saved in {save_path}")
    else:
        logger.info("Best architecture is not saved because file path config not set")


