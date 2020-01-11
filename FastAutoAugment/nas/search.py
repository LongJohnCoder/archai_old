
from typing import Type

import torch

from ..common.common import get_logger
from ..common.config import Config
from .model_desc import ModelDesc, RunMode
from .micro_builder import MicroBuilder
from .arch_trainer import ArchTrainer
from . import nas_utils

def search_arch(conf_search:Config, micro_builder:MicroBuilder,
                trainer_class:Type[ArchTrainer])->None:
    logger = get_logger()

    # region config vars
    conf_model_desc = conf_search['model_desc']
    conf_loader = conf_search['loader']
    conf_train = conf_search['trainer']
    conf_checkpoint = conf_search['checkpoint']
    resume = conf_search['resume']
    final_desc_filename = conf_search['final_desc_filename']
    full_desc_filename = conf_search['full_desc_filename']
    # endregion

    device = torch.device(conf_search['device'])

    model, checkpoint = nas_utils.model_and_checkpoint(
                                conf_checkpoint, resume, full_desc_filename,
                                conf_model_desc, device,
                                run_mode=RunMode.Search,
                                micro_builder=micro_builder)

    # get data
    train_dl, val_dl, _ = nas_utils.get_data(conf_loader)
    assert train_dl is not None

    # search arch
    arch_trainer = trainer_class(conf_train, model, device, checkpoint)
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
    save_path = found_model_desc.save(final_desc_filename)
    if save_path:
        logger.info(f"Best architecture saved in {save_path}")
    else:
        logger.info("Best architecture is not saved because file path config not set")


