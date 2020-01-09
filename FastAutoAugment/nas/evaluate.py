from typing import Optional
import os

import torch
import yaml

from ..common import utils
from ..common.trainer import Trainer
from ..common.config import Config
from ..common.common import get_logger, logdir_abspath
from .model_desc import RunMode
from .micro_builder import MicroBuilder
from . import nas_utils
from torch import mode

def eval_arch(conf_eval:Config, micro_builder:Optional[MicroBuilder]):
    logger = get_logger()

    # region conf vars
    conf_loader       = conf_eval['loader']
    model_desc_file = conf_eval['model_desc_file']
    save_filename    = conf_eval['save_filename']
    conf_model_desc   = conf_eval['model_desc']
    conf_train = conf_eval['trainer']
    # endregion

    # load model desc file to get template model
    model_desc_filepath = logdir_abspath(model_desc_file)
    assert model_desc_filepath
    if not os.path.exists(model_desc_filepath):
        raise RuntimeError("Model description file for evaluation is not found."
              "Typically this file should be generated from the search."
              "Please copy this file to '{}'".format(model_desc_filepath))
    with open(model_desc_filepath, 'r') as f:
        template_model_desc = yaml.load(f, Loader=yaml.Loader)

    device = torch.device(conf_eval['device'])

    # create model
    model = nas_utils.create_model(conf_model_desc, device,
                                   run_mode=RunMode.EvalTrain,
                                   micro_builder=micro_builder,
                                   template_model_desc=template_model_desc)

    # get data
    train_dl, _, test_dl = nas_utils.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None

    trainer = Trainer(conf_train, model, device)
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







