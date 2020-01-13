from typing import Tuple, Optional

from torch.utils.data.dataloader import DataLoader

from .model_desc import RunMode, ModelDesc
from .macro_builder import MacroBuilder
from .micro_builder import MicroBuilder
from ..common.config import Config
from .model import Model
from ..common.data import get_dataloaders
from ..common.common import get_logger, expdir_abspath
from ..common.check_point import CheckPoint


def create_model_desc(conf_model_desc: Config, run_mode:RunMode,
                 micro_builder: Optional[MicroBuilder]=None,
                 template_model_desc:Optional[ModelDesc]=None) -> ModelDesc:
    builder = MacroBuilder(conf_model_desc,
                               run_mode=run_mode,
                               template=template_model_desc)
    model_desc = builder.get_model_desc()

    if micro_builder:
        micro_builder.register_ops()
        # if nodes are already built by template, do not invoke micro builder
        if template_model_desc is None:
            micro_builder.build(model_desc)
    return model_desc

def create_model(conf_model_desc: Config, device, run_mode:RunMode,
                 micro_builder: Optional[MicroBuilder]=None,
                 template_model_desc:Optional[ModelDesc]=None) -> Model:
    model_desc = create_model_desc(conf_model_desc, run_mode,
                                   micro_builder=micro_builder,
                                   template_model_desc=template_model_desc)
    return model_from_desc(model_desc, device)

def model_and_checkpoint(conf_checkpoint:Config, resume:bool,
        full_desc_filename:str, conf_model_desc: Config,
        device, run_mode:RunMode, micro_builder: Optional[MicroBuilder]=None,
        template_model_desc:Optional[ModelDesc]=None)\
                     ->Tuple[Model, Optional[CheckPoint]]:
    logger = get_logger()
    checkpoint = CheckPoint(conf_checkpoint, resume) \
                 if conf_checkpoint is not None else None
    if micro_builder:
        micro_builder.register_ops()
    if checkpoint is None or checkpoint.is_empty():
        logger.info('Checkpoint not found or resume=False, starting from scratch')
        # create model
        model = create_model(conf_model_desc, device,
                                    run_mode=RunMode.EvalTrain,
                                    micro_builder=micro_builder,
                                    template_model_desc=template_model_desc)
        model.desc.save(full_desc_filename) # save copy of full model desc
    else:
        logger.info('Checkpoint found, loading last model')
        model_desc = ModelDesc.load(full_desc_filename)
        model = model_from_desc(model_desc, device)

    return model, checkpoint

def model_from_desc(model_desc, device)->Model:
    model = Model(model_desc)
    # TODO: enable DataParallel
    # if data_parallel:
    #     model = nn.DataParallel(model).to(device)
    # else:
    model = model.to(device)
    return model

def get_data(conf_loader:Config)\
        -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    # region conf vars
    # dataset
    conf_data = conf_loader['dataset']
    ds_name = conf_data['name']
    max_batches = conf_data['max_batches']
    dataroot = conf_data['dataroot']

    aug = conf_loader['aug']
    cutout = conf_loader['cutout']
    val_ratio = conf_loader['val_ratio']
    batch_size = conf_loader['batch']
    val_fold = conf_loader['val_fold']
    n_workers = conf_loader['n_workers']
    horovod = conf_loader['horovod']
    load_train = conf_loader['load_train']
    load_test = conf_loader['load_test']
    # endregion

    train_dl, val_dl, test_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=load_train, load_test=load_test,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod,
        n_workers=n_workers, max_batches=max_batches)
    assert train_dl is not None
    return train_dl, val_dl, test_dl
