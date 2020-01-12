from typing import Callable, Tuple, Optional

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .tester import Tester
from .config import Config
from . import utils
from ..common.common import get_logger
from ..common.check_point import CheckPoint

class Trainer(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module, device,
                 check_point:Optional[CheckPoint])->None:
        # region config vars
        conf_lossfn = conf_train['lossfn']
        self._aux_weight = conf_train['aux_weight']
        self._grad_clip = conf_train['grad_clip']
        self._drop_path_prob = conf_train['drop_path_prob']
        self._logger_freq = conf_train['logger_freq']
        self._title = conf_train['title']
        self._epochs = conf_train['epochs']
        self._conf_optim = conf_train['optimizer']
        self._conf_sched = conf_train['lr_schedule']
        conf_validation = conf_train['validation']
        # endregion

        self.check_point = check_point
        self.model = model
        self.device = device
        self._lossfn = utils.get_lossfn(conf_lossfn).to(device)
        self._tester = Tester(conf_validation, model, device, epochs=self._epochs) \
                        if conf_validation else None
        self._metrics = self._create_metrics(self._epochs)
        self._metrics.custom['param_byte_size'] = utils.param_size(self.model)

    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        logger = get_logger()
        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        optim = self.get_optimizer()
        lr_scheduler = self.get_scheduler(optim)

        start_epoch = 0
        if self.check_point is not None and 'trainer' in self.check_point:
            start_epoch = self._restore_checkpoint(optim, lr_scheduler)

        self.pre_fit(train_dl, val_dl, optim, lr_scheduler, start_epoch>0)

        if start_epoch >= self._epochs:
            logger.warn(f'fit exit: start epoch {start_epoch} > {self._epochs}')
            return # we already finished the run, we might be checkpointed

        self.check_point.clear()
        for epoch in range(start_epoch, self._epochs):
            self._set_drop_path(epoch, self._epochs)

            self.pre_epoch(train_dl, val_dl, optim, lr_scheduler)
            self._train_epoch(train_dl, optim, lr_scheduler)
            self.post_epoch(train_dl, val_dl, optim, lr_scheduler)

        self.post_fit(train_dl, val_dl, optim, lr_scheduler)

    def get_optimizer(self)->Optimizer:
        return utils.get_optimizer(self._conf_optim, self.model.parameters())

    def get_scheduler(self, optim:Optimizer)->_LRScheduler:
        return utils.get_lr_scheduler(self._conf_sched, self._epochs, optim)

    def get_metrics(self)->Tuple[Metrics, Optional[Metrics]]:
        return self._metrics, self._tester.get_metrics() if self._tester else None

    #########################  hooks #########################
    def pre_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                optim:Optimizer, sched:_LRScheduler, resuming:bool)->None:
        self._metrics.pre_run(resuming)
        if self._tester:
            self._tester.pre_test(resuming)

    def post_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                 optim:Optimizer, sched:_LRScheduler)->None:
        if self._tester:
            self._tester.post_run()
        self._metrics.post_run()

    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  optim:Optimizer, sched:_LRScheduler)->None:
        self._metrics.pre_epoch(lr=optim.param_groups[0]['lr'])

    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                   optim:Optimizer, sched:_LRScheduler)->None:
        # first run test before checkpointing
        if val_dl and self._tester:
            self._tester.test_epoch(val_dl)

        self._metrics.post_epoch()
        if self.check_point is not None and \
                self._metrics.epoch % self.check_point.frequency == 0:
            self.check_point.new()
            self.update_checkpoint(self.check_point, optim, sched)
            self.check_point.commit()

    def pre_step(self, x:Tensor, y:Tensor, optim:Optimizer)->None:
        self._metrics.pre_step(x, y)

    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int)->None:
        self._metrics.post_step(x, y, logits, loss, steps)
    #########################  hooks #########################

    def _restore_checkpoint(self, optim:Optimizer, sched:_LRScheduler)->int:
        logger = get_logger()

        last_epoch = self._metrics.epoch
        assert last_epoch > 0, f'While restoring from checkpoint epoch > 0 is expected but it is {last_epoch}'
        start_epoch = last_epoch + 1

        state = self.check_point['trainer']
        self.model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])
        sched.load_state_dict(state['sched'])
        if self._tester:
            self._tester.load_state_dict(state['tester'])

        logger.warn(f'fit will continue from epoch {start_epoch}')
        return start_epoch

    def update_checkpoint(self, check_point:CheckPoint,
                          optim:Optimizer, sched:_LRScheduler)->None:
        state = {
            'metrics': self._metrics.state_dict(),
            'model': self.model.state_dict(),
            'optim': optim.state_dict(),
            'sched': sched.state_dict(),
            'tester': self._tester.state_dict() if self._tester is not None else None
        }
        self.check_point['trainer'] = state

    def _create_metrics(self, epochs:int):
        logger = get_logger()
        m = Metrics(self._title, epochs,logger_freq=self._logger_freq)
        if self.check_point is not None and 'trainer' in self.check_point:
            logger.warn('Metrics loaded from exisitng checkpoint')
            m.load_state_dict(self.check_point['trainer']['metrics'])
        return m

    def _train_epoch(self, train_dl: DataLoader,
                     optim:Optimizer, sched:_LRScheduler)->None:
        steps = len(train_dl)
        self.model.train()
        for x, y in train_dl:
            assert self.model.training # derived class might alter the mode

            # enable non-blocking on 2nd part so its ready when we get to it
            x, y = x.to(self.device), y.to(self.device, non_blocking=True)

            self.pre_step(x, y, optim)

            optim.zero_grad()

            if self._aux_weight > 0.0:
                logits, aux_logits = self.model(x)
            else:
                (logits, *_), aux_logits = self.model(x), None
            loss = self.compute_loss(self._lossfn, x, y, logits,
                                    self._aux_weight, aux_logits)

            loss.backward()

            if self._grad_clip:
                # TODO: original darts clips alphas as well but pt.darts doesn't
                nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)

            optim.step()
            sched.step()

            self.post_step(x, y, logits, loss, steps)

    def compute_loss(self, lossfn:Callable,
                     x:Tensor, y:Tensor, logits:Tensor,
                     aux_weight:float, aux_logits:Optional[Tensor])->Tensor:
        logger = get_logger()
        loss = lossfn(logits, y)
        if aux_weight > 0.0:
            if aux_logits is not None:
                loss += aux_weight * lossfn(aux_logits, y)
            else:
                logger.warn(f'aux_weight is {aux_weight} but aux tower was not generated')
        return loss

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self._drop_path_prob:
            drop_prob = self._drop_path_prob * epoch / epochs
            # set value as property in model (it will be used by forward())
            # this is necessory when using DataParallel(model)
            # https://github.com/pytorch/pytorch/issues/16885
            m = self.model
            if hasattr(self.model, 'module'): # for data parallel model
                m = self.model.module
            if hasattr(m, 'drop_path_prob'):
                m.drop_path_prob(drop_prob)
            else:
                raise RuntimeError('Drop path value {} was specified but model'
                                   ' does not have drop_path_prob() method'\
                                       .format(self._drop_path_prob))
