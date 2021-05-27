import os
import sys
import time

import numpy as np
import torch

from data_util.logging import logger
from data_util.utils import calc_running_avg_loss
from fastNLP.core.callback import Callback, EarlyStopError


class TrainCallback(Callback):
    def __init__(self, config, patience=10, quit_all=True):
        super().__init__()
        self.config = config
        self.patience = patience
        self.wait = 0
        self.running_avg_loss = 0
        self.loss_update_every = []
        if type(quit_all) != bool:
            raise ValueError("In KeyBoardInterrupt, quit_all arguemnt must be a bool.")
        self.quit_all = quit_all

    def on_epoch_begin(self):
        self.epoch_start_time = time.time()
        if self.epoch == self.config.coverage_at:
            self.config.is_coverage = True
        if self.config.is_coverage:
            self.trainer.do_valid = True
        else:
            self.trainer.do_valid = False

    def on_backward_begin(self, loss):
        self.loss_update_every.append(loss.item())
        if isinstance(loss, tuple) and not np.isfinite(loss[0].item()):
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss[0].item())
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
                    logger.info(param.grad.data.sum())
            raise Exception("train Loss is not finite. Stopping.")

        if self.step % self.update_every == 0:
            assert len(self.loss_update_every) == self.update_every
            loss_batch = sum(self.loss_update_every)
            self.loss_update_every = []
            # report the loss
            if self.step < 10 or self.step % 1000 == 0:
                logger.info("|epoch: %d  step: %d  log_loss: %.4f |"
                            % (self.epoch, self.step / self.update_every, loss_batch))
            self.running_avg_loss = calc_running_avg_loss(loss_batch, self.running_avg_loss,
                                                          self.step / self.update_every)

    def on_backward_end(self):
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

    def on_epoch_end(self):
        logger.info(
            '   | end of epoch {:3d} | time: {:5.2f}s | '.format(self.epoch, (time.time() - self.epoch_start_time)))

    def on_valid_begin(self):
        self.valid_start_time = time.time()

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        logger.info(
            '   | end of valid {:3d} | time: {:5.2f}s | '.format(self.epoch, (time.time() - self.valid_start_time)))
        # save the better checkpoint
        if is_better_eval:
            logger.info("got better results on dev, save checkpoint.. ")
            model_save_path = os.path.join(self.config.model_path,
                                           f'CGSum_{self.config.setting}_{self.config.n_hop}hopNbrs.pt')
            checkpoint = {"state_dict": self.model.state_dict(), "config": self.model.config.__dict__}
            torch.save(checkpoint, model_save_path)

        # early stop
        if not is_better_eval:
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            if self.quit_all is True:
                sys.exit(0)
            else:
                pass
        else:
            raise exception


class LRDecayCallback(Callback):
    def __init__(self, parameters, decay_rate=1e-3, steps=100):
        super().__init__()
        self.paras = parameters
        self.decay_rate = decay_rate
        self.steps = steps

    def on_step_end(self):
        if self.step % self.update_every == 0:
            step = self.step // self.update_every
            if step % self.steps == 0:
                for para in self.paras:
                    para['lr'] = para['lr'] * (1 - self.decay_rate)

