import numpy as np
import wandb
from omegaconf.dictconfig import DictConfig
from torch.utils.tensorboard import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, cfg, logdir):
        self.cfg = cfg
        if cfg.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if cfg.log.use_wandb:
            wandb_init_conf = cfg.log.wandb_init_conf
            wandb.init(config=self._wandb_formatter(cfg), **wandb_init_conf)

    def log_image(self, image, logging_name,step):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_image(logging_name, image, step)
        if self.cfg.log.use_wandb:
            wandb.log({logging_name: wandb.Image(image,caption='PED Examples!')}, step=step)

    def logging_with_step(self, value, step, logging_name):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.cfg.log.use_wandb:
            wandb.log({logging_name: value}, step=step)

    def _wandb_formatter(self, cfg):
        formatted_cfg = dict()
        return self._recursive_format('general/', formatted_cfg, cfg)

    def _recursive_format(self, prefix, formatted_cfg, cfg):
        for key, val in dict(cfg).items():
            if not isinstance(val, DictConfig):
                formatted_cfg[prefix + str(key)] = val
            else:
                if prefix == 'general/':
                    self._recursive_format(str(key) + '/', formatted_cfg, val)
                else:
                    self._recursive_format(prefix + str(key) + '/', formatted_cfg, val)
        return formatted_cfg
