import copy
import logging
import random
from logging import Logger
from typing import Sequence

import numpy as np
import rich.syntax
import rich.tree
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf



def unfreeze_dropout(net):
    """Set Dropout mode to train or eval."""
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
                m.train()
    return net


def to_numpy(input):
    return input.cpu().detach().numpy()


def modify_learning_rate(optim:torch.optim,factor=2):
    '''
    modifies lr of the model after initialization 
    of the optimizer object.
    '''
    for g in optim.param_groups:
        g['lr'] = g['lr']/factor

    return optim


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None, disable_console=False):
    if disable_console:
        # stop output to stdout; only output to log file
        if 'console' in cfg.job_logging_cfg.root.handlers:
            cfg = copy.deepcopy(cfg)
            cfg.job_logging_cfg.root.handlers.remove('console')
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
        )
        return logging.getLogger(name)

def print_config(
    config: DictConfig,
    logger: Logger,
    fields: Sequence[str] = (
        "name",
        "device",
        "working_dir",
        "random_seed",
        "model",
        "train",
        "test",
        "gen_dataset",
        "data",
        "dist",
        "log",
        "job_logging_cfg",
        "dist"
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    all_cfg_str = OmegaConf.to_yaml(config)
    logger.info("Config:\n" + all_cfg_str)



# def make_batch_from_list(data_loader,main_list):
#     for p in range(0,int(len(main_list)/2)):
#         sample_list = main_list[p*2:p*2+2]
#         model_input, model_target,layer=[],[],[]
#         for i in sample_list:
#             input, target, _, lyr, _ = data_loader.dataset[i]
#             model_input.append(input) 
#             model_target.append(target) 
#             layer.append(lyr)

#         model_input, model_target, layer = torch.stack(model_input), torch.stack(model_target),torch.stack(layer)