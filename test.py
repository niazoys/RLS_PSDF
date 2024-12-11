import datetime
import os
import random
import traceback

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import  open_dict
from tqdm import tqdm

from dataloader.dataloader import create_dataloader
from dataset.datasets import DataloaderMode
from model.model_handler import Model_handler
from model.net import UnetBart
from utils.utils import (get_logger, is_logging_process, print_config,
                         set_random_seed,modify_learning_rate)
from utils.writer import Writer
from utils.loss import WeightedMSELoss

def setup(cfg, rank):

    # if your GPU is not from nvidia then please comment out this
    torch.backends.cudnn.benchmark = True
    
    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.dist.mode,
        rank=rank,
        world_size=cfg.dist.gpus,
        timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)


def train_loop(rank, cfg):
    logger = get_logger(cfg, os.path.basename(__file__))
    if cfg.device == "cuda" and cfg.dist.gpus != 0:
        cfg.device = rank
        # turn off background generator when distributed run is on
        cfg.data.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)
        writer = None

    # setup writer
    if is_logging_process():
        # set log/checkpoint dir
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
        # set writer (tensorboard / wandb)
        writer = Writer(cfg, "wandb")
        if cfg.data.train_dir == "" or cfg.data.test_dir == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
        logger.info("BackgroundGenerator is turned off when Distributed running is on")

        # download MNIST dataset before making dataloader
        # TODO: This is example code. You should change this part as you need
 
    # Sync dist processes (because of download MNIST Dataset)
    if cfg.dist.gpus != 0:
        dist.barrier()

    # make dataloader

    if is_logging_process():
        logger.info("Making test dataloader...")
    cfg.data.data_root_dir='E:\oct_data\oct_farsiu\split_test\AMD'
    test_loader = create_dataloader(cfg, DataloaderMode.test, rank)

    # init Model
    net_arch = UnetBart(num_class=cfg.model.output_size)
    net_arch=net_arch.cuda()
    # loss_f = WeightedL1Loss()
    loss_f= WeightedMSELoss()
    model = Model_handler(cfg, net_arch, loss_f, writer, rank)

    # load training state / network checkpoint
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            logger.info("Starting new training run.")

    try: 
        model.validate_model(test_loader)
    except Exception as e:
        if is_logging_process():
            logger.error(traceback.format_exc())
            print(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()


@hydra.main(version_base="1.1", config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    
    train_loader=create_dataloader(hydra_cfg,DataloaderMode.train,0)
    next(iter(train_loader))
    
if __name__ == "__main__":
    main()

