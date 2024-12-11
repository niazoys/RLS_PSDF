import argparse
import datetime
import os
import random
import sys
import traceback


import hydra
import torch
from tqdm import tqdm
from utils.writer import Writer
from omegaconf import open_dict
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from dataset.datasets import DataloaderMode
from dataloader.dataloader import create_dataloader
from utils.utils import (get_logger, is_logging_process, 
                         print_config,set_random_seed)


def setup(cfg, rank):

    # if your GPU is not from nvidia then please comment out this
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

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
    mp.spawn(fn, args=(cfg, ), nprocs=cfg.dist.gpus, join=True)


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
        if cfg.data.data_root_dir == "" or cfg.data.data_root_dir_3d == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
        logger.info(
            "BackgroundGenerator is turned off when Distributed running is on")

    # This is just to avoid accident
    cfg.inference_mode = False
    cfg.testing_noise = False

    # Sync dist processes (because of download MNIST Dataset)
    if cfg.dist.gpus != 0:
        dist.barrier()

    if is_logging_process():
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(cfg, DataloaderMode.train, rank)
    if is_logging_process():
        logger.info("Making validation dataloader...")
    val_loader = create_dataloader(cfg, DataloaderMode.validation, rank)
    test_loader = create_dataloader(cfg, DataloaderMode.test, rank)

    
    # TODO: move all the argument inside the model class
    logger.info(cfg.model_obj)
    net_arch= hydra.utils.instantiate(cfg.model_obj,
                        num_class=cfg.model.num_class,
                        gaussian_output=cfg.gaussian_output,
                        out_act=cfg.model.out_act,groupNorm=cfg.sdm_misc.groupNorm,
                        use_multi_head=cfg.sdm_misc.use_multi_head,
                        use_input_instance_norm=cfg.model.use_input_instance_norm).to(cfg.device)
    
    logger.info(cfg.loss_obj)
    loss_f= hydra.utils.instantiate(cfg.loss_obj,cfg)

    
    logger.info(cfg.handler_obj)
    model= hydra.utils.instantiate(cfg.handler_obj,cfg,net_arch,loss_f,
                        writer,
                        rank)

    # load training state / network checkpoint
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            logger.info("Starting new training run.")

    try:
        if cfg.dist.gpus == 0 or cfg.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = cfg.dist.gpus

        for epoch in tqdm(range(model.epoch + 1, cfg.train.num_epoch,
                                epoch_step),
                          desc="Epoch",
                          unit='epoch'):
            
            # Call the scheduler on each epoch            
            
            model.epoch = epoch
            model.train_model(train_loader)
            model.validate_model(val_loader)
            model.log_ped_case(test_loader)
            if cfg.log.save_model_each_epoch:
                model.save_training_state()

        if is_logging_process():
            logger.info("End of Train")
    except Exception as e:
        if is_logging_process():
            logger.error(traceback.format_exc())
            print(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()


def get_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment',type=str, default='default.yaml')
    args = parser.parse_args()
    exp = args.experiment
    if exp.startswith("experiment="):
        exp = exp[len("experiment=") :]
    exp = exp.replace("/", ".")
    return exp



@hydra.main(version_base="1.1", config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    
    print_config(hydra_cfg,get_logger(hydra_cfg, os.path.basename(__file__),
                   disable_console=True))
    # random seed
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    if hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        train_loop(0, hydra_cfg)
    else:
        # because ${hydra:runtime.cwd} is not support for DDP
        hydra_cfg.work_dir = hydra_cfg.work_dir
        distributed_run(train_loop, hydra_cfg)


if __name__ == "__main__":
    experiment = get_experiment()
    print('experiment: ', experiment)
    sys.argv.append(f'exp_name={experiment}')
    main()
