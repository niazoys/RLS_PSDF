
from omegaconf.dictconfig import DictConfig
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
from dataset.datasets import DataloaderMode, DukeOCTLayerDataset, DukeOCTLayerDataset3d, DukeOCTLayerDatasetDirect, DukeOCTLayerDatasetSDM


class DataLoader_(DataLoader):
    # ref: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#issuecomment-495090086
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(cfg, mode, rank):
    if cfg.data.use_background_generator:
        data_loader = DataLoader_
    else:
        data_loader = DataLoader

    ###############################################################################
    # Remark: hydra.utils.instantiate is not for distributed Data Parallel training 
    # Replace next line with `dataset = Dataset_(cfg, mode)`
    ###############################################################################
    # dataset = hydra.utils.instantiate(cfg.gen_dataset, cfg=cfg, mode=mode)
    # dataset = Dataset_(cfg, mode)
    dataset = Dataset_(cfg,mode)

    train_use_shuffle = False
    sampler=None
    if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
        sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        train_use_shuffle = False
    
    if mode is DataloaderMode.train:
        # Sampler to oversample PED
        # if cfg.track_oversampling:
        #     sampler = WeightedRandomSampler(weights=dataset.dataset.weight, num_samples=len(dataset.dataset.weight)+200, replacement=True)
        return data_loader(
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif mode is DataloaderMode.validation:
        return data_loader(
            dataset=dataset,
            batch_size=cfg.test.batch_size,
            shuffle=True,
            sampler=sampler,
            num_workers=cfg.test.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif  mode is DataloaderMode.test:
        return data_loader(
            dataset=dataset,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=cfg.test.num_workers,
            pin_memory=False,
            drop_last=True,
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")


class Dataset_(Dataset):
    '''
        
        This shell Dataset class to wrap the project wise Dataset class 
        to that we don't need to change the DataLoader/train/test 
        interface of the template.

        cfg: Hydra Config object
        mode: Dataloder mode (train,test,validation)

    '''
    def __init__(
        self, 
        cfg: DictConfig = None,
        mode: DataloaderMode = DataloaderMode.train
        ):
        self.cfg = cfg
        self.mode = mode

        if cfg.sdm:
            self.dataset=DukeOCTLayerDatasetSDM(cfg=cfg,mode=self.mode)
        else:
            self.dataset=DukeOCTLayerDatasetDirect(cfg=cfg,mode=self.mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
            
        return self.dataset[idx]

