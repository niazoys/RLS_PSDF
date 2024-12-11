import os

from enum import Enum, auto
import torch.nn.functional as F
import torch
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig
from torchvision.transforms.functional import resize, InterpolationMode, to_tensor
from utils.artifacts import ArtificialArtifact
from monai.transforms import (Compose, RandGaussianNoise, RandAffined,Resized,
                              RandAdjustContrast, RandShiftIntensity, 
                              RandGaussianSmooth,RandSpatialCropd,NormalizeIntensity)
import scipy.io as io
import hydra
import json
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from collections import OrderedDict

class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()
    validation = auto()
    TBD = auto()


class DukeOCTLayerDatasetSDM(Dataset):

    def __init__(self,
                 cfg: DictConfig = None,
                 mode: DataloaderMode = DataloaderMode.TBD) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.DEVICE = cfg.device
        self.external = True
        self.trim_offset = 150
        
        self.artifacts=ArtificialArtifact(prob=1)

        if cfg.train.use_all_data:
            self.path = cfg.data.data_root_dir_all
            print(f"Using all data in {self.path}")
        else:
            self.path = cfg.data.data_root_dir
            print(f"Using selected data in {self.path}")
            
        self.imgs, self.layer_maps = [], []

        self.transformation = Compose([
            NormalizeIntensity(nonzero=False, channel_wise=True),
            RandGaussianNoise(prob=0.25, std=0.15),
            RandAdjustContrast(gamma=(.75, 1.5), prob=0.20),
            RandShiftIntensity(offsets=(0.05, 0.15)),
            RandGaussianSmooth()
        ])

        self.affineTransform = Compose([RandAffined(keys=['image', 'layer_map','mask'],
            rotate_range=(-0.5, 0.5),
            shear_range=(-0.001, 0.001),
            translate_range=(-40,40),
            mode=['area','bilinear','nearest'],
            padding_mode="nearest",
            prob=0.2)   
            ])
        
        self.resize=Compose([Resized(keys=['image','layer_map','mask'],spatial_size=(512,512),mode=('area','bilinear','nearest'))])
        
        if mode == DataloaderMode.train:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'train')
        elif mode == DataloaderMode.validation:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'val')
        elif mode == DataloaderMode.test:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'test')
        else:
            raise ValueError("Invalid dataset mode!")

        self.img_files = os.listdir(os.path.join(self.data_set_path,
                                                 'images/'))

        if self.cfg.debug_mode == 1:
            self.img_files = self.img_files[:150]


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        
        # full_anno=None
        # if self.external:
        #     for i in range(4):
        #         try:
        #             file_name_json =os.path.join(self.data_set_path,'layers/'+self.img_files[idx][:-4]+f'_{i}.json')
        #             with open(file_name_json) as f:
        #                 full_anno = json.load(f,object_pairs_hook=OrderedDict)
        #             f.close()
        #         except FileNotFoundError:
        #             continue
    
        img = np.load(
            os.path.join(self.data_set_path, 'images', self.img_files[idx]))
        
        layer_map = np.load(
            os.path.join(self.data_set_path, 'sdm',
                        self.img_files[idx][:-4] + '_sdm.npy'))
        mask = np.load(
            os.path.join(self.data_set_path, 'sdm',
                        self.img_files[idx][:-4] + '_sdm_mask.npy'))
        layer = np.load(
        os.path.join(self.data_set_path, 'labels',
                        self.img_files[idx][:-4] + '_label.npy'))
        

        img = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        layer_map = torch.from_numpy(layer_map.copy()).type(torch.FloatTensor)
        mask = torch.from_numpy(mask.copy()).type(torch.FloatTensor).repeat(3,1,1)
        layer = torch.from_numpy(layer.copy()).type(torch.FloatTensor)



        # Use the 512x512 image as input
        if self.cfg.train.use_half_size_img:
            img = img[:,:,self.trim_offset:-self.trim_offset]
            layer_map = layer_map[:,:,self.trim_offset:-self.trim_offset]
            mask = mask[:,:,self.trim_offset:-self.trim_offset]
            
            layer = layer[:,self.trim_offset:-self.trim_offset,:]
            out_dict=self.resize({'image': img, 'layer_map': layer_map,'mask':mask})
            img,layer_map,mask=out_dict['image'],out_dict['layer_map'],torch.ceil(out_dict['mask'])
            layer= resize(layer,size=[512,3],interpolation=InterpolationMode.NEAREST)
        
        if self.cfg.validation_time_artifact:
            try:
                print("Prediction with Synthetic Noise!!")
                img, _, _ = self.artifacts(img.numpy(),np.expand_dims(layer.numpy().T,axis=0)*512)
                img = torch.from_numpy(img.copy()).type(torch.FloatTensor) 
            except:
                print("Error in Artifact")
                pass

        if self.mode == DataloaderMode.train:
            img = self.transformation(img)
            # if self.cfg.train.affine_transform:
            #     out_dict=self.affineTransform({'image': img, 'layer': layer_map,'mask':mask})
            #     img,layer_map,mask=out_dict['image'],out_dict['layer'],out_dict['mask']
            
     
            
        return img, layer_map, mask, layer,idx,self.img_files[idx][:-4]
    

class DukeOCTLayerDatasetDirect(Dataset):

    def __init__(self,
                 cfg: DictConfig = None,
                 mode: DataloaderMode = DataloaderMode.TBD) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.DEVICE = cfg.device

        self.trim_offset = 150
        
        self.artifacts=ArtificialArtifact(prob=1)

        if cfg.train.use_all_data:
            self.path = cfg.data.data_root_dir_all
            print(f"Using all data in {self.path}")
        else:
            self.path = cfg.data.data_root_dir
            print(f"Using selected data in {self.path}")
            
        self.imgs, self.layer_maps = [], []

        self.transformation = Compose([
            NormalizeIntensity(nonzero=False, channel_wise=True),
            RandGaussianNoise(prob=0.10, std=0.15),
            RandAdjustContrast(gamma=(.75, 1.5), prob=0.20),
            RandShiftIntensity(offsets=(0.05, 0.15)),
            RandGaussianSmooth()
        ])

        self.affineTransform = Compose([RandAffined(keys=['image', 'layer_map','mask'],
            rotate_range=(-0.5, 0.5),
            shear_range=(-0.001, 0.001),
            translate_range=(-40,40),
            mode=['area','bilinear','nearest'],
            padding_mode="nearest",
            prob=0.2)   
            ])
        
        self.resize=Compose([Resized(keys=['image','layer_map','mask'],
                                     spatial_size=(512,512),
                                     mode=('area','bilinear','nearest'))])
        
        if mode == DataloaderMode.train:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'train')
        elif mode == DataloaderMode.validation:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'val')
        elif mode == DataloaderMode.test:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'test')
        else:
            raise ValueError("Invalid dataset mode!")

        self.img_files = os.listdir(os.path.join(self.data_set_path,
                                                 'images/'))

        if self.cfg.debug_mode == 1:
            self.img_files = self.img_files[:150]


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img = np.load(
            os.path.join(self.data_set_path, 'images', self.img_files[idx]))

        layer = np.load(
        os.path.join(self.data_set_path, 'labels',
                        self.img_files[idx][:-4] + '_label.npy'))
        
        mask=np.load(os.path.join(self.data_set_path,'labels',
                                  self.img_files[idx][:-4]+'_mask.npy'))

        img =   torch.from_numpy(img.copy()).type(torch.FloatTensor)
        layer = torch.from_numpy(layer.copy()).type(torch.FloatTensor)
        mask =   torch.from_numpy(mask.copy()).type(torch.FloatTensor)
        # Use the 512x512 image as input
        if self.cfg.train.use_half_size_img:
            img   = img[:,:,self.trim_offset:-self.trim_offset]
            layer = layer[:,self.trim_offset:-self.trim_offset,:]
            mask  = mask[:,self.trim_offset:-self.trim_offset,:]
            
            img   = resize(img,size=[512,512],interpolation=InterpolationMode.BICUBIC)
            ## TODO: Need to verify if this is working properly!
            layer = resize(layer,size=[512,3],interpolation=InterpolationMode.NEAREST).permute(2,0,1)
            mask  = resize(mask,size=[512,3],interpolation=InterpolationMode.NEAREST).permute(2,0,1)
        if self.cfg.validation_time_artifact:
            try:
                print("Prediction with Synthetic Noise!!")
                img, _, _ = self.artifacts(img.numpy(),np.expand_dims(layer.numpy().T,axis=0)*512)
                img = torch.from_numpy(img.copy()).type(torch.FloatTensor) 
            except:
                print("Error in Artifact")
                pass

        if self.mode == DataloaderMode.train:
            img = self.transformation(img)
            # if self.cfg.train.affine_transform:
            #     out_dict=self.affineTransform({'image': img, 'layer': layer_map,'mask':mask})
            #     img,layer_map,mask=out_dict['image'],out_dict['layer'],out_dict['mask']
            
     
        layer_map=torch.nan

        return img, layer_map, mask, layer,idx,self.img_files[idx][:-4]


class DukeOCTLayerDataset(Dataset):

    def __init__(self,
                 cfg: DictConfig = None,
                 mode: DataloaderMode = DataloaderMode.TBD,sdm:bool= False) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.DEVICE = cfg.device
        self.path = cfg.data.data_root_dir
        self.sdm = cfg.sdm
        self.imgs, self.layer_maps = [], []

        # self.artifact = ArtificialArtifact(prob=self.cfg.train.artifact_prob)
        self.artifact= None

        self.transformation = Compose([
            RandGaussianNoise(prob=0.10, std=0.2),
            RandAdjustContrast(gamma=(.65, 2), prob=0.10),
            RandShiftIntensity(offsets=(0.05, 0.10)),
            RandGaussianSmooth()
        ])

        if mode == DataloaderMode.train:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'train')
        elif mode == DataloaderMode.validation:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'val')
        elif mode == DataloaderMode.test:
            self.data_set_path = os.path.join(cfg.data.data_root_dir, 'test')
        else:
            raise ValueError("Invalid dataset mode!")

        self.img_files = os.listdir(os.path.join(self.data_set_path,
                                                 'images/'))

        if self.cfg.debug_mode == 1:
            self.img_files = self.img_files[:150]

        # self.weight, self.label, _ = generate_sample_weights(
        #     root_dir=self.data_set_path,
        #     weight_prop=[1 - self.cfg.ped_percentage, self.cfg.ped_percentage])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img = np.load(
            os.path.join(self.data_set_path, 'images', self.img_files[idx]))
        
        if self.sdm:
            layer_map = np.load(
                os.path.join(self.data_set_path, 'sdm',
                            self.img_files[idx][:-4] + '_sdm.npy'))
            mask = np.load(
                os.path.join(self.data_set_path, 'sdm',
                            self.img_files[idx][:-4] + '_sdm_mask.npy'))
            layer = np.load(
            os.path.join(self.data_set_path, 'labels',
                         self.img_files[idx][:-4] + '_label.npy'))

        else:
            layer_map = np.load(
            os.path.join(self.data_set_path, 'labels',
                         self.img_files[idx][:-4] + '_label.npy'))
            mask = np.load(
            os.path.join(self.data_set_path, 'labels',
                         self.img_files[idx][:-4] + '_mask.npy'))

        
        if self.cfg.artifact and self.mode == DataloaderMode.train:
            # Apply artifacts
            img, layer_map, gt_variance = self.artifact(img, layer_map)
            gt_variance = torch.from_numpy(gt_variance.copy()).type(
                torch.FloatTensor).permute(2, 0, 1)

        if self.cfg.validation_time_artifact:
            # Apply artifacts
            img, layer_map, gt_variance = self.artifact(img, layer_map)
            gt_variance = torch.from_numpy(gt_variance.copy()).type(
                torch.FloatTensor).permute(2, 0, 1)

        img = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        

        if self.mode == DataloaderMode.train:
            img = self.transformation(img)
            mask = mask.squeeze(0)

        # Dont Move it from here
        if self.sdm:
            layer_map = torch.from_numpy(layer_map.copy()).type(
            torch.FloatTensor)
            mask = torch.from_numpy(mask.copy()).type(torch.FloatTensor).repeat(3,1,1)
            layer = torch.from_numpy(layer.copy()).type(torch.FloatTensor)
        else:
            layer_map = torch.from_numpy(layer_map.copy()).type(
            torch.FloatTensor).permute(2, 0, 1)
            mask = torch.from_numpy(mask.copy()).type(torch.FloatTensor).permute(
                2, 0, 1)
        # Experiment with one Layer only
        if self.cfg.oneLayer_exp:
            layer_map, mask = layer_map[2], mask[2]
        if self.cfg.artifact and self.mode == DataloaderMode.train:
            return img, layer_map, (mask, gt_variance)
        elif self.cfg.hard_sample_mining and self.mode == DataloaderMode.train:
            return img, layer_map, mask, self.label[idx], idx
        elif self.cfg.track_oversampling and self.mode == DataloaderMode.train:
            return img, layer_map, mask, self.label[idx]
        elif self.sdm:
            return img, layer_map, mask, layer
        elif self.mode == DataloaderMode.train:
            return img, layer_map, mask, 0
        else:
            return img, layer_map, mask


def generate_sample_weights(root_dir='data/train/',
                            mean_thr: float = 13.0,
                            dev_thr: float = 20.0,
                            weight_prop: list = [.70, .30]):
    img_files = os.listdir(os.path.join(root_dir, 'images'))
    label = np.zeros(len(img_files), dtype=np.int32)

    means, maxs, dev = [], [], []
    for idx in range(len(img_files)):
        layer_map = np.load(
            os.path.join(root_dir, 'labels',
                         img_files[idx][:-4] + '_label.npy')).squeeze() * 512
        maxs.append(np.max(np.abs(layer_map[:, 1] - layer_map[:, 2])))
        means.append(np.mean(np.abs(layer_map[:, 1] - layer_map[:, 2])))
        dev.append(
            np.max(np.abs(layer_map[:, 1] - layer_map[:, 2])) -
            np.mean(np.abs(layer_map[:, 1] - layer_map[:, 2])))

    #Convert the list to ndarray's
    maxs = np.array(maxs)
    means = np.array(means)
    dev = np.array(dev)

    #  Apply the mean and deviation criteria
    flist = list(
        set(np.where(means > mean_thr)[0]).intersection(
            np.where(dev > dev_thr)[0]))

    # Set the PED cases to 1
    label[flist] = 1

    # Get the count of each categories
    count = np.unique(label, return_counts=True)[1]

    # calculate the weight per sample
    weight_factor = 1 / count

    # generate the weight vector
    weight = [
        weight_factor[0] *
        weight_prop[0] if label[l] == 0 else weight_factor[1] * weight_prop[1]
        for l in range(len(label))
    ]

    return weight, label, flist


class DukeOCTLayerDataset3d(Dataset):

    def __init__(self,
                 cfg: DictConfig = None,
                 mode: DataloaderMode = DataloaderMode.train) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.DEVICE = cfg.device
        self.path = cfg.data.data_root_dir_3d
        self.transformation = Compose([
            RandGaussianNoise(prob=0.10, std=0.2),
            RandAdjustContrast(gamma=(.65, 2), prob=0.10),
            RandShiftIntensity(offsets=(0.05, 0.10)),
            RandGaussianSmooth()
        ])
        

        self.rand_crop=Compose([ RandSpatialCropd(keys=['image', 'layer'], 
                                         roi_size=(32,512,1024),random_size=False)])
        
        # self.rand_crop.set_random_state(123)
        if mode == DataloaderMode.train:
            self.data_set_path = os.path.join(self.path, 'train')
        elif mode == DataloaderMode.validation:
            self.data_set_path = os.path.join(self.path, 'val')
        elif mode == DataloaderMode.test:
            self.data_set_path = os.path.join(self.path, 'test')
        else:
            raise ValueError("Invalid dataset mode!")
        # Get the list of files
        self.img_files = os.listdir(self.data_set_path)
        # less sample for debug
        if self.cfg.debug_mode == 1:
            self.img_files = self.img_files[:15]

    def load_mat_file(self, fname):
        matlab_file = io.loadmat(fname)
        images = matlab_file['images']
        layer_maps = matlab_file['layerMaps']
        return np.rollaxis(images, 2, 0) / 255, layer_maps

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img, layer_map = self.load_mat_file(
            os.path.join(self.data_set_path, self.img_files[idx]))
        # Convert to torch tensors
        img = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        layer_map = torch.nan_to_num(torch.from_numpy(layer_map.copy()).type(torch.FloatTensor)).permute(0,2, 1)
      
        img ,layer_map= img[20:-20,:,:],layer_map[20:-20,:,:]
        img,layer_map=F.pad(img,(12,12,0,0,0,0),'constant',0),F.pad(layer_map,(12,12,0,0,0,0),'constant',0)
      
        # Add channel dimension
        layer_map,img=layer_map.unsqueeze(0),img.unsqueeze(0)   
    
        # Apply random crop on x and z diection for training and validation
        # in case of test the random crop is applied only on z direction
        out_dict=self.rand_crop({'image': img, 'layer': layer_map})
        img,layer_map=out_dict['image'],out_dict['layer']
        
        # Create the mask for loss calculation
        mask = torch.ones_like(layer_map)
        mask[layer_map == 0] = 0
        # Apply transformations
        if self.mode == DataloaderMode.train:
            img = self.transformation(img)

        layer_map, mask = layer_map.permute(2,1,0,3),mask.permute(2,1,0,3)
        
        return img, layer_map/512, mask
    
@hydra.main(version_base="1.1", config_path="/home/mislam/retinal_layers_segmentation/retinal_layers_segmentation/config", config_name="default")
def main(hydra_cfg):
    ds = DukeOCTLayerDataset(cfg=hydra_cfg, mode=DataloaderMode.test,sdm=True)
    # for i in range(100):
    p=np.random.randint(0,15)
    img, layer_map, mask = ds.__getitem__(p)
 
    # layer_map[mask == 0] = torch.nan
    # layer_map = layer_map.permute(2,1,0,3).squeeze(0)
    # mask = mask.permute(2,1,0,3).squeeze(0)
    # print(mask.shape, layer_map.shape,img.shape)   
  

    # n=np.random.randint(0,15)
    # plt.imshow(img[n, :, :]  , cmap='gray')
    # # plt.plot(mask[n, 0, :].T)
    # plt.plot(layer_map[n, :, :].T*512)
    # plt.show()  
 


if __name__ == "__main__":
    main()
