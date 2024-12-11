import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import os
import scipy.io as io
from scipy import ndimage
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

def load(fname):
    matlab_file = io.loadmat(fname)
    images = matlab_file['images']
    layer_maps = matlab_file['layerMaps']
    return np.rollaxis(images, 2, 0) / 255, layer_maps

def extend_to_end(input):
    out,mask=np.zeros_like(input),np.zeros_like(input)
    for b in range(input.shape[0]):
        noi=input[b].nonzero()
        idx_min_all,idx_max_all=np.array(np.where(noi[1]==np.min(noi[1])))[0],np.array(np.where(noi[1]==np.max(noi[1])))[0]
        for layer in range(len(idx_min_all)):
            idx_max,idx_min=idx_max_all[layer],idx_min_all[layer]
            x_min,y_min=noi[0][idx_min],noi[1][idx_min]
            x_max,y_max=noi[0][idx_max],noi[1][idx_max]
            input[b][x_min,0:y_min]=layer+1
            input[b][x_max,y_max:input.shape[2]]=layer+1       
            mask[b][:,y_min:y_max]=1
            out[b]=input[b]
    return out,mask

def transform_full_map(layers,extend_line=False):
    '''
        layers : Expects the shape as (batch_size, 3, 1000)
        extend_line : for Duke dataset with for not having annotation for full Bscan 
    '''
    lmap=np.zeros((layers.shape[0],512,layers.shape[2]))
    lmap_hot=np.zeros((layers.shape[0],512,4,layers.shape[2]))
    lmap_hot[:,:,0,:]=1

    # Get the non-zero indices 
    nonzero_indices = layers.nonzero()
    
    # Fill the lmap and lmap_hot tensors
    lmap[nonzero_indices[0],np.array(512*layers[nonzero_indices[0],nonzero_indices[1],nonzero_indices[2]],dtype=int),
           nonzero_indices[2]] = nonzero_indices[1] + 1
    lmap_hot[nonzero_indices[0],np.array(512*layers[nonzero_indices[0],nonzero_indices[1],nonzero_indices[2]],dtype=int),nonzero_indices[1] + 1,
           nonzero_indices[2]] = 1
    
    if extend_line:
        lmap,mask=extend_to_end(lmap)
        return torch.tensor(lmap),torch.tensor(lmap_hot).permute(0,2,1,3).unsqueeze(0),torch.tensor(mask)
    else:
        return torch.tensor(lmap),torch.tensor(lmap_hot).permute(0,2,1,3).unsqueeze(0),torch.ones_like(torch.tensor(lmap))

def get_signed_distance_maps(layer):
    sdm=np.zeros((3,layer.shape[0],layer.shape[1]))
    for i in range(1,4):
        input=np.array(layer==i,dtype=np.uint8)
        input= sitk.GetImageFromArray(input) 
        image=sitk.SignedDanielssonDistanceMap(input)
        np_img=sitk.GetArrayFromImage(image)
        for c in range(np_img.shape[1]):
            col=np_img[:,c]
            y_idx=np.where(col==0)[0]
            np_img[0:y_idx[0],c]=-1*np_img[0:y_idx[0],c]
        print(np_img.shape)
        sdm[i-1]=np.int32(np_img)    
    return sdm

def sdmToLayer(map,level=0):
    layer=np.zeros((1024,3))
    for i in range(map.shape[1]):
        for j in range(map.shape[2]):
            for l in range(map.shape[0]):
                if map[l,i,j]==level:
                    layer[j,l]=i
    return layer

def fix_discrepancy(path):
    img_files=os.listdir(os.path.join(path,'images'))
    for idx in range(0,len(img_files)):
        # try:
        if os.path.isfile(os.path.join(path,'sdm',img_files[idx][:-4]+'_sdm.npy')):
            print('ok')
        else:
            os.remove(os.path.join(path,'images',img_files[idx]))
            print(f'{img_files[idx]} has been removed!')

        # except FileNotFoundError:   
        #    os.remove(os.path.join(path,'images',img_files[idx]))
        #    print(f'{img_files[idx]} has been removed!')

def generate_sdm_groundtruth(path):
    
    img_files=os.listdir(os.path.join(path,'images'))
    
    for idx in range(0,len(img_files)):
        print(f'Processed {idx} files out of {len(img_files) }')

        try:
            print(os.path.join(path,'labels',img_files[idx][:-4]+'_label.npy'))
            layers=np.load(os.path.join(path,'labels',img_files[idx][:-4]+'_label.npy'))
            if len(layers.nonzero()[1])==0:
                # os.remove(os.path.join(path,'labels',img_files[idx][:-4]+'_label.npy'))
                print(f'{img_files[idx]} has been removed!')
            else:
                lmap,_,mask=transform_full_map(layers.transpose(0,2,1),extend_line=True)
                 

                # plt.plot(layers.squeeze())
                # plt.ylim(512,0)
                # plt.savefig('layer.png')
                # plt.clf()   
                # for i in range(1,4):
                #     plt.imshow(lmap.squeeze()==i)
                #     plt.savefig(f'bsg_{i}.png')
                #     plt.clf()
                
                dmap=get_signed_distance_maps(lmap.squeeze())
    
                # print(dmap.shape)
                # for i in range(0,3):
                #     plt.imshow(dmap[i])
                #     plt.colorbar()
                #     plt.savefig(f'bsg_{i}.png')
                #     plt.clf()
                # layer=sdmToLayer(dmap,level=0)
               
                if not os.path.exists(os.path.join(path,'sdm')):
                    os.makedirs(os.path.join(path,'sdm'))
                    
                np.save(os.path.join(path,'sdm',img_files[idx][:-4]+'_sdm.npy'),dmap)
                np.save(os.path.join(path,'sdm',img_files[idx][:-4]+'_sdm_mask.npy'),mask)
                
        except FileNotFoundError:
            # os.remove(os.path.join(path,'images',img_files[idx]))
            print(f'{img_files[idx]} Image has been removed due to file error!')
        except IndexError:
            # os.remove(os.path.join(path,'images',img_files[idx]))
            print(f'{img_files[idx]} Image has been removed due to index error!')

        finally:
            pass
       
@hydra.main(version_base="1.1", config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    
    cfg=hydra_cfg

    for var in ['test_new']:
        path=os.path.join(cfg.data.data_root_dir,var)
        generate_sdm_groundtruth(path)
        fix_discrepancy(path)
        

if __name__=="__main__":
    main()    
   