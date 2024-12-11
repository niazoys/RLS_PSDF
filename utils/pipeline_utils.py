import os
import warnings
import torch
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def SoftsdmToLayer(map, level=0, sigma=7, beta=10.0):
    """
    Convert a soft SDM (Spatial Distribution Map) to a layer representation.

    Args:
        map (ndarray): The soft SDM map.
        level (float, optional): The level at which to extract the layer. Defaults to 0.
        sigma (float, optional): The standard deviation of the Gaussian mask. Defaults to 5.
        beta (float, optional): The beta parameter for the softmax function. Defaults to 10.0.

    Returns:
        ndarray: The layer representation.

    """
    # Define the softmax function
    def softmax(x, beta):
        e_x = np.exp(beta * (x - np.max(x, axis=0, keepdims=True)))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
    # Apply the Gaussian mask
    mask = np.exp(-0.5 * (map - level)**2 / sigma**2)
    # Compute the soft argmax
    def soft_argmax(mask, beta):
        weights = softmax(mask, beta)
        indices = np.arange(mask.shape[0])
        return np.tensordot(weights, indices, axes=([0],[0]))
    
    # Apply soft argmax along the last dimension of the mask
    layer = np.apply_along_axis(soft_argmax, 2, mask, beta)
    
    # Transpose the layer to match the desired output
    return layer.transpose(0, 2, 1)

def transform_full_map(layers):
    '''
        Expects the shape as (batch_size, 3, 1024)
    '''
    
    layers=layers.squeeze().cpu().numpy()
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

    return torch.tensor(lmap).unsqueeze(0),torch.tensor(lmap_hot).permute(0,2,1,3).unsqueeze(0)

def reverse_transform_full_map(input_map):
    # Squeeze the singleton dimensions from the input tensor
    input_map = input_map.squeeze()
    
    # Initialize the output tensor with zeros
    output_tensor = torch.zeros((input_map.shape[0], 3, 1, input_map.shape[-1]))

    # Loop over the batch dimension and the spatial dimension of the input tensor
    for b in range(input_map.shape[0]):
        for i in range(input_map.shape[2]):
            # Get the non-zero indices along the second dimension
            non_zero_indices = torch.nonzero(input_map[b, :, i])
            # Copy the non-zero indices to the output tensor
            output_tensor[b, :len(non_zero_indices), 0, i] = non_zero_indices.squeeze()

    return output_tensor

def reverse_transform_full_map_test(lmap):

    lmap=lmap.squeeze()
    print(lmap.shape)
    layerR=torch.zeros((lmap.shape[0],3,1024))
    for b in range(lmap.shape[0]):
        for i in range(lmap.shape[2]):
            level=0
            for j in range(lmap.shape[1]):
                if lmap[b,j,i]!=0:
                    layerR[b,level,i]=j
                    level+=1
    return layerR

def plot_graph(path,
               x,
               ys_and_labels,
               axes=("Epochs", "BCELoss"),
               fig_name="loss_plot"):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    for y, label in ys_and_labels:
        ax.plot(x[1:], y[1:], label=label)

    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    _ = ax.legend(loc='upper right')

    plt.savefig(f"{path}/{fig_name}.png")
    plt.close()

def get_folder(ARGS):

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if ARGS.name != "":
        ARGS.name = f"_{ARGS.name}"

    path = os.path.join("output", f"pi_gan_{now}{ARGS.name}")

    # Path(f"{path}").mkdir(parents=True, exist_ok=True)

    try:
        os.mkdir(path)
        print(f'Path creation successful at: {path}')
    except OSError as error:
        print(error)

    ARGS.path = path

    return path

def initialize_path_and_device(ARGS):
    warnings.filterwarnings("ignore")

    ##### path to wich the model should be saved #####
    path = get_folder(ARGS)

    if ARGS.device.lower() == "cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('----------------------------------')
    print('Using device for training:', DEVICE)
    print('----------------------------------')

    # with open(os.path.join(path, "ARGS.txt"), "w") as f:
    #     print(vars(ARGS), file=f)

    return path, DEVICE

def save_loss(path, loss, models, optims, name="loss", save_models=True):
    np.save(f"{path}/{name}.npy", loss)

    eps, t_loss, v_loss = loss[:, 0], loss[:, 1], loss[:, 3]

    print(
        f"{name.ljust(15)} Train: {str(round(t_loss[-1], 6)).ljust(8, '0')}, \t Eval: {str(round(v_loss[-1], 6)).ljust(8, '0')}"
    )

    if "imgs" in name:
        models_to_be_saved = ["cnn", "imgs_mapping", "imgs_siren"]
    else:
        models_to_be_saved = ["cnn", "mapping", "siren"]

    if save_models:
        if t_loss.shape[0] == 1 or t_loss[-1] < t_loss[:-1].min():
            print("New best train loss, saving model.")
            if save_models:
                for model in models_to_be_saved:
                    torch.save(models[model].state_dict(),
                               f"{path}/{model}_{name}_train.pt")
                    torch.save(optims[model].state_dict(),
                               f"{path}/{model}_optim_{name}_train.pt")

        if v_loss.shape[0] == 1 or v_loss[-1] < v_loss[:-1].min():
            print("New best eval  loss, saving model.")
            if save_models:
                for model in models_to_be_saved:
                    torch.save(models[model].state_dict(),
                               f"{path}/{model}_{name}_val.pt")
                    torch.save(optims[model].state_dict(),
                               f"{path}/{model}_optim_{name}_val.pt")

    plot_graph(path,
               eps, [(t_loss, "Train loss"), (v_loss, "Eval loss")],
               axes=("Epochs", "Loss"),
               fig_name=f"{name}_plot")

def load_from_saved_run(models, optims, DEVICE, ARGS):
    if ARGS.pretrained:
        print(f"Loading pretrained model from '{ARGS.pretrained}'.")
        load_pretrained_models(ARGS.pretrained,
                               ARGS.pretrained_best_dataset,
                               ARGS.pretrained_best_loss,
                               models,
                               optims,
                               DEVICE,
                               pretrained_models=ARGS.pretrained_models)

        if ARGS.pretrained_lr_reset:
            for name, optim in optims.items():
                for param_group in optim.param_groups:
                    param_group["lr"] = ARGS.pretrained_lr_reset
                print(f"{name} lr reset to: {optim.param_groups[0]['lr']}")

def load_pretrained_models(folder,
                           best_dataset,
                           best_loss,
                           models,
                           optims,
                           DEVICE,
                           pretrained_models=None):
    path = f"saved_runs/{folder}/"

    for key in models.keys():
        if pretrained_models == None or key in pretrained_models:
            if os.path.exists(
                    f"{path}/{key}_{best_loss}_loss_{best_dataset}.pt"):
                print(f"Loading params from {key}")
                models[key].load_state_dict(
                    torch.load(
                        f"{path}/{key}_{best_loss}_loss_{best_dataset}.pt",
                        map_location=torch.device(DEVICE)))
                optims[key].load_state_dict(
                    torch.load(
                        f"{path}/{key}_optim_{best_loss}_loss_{best_dataset}.pt",
                        map_location=torch.device(DEVICE)))

def get_layer_wise_exp( y_true,y_pred):
    '''
    this is just for the single layer experiment
    '''
    layers = 'ILM', 'RPEDC', 'BM'
    
    y_pred=y_pred.squeeze(1)

    result = {}
    y_pred = 512 * y_pred[:,:,12:-12]
    y_true = 512 * y_true[:,:,12:-12]
    
    y_true[y_true == 0] = np.nan
    dif = (y_pred - y_true)
    abs_dif = abs(dif)

    mad_all=[]
    for i, layer in enumerate(layers):
        abs_dif_i = abs_dif

        # dif_i = dif[:,:,i]
        # result[layer] = np.mean(abs_dif_i[np.where(~np.isnan(abs_dif_i))]), np.mean(dif_i[np.where(~np.isnan(dif_i))])
        mad_layer = np.mean(abs_dif_i[np.where(~np.isnan(abs_dif_i))])
        
        mad_all.append(mad_layer)
        
        result[layer]=mad_layer
    
    return result,np.mean(np.array(mad_all))

def get_layer_wise_( y_true,y_pred):
    '''
    Layers Wise MAE and mean of all Layers MAE
    '''
    layers = 'ILM', 'RPEDC', 'BM'
    
    result = {}
    
    if y_pred.shape[-1]==1024:
        y_pred = y_pred[:,:,:,12:-12]
        y_true = y_true[:,:,:,12:-12]

    y_pred = 512 * y_pred   
    y_true = 512 * y_true
    
    y_true[y_true == 0] = np.nan
    dif = (y_pred - y_true)
    abs_dif = abs(dif)

    sample_mad_layer=np.mean(np.nan_to_num(abs_dif),axis=(-2,-1))

    mad_all=[]
    for i, layer in enumerate(layers):
        abs_dif_i = abs_dif[:,i,:,:]

        mad_layer = np.mean(abs_dif_i[np.where(~np.isnan(abs_dif_i))])
        
        mad_all.append(mad_layer)
        
        result[layer]=mad_layer
    
    return result,np.mean(np.array(mad_all)),sample_mad_layer

def get_layer_wise_sdm(y_true,y_pred):
    '''
    Layers Wise MAE and mean of all Layers MAE
    '''
    layers = 'ILM', 'RPEDC', 'BM'
    
    result = {}
    if y_pred.shape[-1]==1024:
        y_pred = y_pred[:,:,:,12:-12]
        y_true = 512 * y_true[:,:,:,12:-12]
    else:
        y_true = 512 * y_true
    
    y_true[y_true == 0] = np.nan
    dif = (y_pred - y_true)
    abs_dif = abs(dif)

    
    sample_mad_layer=np.mean(np.nan_to_num(abs_dif),axis=(-1,-3))

    mad_all=[]

    for i, layer in enumerate(layers):
        abs_dif_i = abs_dif[:,:,i,:]

        mad_layer = np.mean(abs_dif_i[np.where(~np.isnan(abs_dif_i))])
        
        mad_all.append(mad_layer)
        
        result[layer]=mad_layer
    
    return result,np.mean(np.array(mad_all)),sample_mad_layer

def get_layer_wise_3d( y_true,y_pred):
    '''
    Layers Wise MAE and mean of all Layers MAE
    '''
    layers = 'ILM', 'RPEDC', 'BM'
    
    result = {}
    y_pred = 512 * y_pred
    y_true = 512 * y_true
    
    y_true[y_true == 0] = np.nan
    dif = (y_pred - y_true)
    abs_dif = abs(dif)

    mad_all=[]
    for i, layer in enumerate(layers):
        abs_dif_i = abs_dif[:,i,:,:,:]

        # dif_i = dif[:,:,i]
        # result[layer] = np.mean(abs_dif_i[np.where(~np.isnan(abs_dif_i))]), np.mean(dif_i[np.where(~np.isnan(dif_i))])
        mad_layer = np.mean(abs_dif_i[np.where(~np.isnan(abs_dif_i))])
        
        mad_all.append(mad_layer)
        
        result[layer]=mad_layer
    
    return result,np.mean(np.array(mad_all))

def show_im(x,y_pred,y_true,sigmas=None,model_var=None,outfile=None):
    fig=plt.figure(figsize=(20,20))
    fig.add_subplot(131)
    plt.imshow(x, cmap='gray')

    colors = ['#62c11d', 'tab:orange', 'tab:blue']
    fig.add_subplot(132)
    plt.imshow(x, cmap='gray')
    if y_true is not None:
        for color, line in zip(colors, y_true.T):
            plt.plot(line, linewidth=1.5, c=color) #ground truth
    
    fig.add_subplot(133)
    plt.imshow(x, cmap='gray')
    
    if isinstance(sigmas, np.ndarray) and model_var is not None:
        for color, line,sigma,mv in zip(colors, y_pred.T, sigmas.T,model_var.T):
            plt.plot(line, linewidth=1.5, c='#FFFFFF') #ground truth
            plt.plot(line+sigma,linewidth=1, c=color)
            plt.plot(line-sigma,linewidth=1, c=color)

            plt.plot(line+mv,linewidth=1, c='#FF0000')
            plt.plot(line-mv,linewidth=1, c='#FF0000')
    
    elif isinstance(sigmas, np.ndarray):
        for color, line,sigma in zip(colors, y_pred.T, sigmas.T):
            plt.plot(line, linewidth=1.5, c='#FFFFFF') #ground truth
            plt.plot(line+sigma,linewidth=1, c=color)
            plt.plot(line-sigma,linewidth=1, c=color)
    else:
        for color, line in zip(colors, y_pred.T):
            plt.plot(line, linewidth=1.5, c=color) #ground truth
    
    img=get_img_from_fig(fig)
    
    plt.close(fig)

    return img

