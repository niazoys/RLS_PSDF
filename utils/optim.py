import sys

sys.path.append('../')



import torch

from model.net import *


def cnn_model_optim_scheduler(ARGS, DEVICE): 
    model = Encoder2D().to(DEVICE)
    optim = torch.optim.Adam(lr=ARGS.cnn_lr, params=model.parameters(), weight_decay=ARGS.cnn_wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, 
                                                           factor=.5, verbose=True, min_lr=ARGS.min_lr)
    
    return model, optim, scheduler


def mapping_model_optim_scheduler(ARGS, lr, DEVICE):
    model= MappingNetwork(ARGS).to(DEVICE)
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, factor=.5, 
                                                           verbose=True, min_lr=ARGS.min_lr)

    return model, optim, scheduler
    

def siren_model_optim_scheduler(ARGS, first_omega_0, hidden_omega_0, lr, wd, final_activation, DEVICE):
    model = Siren(ARGS, in_features=3, out_features=1,first_omega_0=first_omega_0, 
                            hidden_omega_0=hidden_omega_0, final_activation=final_activation).to(DEVICE)
    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=wd)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, factor=.5, 
                                                           verbose=True, min_lr=ARGS.min_lr)
    
    return model, optim, scheduler
