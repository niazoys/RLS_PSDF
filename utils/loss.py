import torch
import torch.nn as nn
import numpy as np



class ClampL1Loss(nn.Module):
    def __init__(self,cfg) -> None:
        super(ClampL1Loss,self).__init__()
        self.delta=torch.tensor(cfg.sdm_misc.clamp_delta)
    def forward(self,pred,label,mask):
        loss=((torch.abs(torch.min(self.delta,torch.max(pred,-self.delta))-
                       torch.min(self.delta,torch.max(label,-self.delta)))*mask).sum((-1,-2,-3)))/mask.sum((-1,-2,-3))
        return  loss

class L1Loss(nn.Module):
    def __init__(self,cfg) -> None:
        super(ClampL1Loss,self).__init__()
    def forward(self,pred,label,mask):
        loss=((pred-label)*mask).sum((-1,-2,-3))/mask.sum((-1,-2,-3))
        return  loss
    
class GNLL(nn.Module):
    def __init__(self,cfg) -> None:
        super(GNLL,self).__init__()
        self.esp=torch.tensor(1e-8)
    def forward(self,pred,label,mask):
        mean,log_var=pred
        loss =( ( 0.5 *(( ((mean - label) ** 2) / log_var.exp()) + log_var))*mask).sum((-1,-2,-3))/mask.sum((-1,-2,-3))
        return loss

class Clamp_GNLL(nn.Module):
    def __init__(self, cfg) -> None:
        super(Clamp_GNLL, self).__init__()
        self.esp = torch.tensor(1e-8)
        self.clamp_min = -torch.tensor(cfg.sdm_misc.clamp_delta).to(cfg.device)
        self.clamp_max = torch.tensor(cfg.sdm_misc.clamp_delta).to(cfg.device)

    def forward(self, pred, label, mask):
        mean, log_var = pred
        var = torch.max(log_var.exp(), self.esp)
        clamped_mean = torch.clamp(mean, self.clamp_min, self.clamp_max)
        clamped_label = torch.clamp(label, self.clamp_min, self.clamp_max)
        loss = ((0.5 * (((clamped_mean - clamped_label) ** 2) / var + log_var)) * mask).sum((-1, -2, -3)) / mask.sum((-1, -2, -3))
        return loss


