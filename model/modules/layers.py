from typing import Tuple
import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd

def conv3x3(
    in_channels: int,
    out_channels:int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    '''
    2D convolution for ResNet 3x3 block
    Parameters
    ----------
    in_channels: int
        Number of channels in
    out_channels: int
        Number of channels out
    stride: int
        stride
    groups: int
        Groups -> see torch.nn.Conv2D for explanation
    dilation: int
        dilation
    Returns
    -------
        nn.Conv2d instance
    '''
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1,
        dilation=dilation, groups=groups, stride=stride)

def deconv3x3(in_channels: int, out_channels:int, kernel_size: int = 3,
        stride: Tuple = (2, 2), groups: int = 1, padding: Tuple[int] = (1,1),
                upsample_padding: Tuple[int] = (1,1), dilation: int = 1) -> nn.ConvTranspose2d:
    '''
    components2D transposed convolutions in pytorch
    Expects an input with 4 dimensions--> [Batch, Channels, Height, Width]
    Parameters
    ----------
    in_channels:int
    out_channels: int
    kernel_size: int
        default 3, kernel size
    stride: Tuple[int]
        default 2
    groups: int
    padding:
        Tuple[int]
    upsample_padding
    dilation
    '''
    return nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size,  padding = padding, output_padding = upsample_padding,
                              stride = stride, groups = groups, dilation = dilation)

def conv1x1(in_channels: int, out_channels:int,
        stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    '''
    2D convolution for ResNet 3x3 kernel
    Parameters
    ----------
    in_channels
    out_channels
    stride
    groups
    dilation
    '''
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, padding=0,
        dilation=dilation, groups=groups, stride=stride)
    
class ConvLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, 
                 activation="relu", max_pool=None, layer_norm=None, batch_norm=False):
        
        super(ConvLayer3D, self).__init__()
        
        net = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding)]
        
        if activation == "relu":
            net.append(nn.ReLU())
        
        elif activation == "leakyrelu": 
            net.append(nn.LeakyReLU())
            
        if layer_norm: 
            net.append(nn.LayerNorm(layer_norm)) # add layer normalization
        
        if batch_norm: 
            net.append(nn.BatchNorm3d(out_channels)) # add batch normalization
        
        if max_pool: 
            net.append(nn.MaxPool3d(max_pool)) # add max_pooling
        
        self.model = nn.Sequential(*net)
        
    def forward(self, input):
        out = self.model(input)
        return out

class Conv2d_MeanVariance(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0 ,dilation=1):
        super(Conv2d_MeanVariance, self).__init__()
        
        self.mean_1    = nn.Conv2d(in_channels=in_channels,out_channels=int(in_channels/2),
                                 kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        
        self.log_var_1 = nn.Conv2d(in_channels=in_channels,out_channels=int(in_channels/2),
                                 kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        
        self.mean_2    = nn.Conv2d(in_channels=int(in_channels/2),out_channels=out_channels,
                                 kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        
        self.log_var_2 =nn.Conv2d(in_channels=int(in_channels/2),out_channels=out_channels,
                                 kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        
    def forward(self, inputs):
        outputs_mean_1 =self.mean_1(inputs) 
        outputs_variance_1 = self.log_var_1(inputs)
        
        outputs_mean=self.mean_2(outputs_mean_1)
        outputs_variance= self.log_var_2(outputs_variance_1)

        return outputs_mean, outputs_variance

class ConvLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, 
                 activation="relu", max_pool=None, layer_norm=None, batch_norm=False):
        
        super(ConvLayer3D, self).__init__()
        
        net = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding)]
        
        if activation == "relu":
            net.append(nn.ReLU())
        
        elif activation == "leakyrelu": 
            net.append(nn.LeakyReLU())
            
        if layer_norm: 
            net.append(nn.LayerNorm(layer_norm)) # add layer normalization
        
        if batch_norm: 
            net.append(nn.BatchNorm3d(out_channels)) # add batch normalization
        
        if max_pool: 
            net.append(nn.MaxPool3d(max_pool)) # add max_pooling
        
        self.model = nn.Sequential(*net)
        
    def forward(self, input):
        out = self.model(input)
        return out

class ConvLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, 
                 activation="relu", max_pool=None, layer_norm=None, batch_norm=False):
        
        super(ConvLayer2D, self).__init__()
        
        net = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding)]
        
        if activation == "relu":
            net.append(nn.ReLU())
        
        elif activation == "leakyrelu": 
            net.append(nn.LeakyReLU())
            
        if layer_norm: 
            net.append(nn.LayerNorm(layer_norm)) # add layer normalization
        
        if batch_norm: 
            net.append(nn.BatchNorm2d(out_channels)) # add batch normalization
        
        if max_pool: 
            net.append(nn.MaxPool2d(max_pool)) # add max_pooling
        
        self.model = nn.Sequential(*net)
        
    def forward(self, input):
        out = self.model(input.to(torch.float32))
        return out
    
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, gamma, beta):
        out = self.linear(input)
        
        out = self.omega_0 * out        
        
        out = out.permute(1, 0, 2)
        out = gamma * out + beta
        out = out.permute(1, 0, 2)
        
        out = torch.sin(out)
        
        return out
