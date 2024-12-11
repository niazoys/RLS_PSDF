from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from model.modules.layers import *
# from modules.layers import *


class Flatten(nn.Module):
    """ Reshapes a 4d matrix to a 2d matrix. """
    def forward(self, input):
        return input.view(input.size(0), -1)

class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 kernel_size = 3,
                 padding =1,
                 downsample: bool = False,
                 activation: Optional[Callable[..., nn.Module]] = nn.LeakyReLU(negative_slope=0.0,inplace=True),
                 **kwargs) -> None:
        '''
        components2D Residual blocks implementation
        Parameters
        ----------
        in_channels: int
            Channels in
        out_channels: int
            Channels out
        stride: int
            Stride
        downsample: bool
            Downsample the result
        activation: Optional[Callable[..., nn.Module]]
            Activation function, default nn.LeakyReLU
        '''
        super().__init__()

        self.activation = activation

        # Block one
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        # self.norm1 = nn.InstanceNorm2d(out_channels)
        self.bnorm1= nn.BatchNorm2d(out_channels)

        # Block two
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # self.norm2 = nn.InstanceNorm2d(out_channels)
        self.bnorm2= nn.BatchNorm2d(out_channels)
        
        # This is not present in the bart's original paper!
        self.se=ChannelSpatialSELayer(out_channels,reduction_ratio=1)
        
        # other parameters
        self.downsample = downsample

        # This particularly modified to match the bart's implemntation.
        self.downsample_func = nn.Conv2d(in_channels, out_channels* self.expansion, kernel_size=kernel_size, padding=padding, stride=stride)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        # out = self.norm1(out)
        out = self.bnorm1(out)
        
        out = self.activation(out)

        out = self.conv2(out)
        # out = self.norm2(out)
        out= self.bnorm2(out)

        

        if self.downsample is not None:
            residual = self.downsample_func(x)

        out += residual
        
        out=self.se(out)

        out = self.activation(out)

        return out

class DownsamplingBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 blocks: int,
                 downsample: bool = False,
                **kwargs) -> None:

        '''
        Block in downsampling path UNet Generator. Uses the following:
        ResNet Basic Block, Leaky Relu, and Instance Normalization.
        Parameters
        ----------
        in_channels: int
            in channels to the downsampling block
        out_channels:
            out channels in the block
        blocks: int
            number of layers in block
        downsample: bool
            default False
        kwargs
        '''
        super().__init__()

        entry_block = BasicBlock(
            in_channels,
            out_channels,
            activation = nn.LeakyReLU(inplace = True),
            downsample = downsample
        )

        self.no_blocks = blocks

        if self.no_blocks == 1:
            self.block_layers = nn.ModuleList([entry_block])

        else:
            block_list = [entry_block] + [BasicBlock(out_channels, out_channels) for x in range(blocks)]
            self.block_layers = nn.ModuleList(block_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.block_layers:
            x = layer(x)

        return 

class UpsamplingBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        upsample_stride: Tuple = (2, 2),
        upsample_padding=(1, 1)
    ) -> None:
        """
        Upsampling block that uses ResNet blocks. Component of U-Net
        Generator.
        Parameters
        ----------
        in_channels: int
            in channels to the downsampling block
        out_channels:
            out channels in the block
        blocks: int
            number of layers in block
        upsample_stride: Tuple
            default, (2,2)
        upsample_padding: Tuple
            default, (1,1)
        """
        super().__init__()

        # RESNET blocks
        self.block_layers = nn.ModuleList()
        in_channels = in_channels
        for i in range(blocks):
            self.block_layers.append(
                BasicBlock(
                    in_channels,
                    out_channels * 2,
                    downsample=True if i == 0 else False,
                    activation=nn.LeakyReLU(
                        inplace=True
                    )
                )
            )

            in_channels = out_channels * 2

        # transposed convolution
        self.upsampling_layer = deconv3x3(
            in_channels,
            out_channels,
            stride=upsample_stride,
            upsample_padding=upsample_padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.block_layers:
            x = layer(x)

        x = self.upsampling_layer(x)

        return x

class DownsamplingPath(nn.Module):

    def __init__(
        self,
        network_blocks: List[Tuple[int]],
        maxpool_kernel_size: List[int],
        maxpool_stride: List[int],
        in_channels: int,
        out_channels: int,
        **kwargs
        ) -> None:

        '''
        Parameters
        ----------
        network_blocks: List[Tuple[int]]
            Indicates the out_channels and the number of ResNet blocks
        maxpool_kernel_size: List[int]
            Size of the max pooling kernel
        maxpool_stride: List[int]
            Maxpool stride
        in_channels: int
            In channels of the input block
        out_channels: int
            out channels of the input block
        kwargs
        '''
        super().__init__()

        # number of blocks in downsampling path
        self.size_downsampling_path = len(network_blocks)

        # input size is [N, C, H, W]
        # input layer
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.input_block = DownsamplingBlock(out_channels, out_channels, 2)
        self.activation = nn.LeakyReLU(inplace =True)
        self.pool1 = nn.MaxPool2d(
            maxpool_kernel_size[0],
            stride= maxpool_stride[0]
        )

        # input block squeeze excite layers
        self.se_one = SqueezeExciteLayer(
                    out_channels,
                    reduction=1
                )
        self.se_two = SqueezeExciteLayer(
            out_channels,
            reduction=1
        )

        # create the operations for the downsampling path
        in_channels = out_channels
        self.downsampling_blocks = nn.ModuleList()

        # normalization and pooling
        self.norm_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()

        # squeeze excite blocks
        self.skip_connection_attention = nn.ModuleList()
        self.downsampling_attention = nn.ModuleList()

        for index, (out_channels, no_blocks) in enumerate(network_blocks):

            # create downsampling block
            block = DownsamplingBlock(in_channels, out_channels, no_blocks)

            # create max pooling op
            pooling_op = nn.MaxPool2d(
            maxpool_kernel_size[index+1],
            stride= maxpool_stride[index+1])

            in_channels = out_channels

            # add to module lists
            self.downsampling_blocks.append(block)
            self.norm_list.append(nn.InstanceNorm2d(out_channels))
            self.pool_list.append(pooling_op)

            # Squeeze excite layer modules
            self.skip_connection_attention.append(
                SqueezeExciteLayer(
                    out_channels,
                    reduction=1
                )
            )
            self.downsampling_attention.append(
                SqueezeExciteLayer(
                    out_channels,
                    reduction=1
                )
            )

        # create the bottleneck layer
        self.bottleneck_conv = conv3x3(out_channels, out_channels*2)
        self.bottleneck_norm = nn.InstanceNorm2d(out_channels*2)
        self.se_bottleneck = SqueezeExciteLayer(
            out_channels*2,
            reduction=1
        )

    def forward(self, x: Tensor) -> List[Tensor]:

        # input layer + input block
        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm1(x)

        x = self.input_block(x)
        x = self.activation(x)
        x = self.norm2(x)

        # squeeze excite before skip connection and downsampling path start
        x_i = self.se_one(x)
        x = self.se_two(x)

        # list of features for residual connections
        features = [x_i]
        x = self.pool1(x)

        for i, block in enumerate(self.downsampling_blocks):

            # run features through encoder blocks
            x = block(x)
            x = self.activation(x)
            x = self.norm_list[i](x)

            # attention modules
            x_j = self.skip_connection_attention[i](x)
            x = self.downsampling_attention[i](x)

            # add features to skip connections and max pool
            features.append(x_j)

            x = self.pool_list[i](x)

        # bottleneck
        x = self.bottleneck_conv(x)
        x = self.activation(x)
        x = self.bottleneck_norm(x)
        x = self.se_bottleneck(x)

        return features + [x]

class UpsamplingPath(nn.Module):

    def __init__(
        self,
        bottleneck_features: int,
        upsample_stride: List[int],
        upsample_padding: List[int],
        network_blocks: List[Tuple[int]],
        out_channels_path: int,
        **kwargs
    ) -> None:
        """
        Upsampling path for the U-Net Generator
        Parameters
        ----------
        bottleneck_features: int
            Number of channels in the UNet bottleneck
        upsample_stride: List[int]
            strides
        upsample_padding: List[int]
            padding
        network_blocks: List[Tuple[int]]
            List of tuples that define the network architecture
        """
        super().__init__()

        self.bottleneck_upconv = deconv3x3(
            bottleneck_features,
            int(bottleneck_features/2),
            stride = upsample_stride[0],
            upsample_padding= upsample_padding[0]
        )

        self.activation = nn.LeakyReLU(inplace = True)

        # create upsampling blocks and normalization layers
        self.upsampling_blocks = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        self.se_list = nn.ModuleList()
        for index, (out_channels, no_blocks) in enumerate(reversed(network_blocks)):

            block = UpsamplingBlock(
                out_channels*2,
                int(out_channels/2),
                no_blocks,
                upsample_stride = upsample_stride[index+1],
                upsample_padding = upsample_padding[index+1]
            )

            self.upsampling_blocks.append(block)
            self.norm_list.append(nn.InstanceNorm2d(int(out_channels/2)))
            self.se_list.append(
                SqueezeExciteLayer(
                    int(out_channels/2),
                    reduction=1
                )
            )

        # output block
        self.out_block = DownsamplingBlock(
            out_channels,
            out_channels_path,
            no_blocks
        )
        self.norm_out = nn.InstanceNorm2d(out_channels)

    def forward(self, x: Tensor, encoder_features: List[Tensor]) -> Tensor:

        # Upsample bottleneck
        x = self.bottleneck_upconv(x)
        x = self.activation(x)

        # reverse order of features
        encoder_features = encoder_features[::-1]
        for i, block in enumerate(self.upsampling_blocks):

            print(f'{str(i)}\t{str(x.shape)} \t {str(encoder_features[i].shape)}')
            cropped_feats = self.central_crop_2d(x, encoder_features[i])

            x = torch.cat([x, cropped_feats], dim=1)

            x = block(x)
            x = self.activation(x)
            x = self.norm_list[i](x)
            x = self.se_list[i](x)

        # create the output block
        print(f'{str(i)}\t{str(x.shape)} \t {str(encoder_features[-1].shape)}')
        cropped_feats = self.central_crop_2d(x, encoder_features[-1])
        x = torch.cat([x, cropped_feats], dim=1)

        x = self.out_block(x)
        x = self.activation(x)
        x = self.norm_out(x)


        return x

    def central_crop_2d(self, x: Tensor, featuremap: Tensor) -> Tensor:
        '''
        function to crop the feature map from the skip connection to the image dimension
        '''
        _, _,  X_H, X_W = x.shape
        _, _, F_H, F_W = featuremap.shape

        crop_top = torch.round(torch.as_tensor((X_H - F_H) // 2, dtype = torch.float)).int()
        crop_left = torch.round(torch.as_tensor((X_W - F_W) // 2, dtype = torch.float)).int()

        return featuremap[
                   :,
                   :,
                   crop_top: crop_top + X_H,
                   crop_left: crop_left + X_W
               ]

class SqueezeExciteLayer(nn.Module):

    def __init__(
            self,
            features: int,
            reduction: int = 2,
            activation: Any = nn.LeakyReLU(inplace=True)
    ) -> None:
        '''
        Squeeze-Excite module for 2D Images
        Parameters
        ----------
        features: int
            Number of input features
        reduction: int
            Factor by which the features are reduced after the first layer: default 2
            (halves number of feats)
        activation: nn.Module
            default, LeakyReLU
        '''
        super(SqueezeExciteLayer, self).__init__()

        # global average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feed forward layer
        self.fc = nn.Sequential(
            nn.Linear(features, features // reduction, bias=False),
            activation,
            nn.Linear(features // reduction, features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = nn.functional.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

class UpsamplingBart(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple=(1,2),
        padding= (0,0),
        padding_residual=(0,1)
    ) -> None:
 
        super().__init__()
  
        self.upSample=nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=kernel),
            nn.LeakyReLU(0.0, inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel,padding=padding,stride=1)
        )
        

        self.conv1x1= nn.Sequential(
            nn.LeakyReLU(0.0,inplace=True),
            nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels,kernel_size=1)
            )

        self.decoderResidual=BasicBlock(in_channels=out_channels,out_channels=out_channels,kernel_size=(1,3),padding=padding_residual)

    def concat(self, x1, x2):
        """Crop and concatenate two feature maps
        """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat((x1, x2), 1)

    def forward(self, z: Tensor , z_horizontal: Tensor) -> Tensor:
       out=self.upSample(z)
       out=self.concat(out,z_horizontal)
       out=self.conv1x1(out)
       self.decoderResidual(out)

       return out

class EncoderBart(nn.Module):

    NUM_RETURN_VALUES = 1

    def __init__(self, in_channels=1) -> None:
        super().__init__()

        self.activation = {}

        NUM_FILTERS = [32, 32, 32, 64, 64, 128, 128, 256, 256]

        NUM_FILTERS_B = [32, 32, 64, 64, 128, 128, 256, 256, 1024]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=NUM_FILTERS[0],
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(0.0),
            BasicBlock(in_channels=NUM_FILTERS[0],
                       out_channels=NUM_FILTERS[1],
                       stride=1,
                       downsample=False))

        # this like going down vertically like classical Unet
        self.downPathVertical = nn.Sequential(
            self.conv, *[nn.Sequential(nn.Conv2d(NUM_FILTERS[i],NUM_FILTERS[i+1],kernel_size=2,stride=2),
                BasicBlock(in_channels=NUM_FILTERS[i+1],
                           out_channels=NUM_FILTERS[i + 1],
                           stride=1,
                           downsample=False))
                for i in range(0,
                               len(NUM_FILTERS) - 1)
            ])
        #This is the bottom residual block
        self.downPathVerticalOut = nn.Sequential(nn.Conv2d(NUM_FILTERS[-1], 1024, kernel_size=1, stride=2),
            BasicBlock(1024,
                       1024,
                       stride=1,
                       kernel_size=1,
                       downsample=False,
                       padding=0), BasicBlock(1024,
                       1024,
                       stride=1,
                       kernel_size=1,
                       downsample=False,
                       padding=0))

        #Horizontal path of each level (list of nn.Sequential)
        self.pathHorizontal = nn.ModuleList()

        # Going horizontal in each level
        for idx in range(1, len(self.downPathVertical) + 1):
            self.pathHorizontal.append(
                nn.Sequential(
                    *[nn.Sequential(nn.Conv2d(NUM_FILTERS[i],NUM_FILTERS[i+1],kernel_size=(2,1),stride=(2,1)),
                        BasicBlock(in_channels=NUM_FILTERS[i+1],
                                   out_channels=NUM_FILTERS[i + 1],
                                   stride=(1, 1),
                                   downsample=False))
                                for i in range(idx - 1,len(NUM_FILTERS) - 1)
                    ],
                    nn.Conv2d(in_channels=NUM_FILTERS[-1],
                              out_channels=NUM_FILTERS_B[idx - 1],
                              kernel_size=(2, 1)))
                )

        # Register forward hook for all the layer for vertical path
        for i in range(len(self.downPathVertical)):
            self.downPathVertical[i].register_forward_hook(
                self.get_activation(f'bb{i}'))

    # Forward hooking function
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output

        return hook

    def forward(self, x: Tensor) -> Tensor:

        # collect the forward pass activation here
        outH = []

        outV = self.downPathVertical(x)

        # bottom 1*1 residual block
        outV = self.downPathVerticalOut(outV)

        # Get the activation for the side way paths
        for i, key in enumerate(self.activation):
            outH.append(self.pathHorizontal[i](self.activation[key]))

        return outV, outH

class DecoderBart(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.UpsampleModules = torch.nn.ModuleList()
        filter_list = [1024, 256, 256, 128, 128, 64, 64, 32, 32, 32]

        for i in range(len(filter_list) - 1):
            self.UpsampleModules.append(
                UpsamplingBart(in_channels=filter_list[i],
                               out_channels=filter_list[i + 1]))

    def forward(self, z: Tensor, z_horitontal: Tensor) -> Tensor:

        reversed_z_horizontal = z_horitontal[::-1]

        for i in range(0, len(reversed_z_horizontal)):
            out = self.UpsampleModules[i](z, reversed_z_horizontal[i])
            z = out

        return out

class Encoder3D(nn.Module):
    def __init__(self):

        super(Encoder3D, self).__init__()

        self.model = nn.Sequential(
            ConvLayer3D(1,
                        8,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer3D(8,
                        8,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu",
                        max_pool=(1, 2, 2),
                        layer_norm=(24, 128, 128)),
            ConvLayer3D(8,
                        16,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer3D(16,
                        16,
                        kernel_size=5,
                        stride=(1, 2, 2),
                        padding=2,
                        activation="relu",
                        layer_norm=(24, 32, 32)),
            ConvLayer3D(16,
                        32,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer3D(32,
                        32,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu",
                        layer_norm=(12, 16, 16)),
            ConvLayer3D(32,
                        64,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer3D(64,
                        64,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu",
                        layer_norm=(6, 8, 8)),
            ConvLayer3D(64,
                        128,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer3D(128,
                        128,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu"),
        )

    def forward(self, x):
        out = self.model(x)

        return out

class Encoder2D(nn.Module):
    def __init__(self):

        super(Encoder2D, self).__init__()

        self.model = nn.Sequential(
            ConvLayer2D(1,
                        8,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer2D(8,
                        8,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu",
                        max_pool=(2, 2),
                        layer_norm=(128, 128)),
            ConvLayer2D(8,
                        16,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer2D(16,
                        16,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu",
                        layer_norm=(32, 32)),
            ConvLayer2D(16,
                        32,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer2D(32,
                        32,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu",
                        layer_norm=(16, 16)),
            ConvLayer2D(32,
                        64,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer2D(64,
                        64,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu",
                        layer_norm=(8, 8)),
            ConvLayer2D(64,
                        128,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        activation="relu"),
            ConvLayer2D(128,
                        128,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        activation="relu"),
        )

    def forward(self, x):
        out = self.model(x)

        return out