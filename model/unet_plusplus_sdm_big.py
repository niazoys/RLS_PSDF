import torch
import torch.nn as nn

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):

    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18],group=1,groupNorm=False):
        super().__init__()
    
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[0],
                      padding=rate[0],groups=group),
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[1],
                      padding=rate[1],groups=group), 
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))


        self.c3 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[2],
                      padding=rate[2],groups=group), 
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[3],
                      padding=rate[3],groups=group), 
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0,groups=group)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class CustomHead(nn.Module):

    def __init__(self, num_class=1,activation=False) -> None:
        super().__init__()
        self.activation=activation

        self.d5 = Decoder_Block([32, 128], 64)
        self.d6 = Decoder_Block([16, 64], 64)
        self.aspp = ASPP(64, 32)
        self.output =nn.Sequential(
            nn.Conv2d(32, 16,kernel_size=1, padding=0),
            nn.Conv2d(16, num_class,kernel_size=1, padding=0)
            )

    def forward(self, c1, c2, d4):
        d5 = self.d5(c2, d4)
        d6 = self.d6(c1, d5)
        out = self.aspp(d6)
        out = self.output(out)
        if self.activation:
            out=nn.functional.relu(out)
        return out




class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class Unet_sdm(nn.Module):
    def __init__(self,num_class=3,in_channels=1,
                        gaussian_output=True,
                        out_act=False,groupNorm=False,use_multi_head=False,use_input_instance_norm=True) -> None:
        super().__init__()

        self.use_multi_head = use_multi_head
        self.in_channels = in_channels
        self.use_input_instance_norm = use_input_instance_norm
        self.gaussian_output = gaussian_output
        self.out_act = out_act

        self.c1 = Stem_Block(in_channels, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)
        self.c5 = ResNet_Block(128, 256, stride=2)
        self.c6 = ResNet_Block(256, 256, stride=2)
        self.c7 = ResNet_Block(256, 512, stride=2)

        self.b1 = ASPP(512, 1024)

        self.d1 = Decoder_Block([256, 1024], 512)
        self.d2 = Decoder_Block([256, 512], 256)
        self.d3 = Decoder_Block([128, 256], 256)
        self.d4 = Decoder_Block([64, 256], 128)

        ###### Usual Decoder Block ###########
        # self.d5 = Decoder_Block([32, 128], 64)
        # self.d6 = Decoder_Block([16, 64], 64)
        # self.aspp = ASPP(64, 32)
        # self.output =nn.Sequential(nn.Conv2d(32, 16,kernel_size=1, padding=0), 
        #                            nn.Conv2d(16, num_class,kernel_size=1, padding=0))
        
        if self.use_multi_head:
            # ####### Multi Head Block ########
            self.l1=CustomHead(num_class=1,activation=self.out_act)
            self.l2=CustomHead(num_class=1,activation=self.out_act)
            self.l3=CustomHead(num_class=1,activation=self.out_act)
            if self.gaussian_output:
                self.log_var_1=CustomHead(num_class=1,activation=True)
                self.log_var_2=CustomHead(num_class=1,activation=True)
                self.log_var_3=CustomHead(num_class=1,activation=True)
        else:
            ####### DepthWise Convolution Block #######
            self.d5 = Decoder_Block([32, 128], 81)
            self.d6 = Decoder_Block([16, 81], 81)
            self.aspp = ASPP(81, 27,group=3,groupNorm=groupNorm)
            if groupNorm:
                self.output =nn.Sequential(nn.Conv2d(27, 9, groups=3,kernel_size=1, padding=0),
                                        nn.GroupNorm(num_groups=3,num_channels=9), 
                                        nn.Conv2d(9, num_class, groups=3,kernel_size=1, padding=0))
            else:
                self.output =nn.Sequential(nn.Conv2d(27, 9, groups=3,kernel_size=1, padding=0), 
                                    nn.Conv2d(9, num_class, groups=3,kernel_size=1, padding=0))
            



      

    def forward(self, inputs):
        if self.use_input_instance_norm:
            inputs=nn.InstanceNorm2d(self.in_channels)(inputs)
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)
        c7 = self.c7(c6)

        b1 = self.b1(c7)

        d1 = self.d1(c6, b1)
        d2 = self.d2(c5, d1)
        d3 = self.d3(c4, d2)
        d4 = self.d4(c3, d3)

        if self.use_multi_head:
            l1=self.l1(c1,c2,d4)
            l2=self.l2(c1,c2,d4)
            l3=self.l3(c1,c2,d4)
            if self.gaussian_output:
                log_var_1=self.log_var_1(c1,c2,d4)
                log_var_2=self.log_var_2(c1,c2,d4)
                log_var_3=self.log_var_3(c1,c2,d4)
                return (torch.cat([l1,l2,l3],dim=1),torch.cat([log_var_1,log_var_2,log_var_3],dim=1))
            else:
                return torch.cat([l1,l2,l3],dim=1)
        else:
            d5 = self.d5(c2, d4)
            d6 = self.d6(c1, d5)
            output = self.aspp(d6)
            output =self.output(output)
            return output




from torchinfo import summary

if __name__ == '__main__':
    model = Unet_sdm(use_multi_head=True)
    summary(model, (2, 1, 512, 512))
    # input= torch.rand((6,1,512,1024))
    # mean=model(input)
    # print(mean.shape)
