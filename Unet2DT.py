#!/usr/bin/env python3
# encoding: utf-8



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



'''
    Basic Block     
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class Confidence(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(Confidence, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.FL = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 1),
        )
    def forward(self, x):
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.FL(x)



        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
    # x_0 guide x_1
    def forward(self, x_0, x_1):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # from x get the x0 and x1, x0 is the 0 sign with low entropy, x1 is the 1 sign with high entropy.
        # print("x.shape, zero_low.shape:", x.shape, torch.tensor(zero_low).cuda(async=True).shape)
        # x_0 = torch.index_select(x, 0, torch.tensor(zero_low).cuda(async=True))     # good cases
        # x_1 = torch.index_select(x, 0, torch.tensor(one_high).cuda(async=True))     # error cases
        batch_0, C_0, width_0, height_0 = x_0.size()
        batch_1, C_1, width_1, height_1 = x_1.size()
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape)
        proj_query = self.query_conv(x_0).view(batch_0, -1).permute(1, 0)     # good cases---> N_number*(512*8*8)---> (512*8*8)*N_number

        # print("x_0.shape:", x_0.shape)
        # print("proj_query.shape:", proj_query.shape)
        proj_key = self.key_conv(x_1).view(batch_1, -1)  # error cases---> K_number*(512*8*8)
        # print("x_1.shape:", x_1.shape)
        # print("proj_key.shape:", proj_key.shape)
        energy = torch.mm(proj_key, proj_query)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        attention = self.softmax(energy)  # the shape are K_number * N_number
        proj_value = self.value_conv(x_0).view(batch_0, -1)  # good cases, the two conv process, output a N_number*(512*8*8)
        # print("proj_value.shape:", proj_value.shape)

        out = torch.mm(attention, proj_value)     # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out.view(batch_1, C_1, width_1, height_1)     # output the shape is (K_number * 512 * 8 * 8)

        out = self.gamma * out + x_1     # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out, attention

class UNet2DT(nn.Module):
    """
    2d unet
    Ref:
        3D MRI brain tumor segmentation.
    Args:
        input_shape: tuple, (height, width, depth)
    """


    def __init__(self, in_channels=3, out_channels=4, init_channels=16, p=0.2):
        super(UNet2DT, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()


    def make_encoder(self):
        init_channels = self.init_channels

        #self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
        # self.pool_1 = nn.functional.interpolate(scale_factor=0.5, mode="bilinear")
        # self.pool_2 = nn.functional.interpolate(scale_factor=0.25, mode="bilinear")
        # self.pool_3 = nn.functional.interpolate(scale_factor=0.125, mode="bilinear")

        self.conv1a = nn.Conv2d(self.in_channels, init_channels, 3, padding=1)
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv2d(init_channels, init_channels * 2, 3, stride=2, padding=1)  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv2d(init_channels * 2, init_channels * 4, 3, stride=2, padding=1)

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv2d(init_channels * 4, init_channels * 8, 3, stride=2, padding=1)

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)

        self.ds4 = nn.Conv2d(init_channels * 8, init_channels * 16, 3, stride=2, padding=1)

        self.conv5a = BasicBlock(init_channels * 16, init_channels * 16)
        self.conv5b = BasicBlock(init_channels * 16, init_channels * 16)

        self.ds5 = nn.Conv2d(init_channels * 16, init_channels * 32, 3, stride=2, padding=1)

        self.conv6a = BasicBlock(init_channels * 32, init_channels * 32)
        self.conv6b = BasicBlock(init_channels * 32, init_channels * 32)

        self.conf = Confidence(init_channels * 16, 1)

        self.attention = Self_Attn(init_channels * 16)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up6conva = nn.Conv2d(init_channels * 32, init_channels * 16, 1)
        self.up6 = nn.ConvTranspose2d(init_channels * 16,init_channels * 16,2,stride=2)  # mode='bilinear'
        self.up6convb = BasicBlock(init_channels * 16, init_channels * 16)

        self.up5conva = nn.Conv2d(init_channels * 16, init_channels * 8, 1)
        self.up5 = nn.ConvTranspose2d(init_channels * 8,init_channels * 8,2,stride=2)  # mode='bilinear'
        self.up5convb = BasicBlock(init_channels * 8, init_channels * 8)

        self.up4conva = nn.Conv2d(init_channels * 8, init_channels * 4, 1)
        self.up4 = nn.ConvTranspose2d(init_channels * 4,init_channels * 4,2,stride=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv2d(init_channels * 4, init_channels * 2, 1)
        self.up3 = nn.ConvTranspose2d(init_channels * 2,init_channels * 2,2,stride=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv2d(init_channels * 2, init_channels, 1)
        self.up2 = nn.ConvTranspose2d(init_channels,init_channels,2,stride=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv2d(init_channels, self.out_channels, 1)


    def forward(self, x):
        # #print("input - x :", x.shape)
        # x_1 = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")
        # x_2 = nn.functional.interpolate(x, scale_factor=0.25, mode="bilinear")
        # x_3 = nn.functional.interpolate(x, scale_factor=0.125, mode="bilinear")
        # print("x.shape:", x.shape, y.shape)

        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)

        c1d = self.ds1(c1)
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)

        c4d = self.ds4(c4)
        c5 = self.conv5a(c4d)
        c5 = self.conv5b(c5)

        c5d = self.ds5(c5)
        c6 = self.conv6a(c5d)
        c6d = self.conv6b(c6)

        u6 = self.up6conva(c6d)
        u6 = self.up6(u6)
        u6 = u6 + c5
        u6 = self.up6convb(u6)


        u5 = self.up5conva(u6)
        u5 = self.up5(u5)
        u5 = u5 + c4
        u5 = self.up5convb(u5)
        u4 = self.up4conva(u5)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)
        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)
        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)
        uout = self.up1conv(u2)
        uout = torch.softmax(uout,dim=1)

        # print("uout.shape:", uout.shape)
        return uout