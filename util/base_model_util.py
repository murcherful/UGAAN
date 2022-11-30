import os
import sys
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
basic shape:  [bs, feature_channel, point_num]
'''

class MlpConv(nn.Module):
    def __init__(self, input_channel, channels, activation_function=None):
        super(MlpConv, self).__init__()
        self.layer_num = len(channels)
        self.net = nn.Sequential()
        last_channel = input_channel
        for i, channel in enumerate(channels):   
            self.net.add_module('Conv1d_%d' % i, nn.Conv1d(last_channel, channel, kernel_size=1))
            if i != self.layer_num - 1:
                self.net.add_module('ReLU_%d' % i, nn.ReLU())
            last_channel = channel
        if activation_function != None:
            self.net.add_module('af', activation_function)

    def forward(self, x):
        return self.net(x)        


class PcnEncoder(nn.Module):
    def __init__(self, input_channel=3, out_c=1024):
        super().__init__()
        self.mlp_conv_1 = MlpConv(input_channel, [128, 256])
        self.mlp_conv_2 = MlpConv(512, [512, out_c])

    def forward(self, x):
        '''
        x : [B, N, 3]
        '''
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.mlp_conv_1(x)

        x_max = torch.max(x, 2, keepdim=True).values
        x_max = x_max.repeat(1, 1, N) 
        x = torch.cat([x, x_max], 1)
        
        x = self.mlp_conv_2(x)
        
        x_max = torch.max(x, 2, keepdim=True).values
        return x_max


class PcnDecoder(nn.Module): 
    def __init__(self, grid_size=4, has_coarse=False, num_coarse=None):
        super().__init__()
        self.grid_scale = 0.05
        self.grid_size = grid_size
        if has_coarse:
            self.num_coarse = num_coarse
        else:
            self.num_coarse = 1024
        self.num_fine = (self.grid_size**2) * self.num_coarse
        #self.mlp_conv_1 = MlpConv(1024, [1024, 1024, self.num_coarse*3])
        self.has_coarse = has_coarse
        if not self.has_coarse:
            coarse_lst = [1024, 1024, self.num_coarse*3]
            in_features = 1024
            decoder_lst = []
            for i in range(len(coarse_lst)):
                decoder_lst.append(nn.Linear(in_features, coarse_lst[i]))
                in_features = coarse_lst[i]
            self.mlp_1 = nn.Sequential(*decoder_lst)

        self.mlp_conv_2 = MlpConv(1024+3+2, [512, 512, 3])

    def forward(self, x, coarse=None):
        ### Decoder coarse
        
        if not self.has_coarse: 
            fd1 = self.mlp_1(x)
            coarse = fd1.view(-1, self.num_coarse, 3)

        ### Folding
        g1 = torch.linspace(self.grid_scale*(-1), self.grid_scale, self.grid_size).cuda()
        g2 = torch.linspace(self.grid_scale*(-1), self.grid_scale, self.grid_size).cuda()
        grid = torch.meshgrid(g1, g2)
        grid = torch.reshape(torch.stack(grid, dim=2), (1, -1, 2))
        grid_feat = grid.repeat([x.shape[0], self.num_coarse, 1])
        point_feat = coarse.unsqueeze(2).repeat([1, 1, self.grid_size**2, 1])
        point_feat = torch.reshape(point_feat, (-1, self.num_fine, 3))
        glob_feat = x.unsqueeze(1).repeat([1, self.num_fine, 1])

        feat = torch.cat([grid_feat, point_feat, glob_feat], dim=2)
        fine = self.mlp_conv_2(feat.permute(0, 2, 1))
        fine = fine.permute(0, 2, 1) + point_feat

        return coarse, fine
