import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch.nn as nn 
import torch 
import torch.nn.functional as F
from pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule, PointnetFPModule, FPS, FPS2, group, three_nn, three_interpolate, PointnetSAModule_test


class basic_conv1d_seq(nn.Module):
    def __init__(self, channels, BNDP=True):
        super(basic_conv1d_seq, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(channels)-2):
            self.net.add_module('conv1d_%d' % i, nn.Conv1d(channels[i], channels[i+1], 1))
            if BNDP:
                self.net.add_module('bn1d_%d' % i, nn.BatchNorm1d(channels[i+1]))
            self.net.add_module('relu_%d' % i, nn.ReLU())
            if BNDP:
                self.net.add_module('drop_%d' % i, nn.Dropout(0.5))
        self.net.add_module('conv1d_%d' % (len(channels)-2), nn.Conv1d(channels[-2], channels[-1], 1))
    
    def forward(self, x):
        return self.net(x)


class pointnet2_encoder(nn.Module):
    def __init__(self, input_channel_num=3, gf_channel_num=1024):
        '''
        input_channel_num: 3+C
        '''
        super(pointnet2_encoder, self).__init__()
        c_in = input_channel_num
        self.sa1 = PointnetSAModuleMSG(512, [0.1, 0.2, 0.4], [32, 64, 128], [[c_in, 32, 32, 64], [c_in, 64, 64, 128], [c_in, 64, 96, 128]])
        c_in = 128+128+64
        self.sa2 = PointnetSAModuleMSG(128, [0.4,0.8], [64, 128], [[c_in, 128, 128, 256], [c_in, 128, 196, 256]])
        c_in = 256+256
        self.sa3 = PointnetSAModule([c_in, 256, 512, gf_channel_num], npoint=None, radius=None, nsample=None)

    def forward(self, xyz, feature=None):
        '''
        xyz:        [B, N, 3]
        feature:    [B, C, N]
        '''
        # Set Abstraction layers
        if feature is not None:
            l0_points = torch.cat([xyz.permute(0, 2, 1), feature], 1).contiguous()
        else:
            l0_points = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz.contiguous()        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points
