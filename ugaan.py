import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    cuda_index = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append('./util')
from loss_util import *
from point_util import *
from base_model_util import *
import pointnet2_model_api_2 as PN2

class UGAAN_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.GPN = 512
        self.E = PN2.pointnet2_encoder(gf_channel_num=256)
        self.D = MlpConv(256, [512, 512, 1024, self.GPN*3])
        self.trans_RS = MlpConv(256, [128, 128, 9])
        self.trans_T = MlpConv(256, [128, 128, 3])
        self.PcnE = PcnEncoder()
        self.PcnD = PcnDecoder(grid_size=2, has_coarse=True, num_coarse=self.GPN)
    
    def forward(self, points):
        B, N, _ = points.shape
        gf = self.E(points)
        x = self.D(gf)
        x = x.reshape([B, self.GPN, 3])
        m_RS = self.trans_RS(gf).reshape([B, 3, 3])
        m_T = self.trans_T(gf).reshape([B, 1, 3])
        x = torch.matmul(x, m_RS)
        x = x + m_T
        gf2 = self.PcnE(x)[:,:,0]
        _, x2 = self.PcnD(gf2, x)
        return gf, x, x2


class UGAAN_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PcnEncoder()
        self.d_mlp = MlpConv(1024+256, [64, 64, 1])
    
    def forward(self, gf, points):
        feature = self.encoder(points)
        feature = torch.cat([feature, gf], 1)
        d = self.d_mlp(feature)
        d = torch.sigmoid(d)
        d = d[:,0,0]
        return d


class UGAAN(nn.Module):
    def __init__(self, dis=0.03):
        super().__init__()
        self.G = UGAAN_G()
        self.D = UGAAN_D()
        self.loss = UGAANLoss()
        self.loss_test = UGAANLoss_test(dis)
    
    def forward(self, data):
        ws_data, sn_data = data
        points, colors = ws_data[0]
        sn_points = sn_data[0]
        B, N, _ = points.shape
        _points = torch.cat([points, sn_points], 0)
        _gf, _x, _x2 = self.G(_points)
        _d = self.D(_gf, _x)
        d_fake, d_real = torch.split(_d, [B, B], 0)
        x_fake, x_real = torch.split(_x, [B, B], 0)
        x_fake_2, x_real_2 = torch.split(_x2, [B, B], 0)
        return  [d_fake, d_real, x_fake, x_real, x_fake_2, x_real_2, sn_points, points]


class UGAANLoss(BasicLoss):
    def __init__(self):
        super().__init__()
        self.loss_name = ['loss_g', 'loss_d', 'g_fake_loss', 'g_rsl', 'g_rsl_2', 'g_fsl', 'd_fake_loss', 'd_real_loss']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistance()
    
    def cd(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = p2g + g2p
        return cd, p2g, g2p
    
    def batch_forward(self, outputs, data):
        __E = 1e-8
        d_fake, d_real, x_fake, x_real, x_fake_2, x_real_2, sn_points, points = outputs

        g_fake_loss = -torch.log(d_fake+__E)
        g_rsl, _, _ = self.cd(x_real, sn_points)
        g_rsl_2, _, _ = self.cd(x_real_2, sn_points)
        _, g_fsl, _ = self.cd(x_fake, points)

        loss_g = g_fake_loss + 1e2 * g_rsl + 1e1 * g_fsl + 1e2 * g_rsl_2    # for chair
        # loss_g = g_fake_loss + 1e2 * g_rsl + 5 * g_fsl + 1e2 * g_rsl_2    # for other
        
        d_fake_loss = -torch.log(1-d_fake+__E)
        d_real_loss = -torch.log(d_real+__E)
        loss_d = (d_real_loss+d_fake_loss)/2

        return [loss_g, loss_d, g_fake_loss, g_rsl, g_rsl_2, g_fsl, d_fake_loss, d_real_loss]




class UGAANLoss_test(BasicLoss):
    def __init__(self, dis=0.03):
        super().__init__()
        self.loss_name = ['acc', 'F1', 'IOU1', 'acc2', 'F12', 'IOU2']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistanceIdx()
        # self.dis = 0.04
        self.dis = dis

    def get_seg(self, idx, N):
        B, _ = idx.shape
        res = []
        for i in range(B):
            seg = torch.bincount(idx[i])
            n = seg.shape[0]
            seg = torch.cat([seg, torch.zeros([N-n]).cuda()])
            res.append(seg)
        res = torch.stack(res)
        return res

    
    def get_seg_2(self, dis, d):
        return (dis < d)

    def acc(self, seg, seg_gt):
        #seg = seg.permute(0, 2, 1)
        seg = torch.unsqueeze(seg, -1)
        seg = (seg >= 0.5).float()
        acc = (seg_gt == seg).float()
        acc = torch.mean(acc, -1)
        acc = torch.mean(acc, -1)
        acc = acc*100
        return acc
    
    def batch_forward(self, outputs, data):
        x_fake = outputs[2]
        x_fake_2 = outputs[4]

        points, colors = data[0][0]
        seg_gt = data[0][1]

        p2g, g2p, idx1, idx2 = self.distance(x_fake, points)
        seg = self.get_seg_2(g2p, self.dis)
        outputs.append(seg)
        p2g, g2p, idx1, idx2 = self.distance(x_fake_2, points)
        seg2 = self.get_seg_2(g2p, self.dis)
        outputs.append(seg2)

        acc = self.acc(seg, seg_gt)
        acc2 = self.acc(seg2, seg_gt)

        f1 = F1_for_seg(seg, seg_gt[:,:,0])
        f1 = f1*100

        f12 = F1_for_seg(seg2, seg_gt[:,:,0])
        f12 = f12*100

        iou1 = IOU_for_seg(seg, seg_gt[:,:,0])
        iou1 = iou1 * 100
        iou2 = IOU_for_seg(seg2, seg_gt[:,:,0])
        iou2 = iou2 * 100
        return [acc, f1, iou1, acc2, f12, iou2]


class USGANLoss_test_gt(BasicLoss):
    def __init__(self):
        super().__init__()
        self.loss_name = ['cd1', 'cd2', 'fcd1', 'fcd2']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistanceIdx()
        # self.dis = 0.09
        self.dis = 0.04
        # self.dis = 0.03

    def get_seg(self, idx, N):
        B, _ = idx.shape
        res = []
        for i in range(B):
            seg = torch.bincount(idx[i])
            n = seg.shape[0]
            seg = torch.cat([seg, torch.zeros([N-n]).cuda()])
            res.append(seg)
        res = torch.stack(res)
        return res

    
    def get_seg_2(self, dis, d):
        return (dis < d)

    def acc(self, seg, seg_gt):
        #seg = seg.permute(0, 2, 1)
        seg = torch.unsqueeze(seg, -1)
        seg = (seg >= 0.5).float()
        acc = (seg_gt == seg).float()
        acc = torch.mean(acc, -1)
        acc = torch.mean(acc, -1)
        acc = acc*100
        return acc
    
    def cd(self, p1, p2):
        p2g, g2p, _, _ = self.distance(p1, p2)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = p2g + g2p
        return cd, p2g, g2p

    def batch_forward(self, outputs, data):
        x_fake = outputs[2]
        x_fake_2 = outputs[4]

        points, colors = data[0][0]
        seg_gt = data[0][1]
        com_gt = data[1][0]

        p2g, g2p, idx1, idx2 = self.distance(x_fake, points)
        seg = self.get_seg_2(g2p, self.dis)
        outputs.append(seg)
        p2g, g2p, idx1, idx2 = self.distance(x_fake_2, points)
        seg2 = self.get_seg_2(g2p, self.dis)
        outputs.append(seg2)

        cd1, _, _ = self.cd(x_fake, com_gt)
        cd2, _, _ = self.cd(x_fake_2, com_gt)

        fcd1 = calc_fcd(x_fake, com_gt, a=0.001)
        fcd2 = calc_fcd(x_fake_2, com_gt, a=0.001)

        return [cd1, cd2, fcd1, fcd2]



if __name__ == '__main__':
    model = UGAAN()