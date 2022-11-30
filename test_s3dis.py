import os
import sys
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true', help='Continue to train with the last saved weight.')
parser.add_argument('--cate', default='chair', choices=['chair', 'table', 'bookshelf', 'sofa'], help='Category.')
parser.add_argument('--cuda_index', default='0', help='Use which GPU.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index

cat_to_label = {'table':7, 'chair':8, 'sofa':9, 'bookshelf':10}


import torch
from torch.utils.data import Dataset

sys.path.append('./util')
from args_scannet import ARGS_SCANNET
from test_util_gan import TestFramework
from my_dataset import USGANDataset_S3DIS
from io_util import *
from ugaan import UGAAN


def save_func(path, data, outputs, criterion, loss, time):
    if not os.path.exists(path):
        os.mkdir(path)
    inputs = data[0][0][0][0].to('cpu').detach().numpy()
    sn_data = data[1][0][0].to('cpu').detach().numpy()

    gt_seg = data[0][1][0].to('cpu').detach().numpy()
    x_fake = outputs[2][0].to('cpu').detach().numpy()
    x_real = outputs[3][0].to('cpu').detach().numpy()
    seg = outputs[-2][0].to('cpu').detach().numpy()
    x_fake_2 = outputs[4][0].to('cpu').detach().numpy()
    x_real_2 = outputs[5][0].to('cpu').detach().numpy()
    seg_2 = outputs[-1][0].to('cpu').detach().numpy()
    seg = np.expand_dims(seg, -1)
    seg_2 = np.expand_dims(seg_2, -1)

    write_point_cloud_with_seg_as_ply(os.path.join(path, 'input'), inputs, gt_seg)
    write_point_cloud_with_seg_as_ply(os.path.join(path, 'res'), inputs, seg)
    write_point_cloud_as_ply(os.path.join(path, 'x_fake'), x_fake)
    write_point_cloud_as_ply(os.path.join(path, 'x_real'), x_real)
    write_point_cloud_with_seg_as_ply(os.path.join(path, 'res_2'), inputs, seg_2)
    write_point_cloud_as_ply(os.path.join(path, 'x_fake_2'), x_fake_2)
    write_point_cloud_as_ply(os.path.join(path, 'x_real_2'), x_real_2)
    write_point_cloud_as_ply(os.path.join(path, 'sn_data'), sn_data)
    text = ''
    for j, name in enumerate(criterion.loss_name):
        text += '%s: %f, ' % (name, loss[j])
    with open(os.path.join(path, 'loss.log'), 'w') as f:
        f.write(text + '\n')
        f.write('time: %f\n' % time)


def test(args):
    ARGS = ARGS_SCANNET(args.cate)
    valid_dataset = USGANDataset_S3DIS(ARGS.lmdb_test_s3dis, ARGS.lmdb_sn, ARGS.input_point_number, False, s3dis_area='Area_5_*', s3dis_obj=cat_to_label[args.cate])
    net = UGAAN(ARGS.dis)

    test_framework = TestFramework(ARGS.log_dir, args.cuda_index, ARGS.res_dir_s3dis)
    test_framework._set_dataset(ARGS.lmdb_test_s3dis, valid_dataset)
    test_framework._set_net(net, 'WSNet')
    # test_framework.test(save_func, last_epoch=ARGS.max_epoch, save_index=list(range(len(valid_dataset))))
    test_framework.test(save_func, last_epoch=ARGS.max_epoch, save_index=list(range(10)))




if __name__ == '__main__':
    test(args)