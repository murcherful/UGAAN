import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true', help='Continue to train with the last saved weight.')
parser.add_argument('--cate', default='chair', choices=['chair', 'table', 'bookshelf', 'sofa'], help='Category.')
parser.add_argument('--cuda_index', default='0', help='Use which GPU.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('./util')
from args_scannet import ARGS_SCANNET
from train_util_gan import TrainFramework
from my_dataset import USGANDataset
from ugaan import UGAAN


def train(args):
    ARGS = ARGS_SCANNET(args.cate)
    train_dataset = USGANDataset(ARGS.lmdb_train, ARGS.lmdb_sn, ARGS.input_point_number, False)
    train_dataset.sn.set_args(ARGS.min_scale, ARGS.max_scale, ARGS.max_center_shift)
    train_dataset.sn.pr = ARGS.pr
    valid_dataset = USGANDataset(ARGS.lmdb_valid, ARGS.lmdb_sn, ARGS.input_point_number, False)
    valid_dataset.sn.set_args(ARGS.min_scale, ARGS.max_scale, ARGS.max_center_shift)
    valid_dataset.sn.pr = ARGS.pr

    
    net = UGAAN(ARGS.dis)

    optimizer = 'Adam'

    train_framework = TrainFramework(ARGS.batch_size, ARGS.log_dir, args.restore, args.cuda_index)
    train_framework._set_dataset(ARGS.lmdb_train, ARGS.lmdb_valid, train_dataset, valid_dataset)
    train_framework._set_net(net, 'WSGANNet')
    train_framework._set_optimzer(optimizer, lr=1e-5, weight_decay=0)
    train_framework.train(ARGS.max_epoch, save_pre_epoch=ARGS.save_pre_epoch, print_pre_step=ARGS.print_pre_step, test_pre_step=ARGS.test_pre_step)


if __name__ == '__main__':
    train(args)