import os
import sys
import numpy as np
import torch

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


def resample_pcd_idx(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    idx = idx[:n]
    return pcd[idx], idx


def get_sxyz_of_pc(pc):
    maxx = abs(np.max(pc[:,0]))
    minx = abs(np.min(pc[:,0]))
    maxy = abs(np.max(pc[:,1]))
    miny = abs(np.min(pc[:,1]))
    maxz = abs(np.max(pc[:,2]))
    minz = abs(np.min(pc[:,2]))
    return np.array([max(maxx, minx), max(maxy, miny), max(maxz, minz)], np.float32)


def get_sxyz_of_pcs(pcs):
    return np.array([get_sxyz_of_pc(pc) for pc in pcs], np.float32)

