import os
# cuda_index = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index
import sys
from unicodedata import decimal

import torch
from torch.utils.data import Dataset, DataLoader

import lmdb
import numpy as np

import msgpack
import msgpack_numpy   
msgpack_numpy.patch()

from print_util import *
from point_util import *
from io_util import *
from s3dis_dataset import S3DIS_coseg
from scanobjectnn_dataset import ScanObject_coseg

import open3d as o3d


class MyDataset(Dataset):
    def __init__(self, lmdb_path, input_point_num, gt_point_num, prefix='[MYDARASET]'):
        #print(lmdb_path, input_point_num, gt_point_num, prefix)
        self.lmdb_path = lmdb_path
        self.input_point_num = input_point_num
        self.gt_point_num = gt_point_num
        self.prefix = prefix
        self.have_self_collate_fn = False
        # load lmdb
        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, map_size=1099511627776 * 2)
        self.db = self.env.begin()
        #print(self.prefix)
        print_info('open db: ' + self.lmdb_path, prefix=self.prefix)
        # load keys
        self.keys = []
        keys = self.db.get(b'__keys__')
        if keys is not None:
            self.keys = msgpack.loads(keys, raw=False)
        else:
            for k in self.db.cursor(): 
                self.keys.append(k[0])
        self.size = len(self.keys)
        print_info('get %d entries' % self.size, prefix=self.prefix)


    def __del__(self):
        self.env.close()
        print_info('close db: ' + self.lmdb_path, prefix=self.prefix)


    def __len__(self):
        return self.size


    def __getitem__(self, index):
        buf = self.db.get(self.keys[index])
        data = msgpack.loads(buf, raw=False)
        _id, _image, _input, _gt, _camera_x, _camera_y = data
        _input = resample_pcd(_input, self.input_point_num)
        _input = _input.astype(np.float32)
        _gt = resample_pcd(_gt, self.gt_point_num)
        _gt = _gt.astype(np.float32)   # gt in lmdb data is string type 
        return _input, _gt, _id


    @staticmethod 
    def to_device(data, device):
        _ipcs = data[0].to(device)
        _cpcs = data[1].to(device)
        _ids = data[2]
        return _ipcs, _cpcs, _ids
    
    @staticmethod
    def _collate_fn(datas):
        _ipcs = torch.from_numpy(np.array([data[0] for data in datas]))
        _cpcs = torch.from_numpy(np.array([data[1] for data in datas]))
        _ids = [data[2] for data in datas]
        return _ipcs, _cpcs, _ids


def point_multiply(M, pc):
    xyz1 = np.concatenate([pc, np.ones([pc.shape[0], 1])],1).transpose(1, 0)
    xyz1 = np.dot(M, xyz1)
    xyz1 = xyz1.transpose(1, 0)
    return xyz1[:,:3]


def get_random_pos(center, r):
    a, b = np.random.rand(2)
    a = a * 2 * np.pi - np.pi
    b = b * np.pi - np.pi / 2 
    r = r * 2
    x = np.cos(a) * r + center[0]
    y = np.sin(a) * r + center[1]
    z = np.sin(b) * r + center[2]
    return x, y, z 


def random_remove_unseen(points):
    points = points.copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #pcd = pcd.voxel_down_sample(0.03)

    center = pcd.get_center()
    max_bound = pcd.get_max_bound()
    r = max_bound - center 
    r = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
    r = np.sqrt(r)
    x, y, z = get_random_pos(center, r)

    res, index = pcd.hidden_point_removal([x, y, z], 1000)
    res = points[index,:]
    return res


def trans(_points, _center, _scale, a, b):
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    cos_b = np.cos(b)
    sin_b = np.sin(b)

    # scale
    _points = _scale*_points
    # rotate
    x = _points[:,0:1]
    y = _points[:,1:2]
    z = _points[:,2:3]

    new_z = cos_b*z - sin_b*y
    new_y = sin_b*z + cos_b*y

    new_x = cos_a*x - sin_a*new_z 
    new_z = sin_a*x + cos_a*new_z  
    _points = np.concatenate([new_x, new_y, new_z], 1)
    # shift
    _points = _center + _points
    return _points

class ShapeNetDataset(MyDataset):
    # # MAX_CENTER_SHIFT = 0.5
    # MAX_CENTER_SHIFT = 0.1
    # MIN_SCALE = 0.2
    # MAX_SCALE = 1.0
    
    def __init__(self, lmdb_path, input_point_num, trans=True, r=True, remove_unseen=False, prefix='[SHAPENETDARASET]',
                 min_scale=0.5, max_scale=0.8, max_center_shift=0.0):
        super(ShapeNetDataset, self).__init__(lmdb_path, input_point_num, 0, prefix)
        self.MAX_CENTER_SHIFT = max_center_shift
        self.MIN_SCALE = min_scale
        self.MAX_SCALE = max_scale
        self.AVG_SCALE = (self.MAX_SCALE+self.MIN_SCALE)/2
        self.SSS_SCALE = (self.MAX_SCALE-self.MIN_SCALE)/2
        self.trans = trans
        self.r = r
        self.pr = False
        self.remove_unseen = remove_unseen
    
    def set_args(self, min_scale, max_scale, max_center_shift):
        self.MAX_CENTER_SHIFT = max_center_shift
        self.MIN_SCALE = min_scale
        self.MAX_SCALE = max_scale
        self.AVG_SCALE = (self.MAX_SCALE+self.MIN_SCALE)/2
        self.SSS_SCALE = (self.MAX_SCALE-self.MIN_SCALE)/2

    def __getitem__(self, index):
        buf = self.db.get(self.keys[index])
        data = msgpack.loads(buf, raw=False)
        _id, _points = data
        _points = _points[:,:3]

        _points = resample_pcd(_points, self.input_point_num*2)

        if self.remove_unseen:
            incom_points = random_remove_unseen(_points)

        _points = resample_pcd(_points, self.input_point_num)

        if self.remove_unseen:
            incom_points = resample_pcd(incom_points, self.input_point_num)

        if self.trans:
            cxyz = np.random.rand(1, 3)*2 - 1
            _center = self.MAX_CENTER_SHIFT*cxyz
            sxyz = np.random.rand(1, 3)*2 - 1
            _scale = self.AVG_SCALE + self.SSS_SCALE*sxyz
            if self.r:
                a = np.random.rand(1, 1)*2 - 1
                a = a * np.pi
                if self.pr:
                    b = np.random.rand(1, 1) - 0.5
                    b = b * np.pi
                else:
                    b = 0
            else:
                a = 0
                b = 0
            _points = trans(_points, _center, _scale, a, b)
            
            if self.remove_unseen:
                incom_points = trans(incom_points, _center, _scale, a)

        if self.remove_unseen:
            return incom_points, _points, _id 
        else:
            return _points, _id 
    
    @staticmethod
    def to_device(data, device):
        data = list(data)
        data[0] = data[0].to(device)
        if len(data) == 3:
            data[1] = data[1].to(device)
        return data

    @staticmethod
    def _collate_fn(datas):
        if len(datas[0]) == 3:
            incom_points = torch.from_numpy(np.array([data[0] for data in datas])).float()
            _points = torch.from_numpy(np.array([data[1] for data in datas])).float()
            _ids = [data[2] for data in datas]
            return incom_points, _points, _ids
        else:
            _points = torch.from_numpy(np.array([data[0] for data in datas])).float()
            _ids = [data[1] for data in datas]
            return _points, _ids


class USDataset(MyDataset):
    def __init__(self, lmdb_path, input_point_num, prefix='[USDARASET]'):
        super(USDataset, self).__init__(lmdb_path, input_point_num, 0, prefix)
    
    def __getitem__(self, index):
        buf = self.db.get(self.keys[index])
        data = msgpack.loads(buf, raw=False)
        _id, _points, _colors, _seg = data
        new_points = np.concatenate([_points, _colors, _seg], 1)
        new_points = resample_pcd(new_points, self.input_point_num)
        _points = new_points[:,:3]
        _colors = new_points[:,3:6]
        _seg = new_points[:,6:]
        _id = _id.split('-')[1]
        #print(_points.shape)
        #print(_colors.shape)
        #print(_seg.shape)
        return [_points, _colors], _seg, _id 
    
    @staticmethod
    def to_device(data, device):
        data = list(data)
        data[0][0] = data[0][0].to(device)
        data[0][1] = data[0][1].to(device)
        data[1] = data[1].to(device)
        return data

    @staticmethod
    def _collate_fn(datas):
        _points = torch.from_numpy(np.array([data[0][0] for data in datas]).astype(np.float32))
        _colors = torch.from_numpy(np.array([data[0][1] for data in datas]).astype(np.float32))
        _seg = torch.from_numpy(np.array([data[1] for data in datas]).astype(np.float32))
        _ids = [data[2] for data in datas]
        return [_points, _colors], _seg, _ids 


class USGANDataset(USDataset):
    def __init__(self, lmdb_ws_path, lmdb_sn_path, input_point_num, remove_unseen, prefix='[USGAN]'):
        super().__init__(lmdb_ws_path, input_point_num, prefix+'[US]')
        self.sn = ShapeNetDataset(lmdb_sn_path, input_point_num, r=True, remove_unseen=remove_unseen, prefix=prefix+'[SN]')
        self.sn_len = len(self.sn)
        self.sn_idx = 0
        self.sn_index = list(range(self.sn_len))
        np.random.shuffle(self.sn_index)
    
    def get_sn_index(self):
        if self.sn_idx >= self.sn_len:
            self.sn_idx = 0
            np.random.shuffle(self.sn_index)
        res = self.sn_index[self.sn_idx]
        self.sn_idx += 1
        return res        

    def __getitem__(self, index):
        ws_data = super().__getitem__(index)
        sn_data = self.sn.__getitem__(self.get_sn_index())
        return ws_data, sn_data
    
    @staticmethod
    def to_device(data, device):
        ws_data, sn_data = data
        ws_data = USDataset.to_device(ws_data, device)
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return ws_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        ws_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        ws_datas = USDataset._collate_fn(ws_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return ws_datas, sn_datas


class USGANDataset_S3DIS:
    def __init__(self, lmdb_ws_path, lmdb_sn_path, input_point_num, remove_unseen, s3dis_obj=8, s3dis_area="Area_*", prefix='[WSGAN]'):
        self.ws = S3DIS_coseg(path=lmdb_ws_path, num_points=input_point_num, obj=s3dis_obj, area=s3dis_area)
        self.sn = ShapeNetDataset(lmdb_sn_path, input_point_num, r=True, remove_unseen=remove_unseen, prefix=prefix+'[SN]', 
                                  #min_scale=0.5, max_scale=0.8)
                                  min_scale=0.3, max_scale=0.7)
        self.sn_len = len(self.sn)
        self.sn_idx = 0
        self.sn_index = list(range(self.sn_len))
        np.random.shuffle(self.sn_index)
    
    def __len__(self):
        return len(self.ws)

    def get_sn_index(self):
        if self.sn_idx >= self.sn_len:
            self.sn_idx = 0
            np.random.shuffle(self.sn_index)
        res = self.sn_index[self.sn_idx]
        self.sn_idx += 1
        return res        

    def __getitem__(self, index):
        ws_data = self.ws.__getitem__(index)
        sn_data = self.sn.__getitem__(self.get_sn_index())
        return ws_data, sn_data
    
    @staticmethod
    def to_device(data, device):
        ws_data, sn_data = data
        ws_data = S3DIS_coseg.to_device(ws_data, device)
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return ws_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        ws_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        ws_datas = S3DIS_coseg._collate_fn(ws_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return ws_datas, sn_datas


class USGANDataset_ScanObj:
    def __init__(self, lmdb_ws_path, lmdb_sn_path, input_point_num, remove_unseen, scanobj_obj=4, scanobj_par="training", prefix='[WSGAN]'):
        self.ws = ScanObject_coseg(raw_path=lmdb_ws_path, n_points=input_point_num, obj=scanobj_obj, partition=scanobj_par)
        self.sn = ShapeNetDataset(lmdb_sn_path, input_point_num, r=True, remove_unseen=remove_unseen, prefix=prefix+'[SN]', 
                                  min_scale=0.5, max_scale=0.8)
                                #   min_scale=0.3, max_scale=0.7)
        self.sn_len = len(self.sn)
        self.sn_idx = 0
        self.sn_index = list(range(self.sn_len))
        np.random.shuffle(self.sn_index)
    
    def __len__(self):
        return len(self.ws)

    def get_sn_index(self):
        if self.sn_idx >= self.sn_len:
            self.sn_idx = 0
            np.random.shuffle(self.sn_index)
        res = self.sn_index[self.sn_idx]
        self.sn_idx += 1
        return res        

    def __getitem__(self, index):
        ws_data = self.ws.__getitem__(index)
        sn_data = self.sn.__getitem__(self.get_sn_index())
        return ws_data, sn_data
    
    @staticmethod
    def to_device(data, device):
        ws_data, sn_data = data
        ws_data = ScanObject_coseg.to_device(ws_data, device)
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return ws_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        ws_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        ws_datas = ScanObject_coseg._collate_fn(ws_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return ws_datas, sn_datas


class USGANDataset_GT(MyDataset):
    ORG = 1
    STD = 2
    STD_S = 3
    STD_R = 4
    STD_T = 5
    STD_RS = 6

    def __init__(self, lmdb_path, input_point_num, gt_point_num, cate='chair', type=4, is_seg=True, prefix='[SEGCOMDATASET]'):
        MyDataset.__init__(self, lmdb_path, input_point_num, gt_point_num, prefix=prefix)
        self.type = type
        self.cate = cate
        self.is_seg = is_seg
        self.indexes = []
        for i in range(self.size):
            buf = self.db.get(self.keys[i])
            data = msgpack.loads(buf, raw=False)
            _id, _input, _seg, _gt, _M_T, _M_R, _M_S = data
            _id = _id.split('-')[1]
            if _id == cate:
                self.indexes.append(i)
        self.size = len(self.indexes)
        print_info('get %d entries of %s' % (self.size, cate), prefix=self.prefix)

    def remove_noise(self, points, seg):
        seg = seg > 0
        points = points[seg, :]
        seg = np.ones([points.shape[0]])
        return points, seg

    def trans(self, input_points, gt_points, TRS):
        if not self.type == self.ORG:
            M_T, M_R, M_S = TRS
            M = M_T.dot(M_R).dot(M_S)
            M = np.linalg.inv(M)
            if not self.type == self.STD:
                if self.type == self.STD_S:
                    M = M_S.dot(M)
                elif self.type == self.STD_R:
                    M = M_R.dot(M)
                elif self.type == self.STD_T:
                    M = M_T.dot(M)
                elif self.type == self.STD_RS:
                    M = M_R.dot(M_S).dot(M)
            input_points = point_multiply(M, input_points)
            gt_points = point_multiply(M, gt_points)
        return input_points, gt_points
    
    def normalize(self, inputs, gts):
        maxs = np.max(inputs, 0, keepdims=True)
        mins = np.min(inputs, 0, keepdims=True)
        center = (maxs+mins)/2
        scale = (maxs-mins)/2
        max_scale = np.max(scale)
        inputs = inputs-center
        inputs = inputs / max_scale
        gts = gts-center
        gts = gts / max_scale
        return inputs, gts

    def __getitem__(self, index):
        index = self.indexes[index]
        buf = self.db.get(self.keys[index])
        data = msgpack.loads(buf, raw=False)
        _id, _input, _seg, _gt, _M_T, _M_R, _M_S = data
        #_input, _gt = self.to_center(_input, _gt)
        _input, _gt = self.trans(_input, _gt, [_M_T, _M_R, _M_S])
        _input, _gt = self.normalize(_input, _gt)
        if not self.is_seg:
            _input, _seg = self.remove_noise(_input, _seg)
        _input_all = np.concatenate([_input, np.expand_dims(_seg, 1)], 1)
        _input_all = resample_pcd(_input_all, self.input_point_num)
        _input = _input_all[:,:3].astype(np.float32)
        _seg = _input_all[:,3:].astype(np.float32)
        _gt = resample_pcd(_gt, self.gt_point_num)
        _gt = _gt.astype(np.float32)   
        _id = _id.split('-')[1]
        ws_data = [[_input, _gt], _seg, _id]
        sn_data = [_gt, _id]
        return ws_data, sn_data

    @staticmethod
    def to_device(data, device):
        ws_data, sn_data = data
        ws_data = USDataset.to_device(ws_data, device)
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return ws_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        ws_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        ws_datas = USDataset._collate_fn(ws_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return ws_datas, sn_datas



# class FuseMultiDataset(Dataset):
#     def __init__(self, datasets):
#         self.datasets = datasets 
#         self.len = 0
#         self.lens = []
#         self.index_dict = {}
#         for i, dataset in enumerate(self.datasets):
#             self.lens.append(self.len)
#             dataset_len = len(dataset)
#             for j in range(self.len, self.len+dataset_len):
#                 self.index_dict[j] = i 
#             self.len = self.len + dataset_len

#     def __len__(self):
#         return self.len

#     def __getitem__(self, index):
#         dataset_idx = self.index_dict[index]
#         return self.datasets[dataset_idx].__getitem__(index-self.lens[dataset_idx])


# class MultiClassShapeNetDataset(Dataset):
#     def __init__(self, lmdb_paths, input_point_num, trans=True, r=True, remove_unseen=False, prefix='[SHAPENETDARASET]',
#                  min_scale=0.5, max_scale=0.8, max_center_shift=0.0):
#         self.datasets = []
#         for path in lmdb_paths:
#             self.datasets.append(ShapeNetDataset(path, input_point_num, trans, r, remove_unseen, prefix, min_scale, max_scale, max_center_shift))
#         self.dataset = FuseMultiDataset(self.datasets)
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         return self.dataset.__getitem__(index)
    
#     def set_args(self, min_scale, max_scale, max_center_shift):
#         for dataset in self.datasets:
#             dataset.set_args(min_scale, max_scale, max_center_shift)
    
#     @staticmethod
#     def to_device(data, device):
#         data = list(data)
#         data[0] = data[0].to(device)
#         if len(data) == 3:
#             data[1] = data[1].to(device)
#         return data

#     @staticmethod
#     def _collate_fn(datas):
#         if len(datas[0]) == 3:
#             incom_points = torch.from_numpy(np.array([data[0] for data in datas])).float()
#             _points = torch.from_numpy(np.array([data[1] for data in datas])).float()
#             _ids = [data[2] for data in datas]
#             return incom_points, _points, _ids
#         else:
#             _points = torch.from_numpy(np.array([data[0] for data in datas])).float()
#             _ids = [data[1] for data in datas]
#             return _points, _ids


# class MultiClassUSDataset(Dataset):
#     def __init__(self, lmdb_paths, input_point_num, prefix='[USDARASET]'):
#         self.datasets = []
#         for path in lmdb_paths:
#             self.datasets.append(USDataset(path, input_point_num, prefix))
#         self.dataset = FuseMultiDataset(self.datasets)
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         return self.dataset.__getitem__(index)

#     @staticmethod
#     def to_device(data, device):
#         data = list(data)
#         data[0][0] = data[0][0].to(device)
#         data[0][1] = data[0][1].to(device)
#         data[1] = data[1].to(device)
#         return data

#     @staticmethod
#     def _collate_fn(datas):
#         _points = torch.from_numpy(np.array([data[0][0] for data in datas]).astype(np.float32))
#         _colors = torch.from_numpy(np.array([data[0][1] for data in datas]).astype(np.float32))
#         _seg = torch.from_numpy(np.array([data[1] for data in datas]).astype(np.float32))
#         _ids = [data[2] for data in datas]
#         return [_points, _colors], _seg, _ids 


# class MultiClassUSGANDataset(Dataset):
#     def __init__(self, lmdb_ws_paths, lmdb_sn_paths, input_point_num, remove_unseen, prefix='[USGAN]'):
#         self.ws = MultiClassUSDataset(lmdb_ws_paths, input_point_num, prefix+'[US]')
#         self.sn = MultiClassShapeNetDataset(lmdb_sn_paths, input_point_num, r=True, remove_unseen=remove_unseen, prefix=prefix+'[SN]')
#         self.sn_len = len(self.sn)
#         self.sn_idx = 0
#         self.sn_index = list(range(self.sn_len))
#         np.random.shuffle(self.sn_index)
    
#     def get_sn_index(self):
#         if self.sn_idx >= self.sn_len:
#             self.sn_idx = 0
#             np.random.shuffle(self.sn_index)
#         res = self.sn_index[self.sn_idx]
#         self.sn_idx += 1
#         return res        
    
#     def __len__(self):
#         return len(self.ws)

#     def __getitem__(self, index):
#         ws_data = self.ws.__getitem__(index)
#         sn_data = self.sn.__getitem__(self.get_sn_index())
#         return ws_data, sn_data
    
#     @staticmethod
#     def to_device(data, device):
#         ws_data, sn_data = data
#         ws_data = USDataset.to_device(ws_data, device) 
#         sn_data = ShapeNetDataset.to_device(sn_data, device)
#         return ws_data, sn_data

#     @staticmethod
#     def _collate_fn(datas):
#         # fix here, datas = [[w, s], [w, s], ...]
#         ws_datas = [data[0]for data in datas]
#         sn_datas = [data[1]for data in datas]
#         ws_datas = USDataset._collate_fn(ws_datas)
#         sn_datas = ShapeNetDataset._collate_fn(sn_datas)
#         return ws_datas, sn_datas