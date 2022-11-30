import os
import h5py
import glob
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from pointnet2_utils import furthest_point_sample 

def normalize_data(pc):
    # for pc in pcs:
    #get furthest point distance then normalize
    d = max(np.sum(np.abs(pc)**2, axis=-1)**(1./2))
    pc /= d

    # pc[:,0]/=max(abs(pc[:,0]))
    # pc[:,1]/=max(abs(pc[:,1]))
    # pc[:,2]/=max(abs(pc[:,2]))

    return pc

#USE For SUNCG, to center to origin
def center_data(pc):
    # for pc in pcs:
    centroid = np.mean(pc, axis=0)
    pc[:,0] -= centroid[0]
    pc[:,1] -= centroid[1]
    pc[:,2] -= centroid[2]

    return pc

def normalize(pcd):
    maxs = np.max(pcd, 0, keepdims=True)
    mins = np.min(pcd, 0, keepdims=True)
    center = (maxs+mins)/2
    scale = (maxs-mins)/2
    max_scale = np.max(scale)
    pcd = pcd-center
    pcd = pcd / max_scale
    return pcd

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def ori_jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def ori_rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''

    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    sampled_pts_idx = furthest_point_sample(pts, num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
    batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx


def load_data_coseg(path="/home/jimmy15923/mnt/data/s3dis_instance_bg_points/", area="Area_*", obj='chair', num_points=2048, pre_process=True):
    all_files = glob.glob(f"{path}/{area}/{obj}*.h5")

    coord_list, color_list, label_list = [], [], []
    for f in tqdm(all_files):
        file = h5py.File(f, 'r+')
        coord = file["data"][:]
        color = file["color"][:]
        label = file["label"][:]
        file.close()
        
        # if coord.shape[0] < num_points:
        #     continue            
            
#         if num_points > num_points:
        _, indices = farthest_pts_sampling_tensor(torch.tensor(coord)[None].cuda(), num_points, return_sampled_idx=True)
        indices = indices.cpu().numpy()[0]
        coord = coord[indices]
        color = color[indices]
        label = label[indices,0]
        
#         coord = coord[:num_points,:]
#         color = color[:num_points]
#         label = label[:num_points,0]       

        coord_list.append(coord)
        color_list.append(color)
        label_list.append(label)

    coords = np.array(coord_list)
    colors = np.array(color_list)
    labels = np.array(label_list, dtype=np.int32)
    labels = np.expand_dims(labels, -1)

    return coords, colors, labels



class S3DIS_coseg(Dataset):
    def __init__(self, path="/home/jimmy15923/mnt/data/s3dis_instance_bg_points", partition='train', area="Area_*", obj=6,
                 num_points=10240, label_binarize=True, norm=True, center=True, return_color=False):
        
        cat_to_label = {6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase'}
        cat = cat_to_label[obj]
        self.cat = cat
        print(f"Select {cat} object")
        data = load_data_coseg(path=path, area=area, obj=cat, num_points=num_points, pre_process=True)
        self.coord = data[0]
        self.feat = data[1]
        self.label = data[2]

        self.return_color = return_color
        if label_binarize:
            self.label[self.label != obj] = 0
            self.label[self.label == obj] = 1      
                      
        self.partition = partition
        self.norm = norm
        self.center = center
        self.label_binarize = label_binarize
        
    def __getitem__(self, item):
        coord = self.coord[item]
        feat = self.feat[item]
        label = self.label[item]
        
        # if self.center:
        #     coord = center_data(coord)
        # if self.norm:
        #     coord = normalize_data(coord)        
        
        coord = normalize(coord)

        if self.partition == 'train':
            indices = list(range(coord.shape[0]))
            np.random.shuffle(indices)
            coord = coord[indices]
            feat = feat[indices]
            label = label[indices]

        if self.return_color:
            return coord, feat, label, self.cat          
        else:
            return coord, label, self.cat 

    def __len__(self):
        return len(self.coord)
    
    @staticmethod
    def to_device(data, device):
        data = list(data)
        data[0][0] = data[0][0].to(device)
        data[0][1] = data[0][1].to(device)
        data[1] = data[1].to(device)
        return data

    @staticmethod
    def _collate_fn(datas):
        _points = torch.from_numpy(np.array([data[0] for data in datas]).astype(np.float32))
        # _colors = torch.from_numpy(np.array([data[0][1] for data in datas]).astype(np.float32))
        _seg = torch.from_numpy(np.array([data[1] for data in datas]).astype(np.float32))
        _ids = [data[2] for data in datas]
        return [_points, _points], _seg, _ids    
   
    