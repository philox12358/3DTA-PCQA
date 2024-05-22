import os
import torch
import numpy as np
from torch.utils.data import Dataset



def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):                
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud[:,0:3], xyz1), xyz2).astype('float32')      
    x = np.concatenate((translated_pointcloud, pointcloud[:,3:]), axis=1)
    return x

def knearest(point, center, k):
    res = np.zeros((k,))                           # init the index
    xyz = point[:,:3]                              # 
    dist = np.sum((xyz - center) ** 2, -1)         # calcu distance
    order = [(dist[i],i) for i in range(len(dist))]
    order = sorted(order)
    for j in range(k):
        res[j] = order[j][1]
    point = point[res.astype(np.int32)]            # get k nearest point
    return point

def read_data_list(args, pattern):
    if pattern == 'train':
        txtfile = os.path.join(args.data_dir, args.patch_dir, 'train_data_list.txt')
    else:
        txtfile = os.path.join(args.data_dir, args.patch_dir, 'test_data_list.txt')

    shape_ids = [line.rstrip() for line in open(txtfile)]
    data_list = [None] * len(shape_ids)
    for i, shape_id in enumerate(shape_ids):
        data_list[i] = shape_id.split(', ')

    print(f'The size of  \"{args.patch_dir}/{pattern}\"  data is {len(data_list)}')
    return data_list
    
def load_data(message, args, pattern):
    npy_dir = f'{args.data_dir}/{args.patch_dir}/{pattern}/{message[0]}/{message[1]}'
    point_set = np.load(npy_dir)
    point_set = point_set[:,0:6]             # @@@@@@@@@@@@ Limit data dimension
    index = np.arange(point_set.shape[0])
    index = np.random.choice(index, args.point_num, replace=False)
    point = point_set[index]
    mos = torch.tensor(float(message[2])).float()
    filenum = torch.tensor(int(message[3]))

    return point, mos, filenum


def xyzrgb_normalize(point):
    point[:,0:3] = point[:,0:3] - np.mean(point[:,0:3],axis=0)
    point[:,3:6] = point[:,3:6] - np.mean(point[:,3:6],axis=0)
    return point


class WPC_SD(Dataset):
    def __init__(self, args, pattern):      
        self.num_points = args.point_num
        self.pattern = pattern
        self.data_list = read_data_list(args, pattern)
        self.data_len = len(self.data_list)
        self.args = args
        
    def __getitem__(self, item):
        message = self.data_list[item]
        point, mos, filenum= load_data(message, self.args, self.pattern)
        point = xyzrgb_normalize(point)
        if self.pattern == 'train':
            np.random.shuffle(point)
        return point, mos, filenum

    def __len__(self):
        return self.data_len
