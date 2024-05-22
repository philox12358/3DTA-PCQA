import os
import os.path
import numpy as np
from plyfile import PlyData
import pandas as pd                   
import argparse
from multiprocessing import Pool, current_process
import xlrd
#from patch_list_create import create_list
from sklearn.neighbors import KDTree
import open3d as o3d


def rgb_normalize(rgb):
    centroid = np.mean(rgb, axis=0)
    rgb = rgb - centroid                           
    return rgb


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid                            
    return pc


def xyz_1_2001(xyz):
  new_xyz = np.copy(xyz)
  # new_xyz[:0] = xyz[:0] - xyz[:0].min()   # 平移到xyz正坐标
  # new_xyz[:1] = xyz[:1] - xyz[:1].min()
  # new_xyz[:2] = xyz[:2] - xyz[:2].min()
  new_xyz = new_xyz - new_xyz.min()   # 最小值置0
  scale = xyz.max() - xyz.min()       # 最大值最小值之差
  new_xyz = new_xyz / scale           # 归一化
  new_xyz = new_xyz*2000 + 1          # 1-2001之间
  return new_xyz



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D], 
        npoint: number of samples (1024)
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape                  
    xyz = point[:,:3]                   
    centroids = np.zeros((npoint,))    
    distance = np.ones((N,)) * 1e10    
    farthest = np.random.randint(0, N)  
    for i in range(npoint):             
        centroids[i] = farthest       
        centroid = xyz[farthest, :]     
        dist = np.sum((xyz - centroid) ** 2, -1)   
        mask = dist < distance                     
        distance[mask] = dist[mask]                
        farthest = np.argmax(distance, -1)         
    point = point[centroids.astype(np.int32)]      
    return point                                   



def knearest(point, center, k):
    res = np.zeros((k,))                           
    xyz = point[:,:3]                              
    dist = np.sum((xyz - center) ** 2, -1)        
    order = [(dist[i],i) for i in range(len(dist))]      
    order = sorted(order)
    for j in range(k):
        res[j] = order[j][1]
    point = point[res.astype(np.int32)]            
    return point


def write_txt(kpoint, path, filename):
    N, D = kpoint.shape
    txt_file = open(filename,"a+")
    for i in range(N):
        for j in range(D):
            txt_file.write(str(kpoint[i][j]))
            if j != D-1:
                txt_file.write(',')
        txt_file.write('\n')
    txt_file.close()
    

def create_patch(id , path, args):
    ply_str = path.strip().split('.')[0]    
    folder = os.path.join(args.data_dir, args.patch_dir, ply_str)
    if not os.path.exists(folder):
        os.mkdir(folder)                        
        work_dir = folder       
    else:
        print(f'stride the {id}_th file...................')
        return
    
    PC_dir = os.path.join(args.data_dir, args.ply_dir, path)

    # plydata = PlyData.read(PC_dir)            
    # pc_df = pd.DataFrame(plydata['vertex'].data).astype(np.float32)    
    # column = list(pc_df)                   
    # point = np.zeros(pc_df.shape)           
    # for j, col in enumerate(column):
    #     point[:,j] = list(pc_df[col])

    pcd = o3d.io.read_point_cloud(PC_dir)
    PC_points = np.asarray(pcd.points)                         # 拿出点
    PC_colors = np.asarray(pcd.colors)*255
    point_cloud = np.concatenate((PC_points, PC_colors), axis=1)
    
    point_cloud[:,0:3] = xyz_1_2001(point_cloud[:,0:3])        # normalize xyz to 1-2001
    points = point_cloud[:,0:3]
    kd_tree = KDTree(points)
    

    # point[:,0:3] = xyz_1_2001(point[:,0:3])        #! 每一个点云的xyz坐标置1-2001

    centers = farthest_point_sample(point_cloud, args.center_points)    

    for m, center in enumerate(centers):
        center = center[0:3]
        filename = ply_str + '__' + str(m)             
        filename = os.path.join(work_dir, filename)
        if point_cloud.shape[0]<10001:
            kpoint = point_cloud
        else:
            # kpoint = knearest(point_cloud, center, args.k_nearest)[:,:6]
            dist_to_knn, knn_idx = kd_tree.query(X=[center], k=args.k_nearest)   # 0为中心点索引，k为邻居数
            kpoint = point_cloud[knn_idx[0]][:,:6]
        np.save(filename, kpoint)                            
    print(f'The {id+1}_th file completed, name:  {ply_str}.ply')



def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    return dataFile



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='expriment setting')
    
    parser.add_argument('--data_dir', type=str,  default='../data/WPC', help='Where does ply file exist?')
    parser.add_argument('--ply_dir', type=str,   default='Distortion_ply', help='Where does ply file exist?')
    parser.add_argument('--patch_dir', type=str, default='patch_72_10000', help='Where to store patch?')

    parser.add_argument('--center_points', type=int, default=72, help='number of patches?')
    parser.add_argument('--k_nearest', type=int, default=10000, help='points numbers of each patch have?')
    parser.add_argument('--pattern', type=str, default='test', choices=['train','test'])
    
    args = parser.parse_args()


    # Create train patch file ----------------------------------------------------------
    try:
        os.mkdir(os.path.join(args.data_dir, args.patch_dir))    # 新建patch所在文件夹
    except:
        print(f'patch文件夹已经存在。。。')

    exle_file = read_xlrd(os.path.join(args.data_dir, 'mos.xls')) 

    pool = Pool(10)         # Create a process pool of the specified size
    for id, name in enumerate(exle_file):
        pool.apply_async(func=create_patch, args=(id, name[0], args,))  #apply_async提高程序执行的并行度从而提高程序的执行效率
    pool.close()     # process pool is closed
    pool.join()      # The main process waits for the child process to complete before ending


    # create_patch(0, 'bag_level_9.ply', args)      # 单文件调试
    print(f'All files done, and data list created...')


 