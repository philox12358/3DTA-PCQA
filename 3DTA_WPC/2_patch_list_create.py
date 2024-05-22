import os
import os.path
from tqdm import tqdm 
import pandas as pd                         
import argparse
import xlrd


def create_list(args, dic_name_mos):
    dataroot = os.path.join(args.data_dir, args.patch_dir)       # patch_72 / test
    file_name = os.path.join(args.data_dir, f'patch_data_list_{args.pattern}.txt')

    file_file = open(file_name,"a+")                      # 新建并打开txt文件
    file_file.truncate(0)            # 清空文件
    

    for filenum, [name, mos] in enumerate(dic_name_mos):
        patch_folder = name.split('.')[0]
        patchs_name = os.listdir(os.path.join(dataroot, patch_folder))   # 每个文件夹内的文件列表
        for j, patch_name in enumerate(patchs_name):       # 循环单独文件夹的所有文件
            file_temp = f'{patch_folder}, {patch_name}, {mos}, {filenum}, {j}\n'
            file_file.write(file_temp)
    file_file.close()
    print(f'\n\n{args.pattern}文件列表创建完成...\n\n')

    
def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    return dataFile


if __name__ == '__main__':
    # setting
    parser = argparse.ArgumentParser(description='Experiment setting')

    parser.add_argument('--data_dir', type=str,  default='../data/WPC', help='Where does ply file exist?')
    parser.add_argument('--patch_dir', type=str, default='patch_72_10000', help='Where to store patch?')
    parser.add_argument('--pattern', type=str, default='train', choices=['train','test'])

    args = parser.parse_args()

    # Create train file_list ----------------------------------------------------------
    args.pattern = 'train'
    exle_file = read_xlrd(os.path.join(args.data_dir, args.pattern + '.xls'))
    create_list(args, exle_file)


    # Create test file_list ----------------------------------------------------------
    args.pattern = 'test'
    exle_file = read_xlrd(os.path.join(args.data_dir, args.pattern + '.xls'))
    create_list(args, exle_file)

