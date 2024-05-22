import os



ply_dir = '../data/WPC/Distortion_ply'
ply_name_list = os.listdir(ply_dir)

error_count = 0

for ply_name in ply_name_list:
    if '_rounded' in ply_name:
        error_count += 1
        temp_name = ply_name.split('_rounded')[0] + '.ply'
        # os.rename(os.path.join(ply_dir, ply_name), os.path.join(ply_dir, temp_name))     # rename the file

    print(f'There are {error_count} files be renamed.')

