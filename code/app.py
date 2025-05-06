import os
import open3d as o3d
from utility import copy_ply_files, copy_ply_file, get_ply_files, reg_mod, vox
import shutil
from pathlib import Path

input_dir = '/home/eryao/code/point/data/order'
output_dir ='/home/eryao/code/point/data/tmp'

copy_ply_files(input_dir, output_dir)
input_files = get_ply_files(output_dir)
file_num = len(input_files)

trans = []
pcd_final = None
for i in range(0, file_num-1):
    source = input_files[i+1]
    source = copy_ply_file(source, output_dir)
    target = input_files[i]
    trans.append(reg_mod(source,target))
    pcd = vox(source,target)
    dir = os.path.join(output_dir, '1')
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, '1_' + str(i) + '.ply')
    o3d.io.write_point_cloud(filename, pcd)

for j in range(2, file_num):
    input_files = get_ply_files(os.path.join(output_dir, str(j-1)))
    for i in range(0, file_num-j):
        source = input_files[i + 1]
        source = copy_ply_file(source, output_dir)
        target = input_files[i]
        trans.append(reg_mod(source, target))
        pcd = vox(source, target)
        dir = os.path.join(output_dir, 'j')
        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(dir, str(j)+'_' + str(i) + '.ply')
        o3d.io.write_point_cloud(filename, pcd)
