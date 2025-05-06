import os
import open3d as o3d
from fpcs import use_fpcs
from inference import infer
import shutil
from pathlib import Path

def copy_ply_files(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.ply'):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dst_path)

def copy_ply_file(source,dir):
    src_path = source
    dst_path = os.path.join(dir, 'tmp.ply')
    shutil.copy2(src_path, dst_path)
    return dst_path

def get_ply_files(directory):
    ply_files = []
    abs_directory = os.path.abspath(directory)
    for root, dirs, files in os.walk(abs_directory):
        for file in files:
            if file.lower().endswith('.ply'):
                file_path = os.path.join(root, file)
                ply_files.append(file_path)

    return ply_files

def apply_trans(source_path, transform_matrix):
    pcd = o3d.io.read_point_cloud(source_path)
    pcd.transform(transform_matrix)
    o3d.io.write_point_cloud(source_path, pcd)

def reg_mod(source_path,target_path):
    trans_c = use_fpcs(source_path,target_path)
    apply_trans(source_path, trans_c)
    trans_f = infer(source_path,target_path)
    apply_trans(source_path, trans_c)
    trans_all = trans_c * trans_f
    return trans_all

def vox(source_path,target_path):
    pcd1 = o3d.io.read_point_cloud(source_path)
    pcd2 = o3d.io.read_point_cloud(target_path)
    pcd = pcd1 + pcd2
    return pcd