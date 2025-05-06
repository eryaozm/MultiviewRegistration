import os
import open3d as o3d
import numpy as np


def convert_off_to_ply(input_path, output_dir):
    try:
        mesh = o3d.io.read_triangle_mesh(input_path)
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.ply")

        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")


def process_modelnet_off_files(modelnet_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(modelnet_dir):
        for file in files:
            if file.endswith(".off"):
                off_file_path = os.path.join(root, file)
                convert_off_to_ply(off_file_path, output_dir)


def main():
    modelnet_dir = "/home/eryao/code/point/ModelNet40"
    output_dir = "/home/eryao/code/point/source"
    process_modelnet_off_files(modelnet_dir, output_dir)

if __name__ == "__main__":
    main()