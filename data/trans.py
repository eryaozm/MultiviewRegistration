import open3d as o3d
import numpy as np
import math


def random_rotation_translation(point_cloud, max_angle_deg=30, max_translation=1):
    transformed_point_cloud = point_cloud
    angle_x = np.random.uniform(-max_angle_deg, max_angle_deg) * math.pi / 180
    angle_y = np.random.uniform(-max_angle_deg, max_angle_deg) * math.pi / 180
    angle_z = np.random.uniform(-max_angle_deg, max_angle_deg) * math.pi / 180
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([angle_x, angle_y, angle_z])
    translation = np.random.uniform(-max_translation, max_translation, 3)
    transformed_point_cloud.rotate(rotation, center=transformed_point_cloud.get_center())
    transformed_point_cloud.translate(translation)
    return transformed_point_cloud


def main():
    ply_path = "/home/eryao/code/point/data/test/source/airplane_0001.ply"
    point_cloud = o3d.io.read_point_cloud(ply_path)
    point_cloud.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries_with_editing([point_cloud])
    transformed_point_cloud = random_rotation_translation(point_cloud)
    transformed_point_cloud.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.visualization.draw_geometries_with_editing([transformed_point_cloud])
    # output_path = "/home/eryao/code/point/data/test/target/airplane_0001.ply"
    # o3d.io.write_point_cloud(output_path, transformed_point_cloud)


if __name__ == "__main__":
    main()