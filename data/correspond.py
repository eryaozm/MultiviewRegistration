import os
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

class ModelNet40LidarSimulator:
    def __init__(self, modelnet40_path, save_path=None):
        self.modelnet40_path = modelnet40_path
        self.save_path = save_path or os.path.join(os.getcwd(), 'lidar_scans')
        os.makedirs(self.save_path, exist_ok=True)

    def load_off_file(self, file_path):
        mesh = trimesh.load(file_path)
        return mesh

    def load_obj_file(self, file_path):
        mesh = trimesh.load(file_path)
        return mesh

    def center_and_normalize_mesh(self, mesh):
        mesh.vertices -= mesh.center_mass
        max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= max_dist
        return mesh

    def place_virtual_lidars(self, num_lidars=2, radius=2.0):
        lidar_positions = []
        lidar_orientations = []
        phi = np.linspace(np.pi/8*3, np.pi/8*3, num_lidars, endpoint=True)
        theta = np.linspace(np.pi/8*5, np.pi/8*7, num_lidars, endpoint=True)

        for i in range(num_lidars):
            x = radius * np.sin(phi[i]) * np.cos(theta[i])
            y = radius * np.sin(phi[i]) * np.sin(theta[i])
            z = radius * np.cos(phi[i])
            position = np.array([x, y, z])
            lidar_positions.append(position)
            z_axis = -position / np.linalg.norm(position)
            if np.abs(z_axis[2]) < 0.999:
                x_axis = np.cross(np.array([0, 0, 1]), z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
            else:
                x_axis = np.array([1, 0, 0])

            y_axis = np.cross(z_axis, x_axis)
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            lidar_orientations.append(rotation_matrix)

        return np.array(lidar_positions), np.array(lidar_orientations)

    def get_relative_pose(self, pose1, pose2):
        pos1, rot1 = pose1
        pos2, rot2 = pose2
        rot_rel = np.dot(rot2, rot1.T)
        pos_rel = pos2 - np.dot(rot_rel, pos1)
        return pos_rel, rot_rel

    def simulate_lidar_scan(self, mesh, lidar_position, lidar_orientation,
                            horizontal_resolution=1500, vertical_resolution=64,
                            fov_h=180, fov_v=40, max_range=10.0):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))
        h_angles = np.linspace(-fov_h / 2, fov_h / 2, horizontal_resolution) * np.pi / 180
        v_angles = np.linspace(-fov_v / 2, fov_v / 2, vertical_resolution) * np.pi / 180
        rays = []
        directions = []
        for h_angle in h_angles:
            for v_angle in v_angles:
                x = np.cos(v_angle) * np.sin(h_angle)
                y = np.sin(v_angle)
                z = np.cos(v_angle) * np.cos(h_angle)
                direction = np.dot(lidar_orientation, np.array([x, y, z]))
                directions.append(direction)
                rays.append(lidar_position)
        rays = np.array(rays).astype(np.float32)
        directions = np.array(directions).astype(np.float32)
        rays_combined = np.hstack([rays, directions])
        rays_tensor = o3d.core.Tensor(rays_combined)
        ans = scene.cast_rays(rays_tensor)
        hit = ans['t_hit'].numpy()
        valid_hit = hit != np.inf
        points = []
        for i in range(len(rays)):
            if valid_hit[i] and hit[i] <= max_range:
                point = rays[i] + directions[i] * hit[i]
                points.append(point)
        return np.array(points) if points else np.zeros((0, 3))

    def process_model(self, model_file, category, visualize=False):
        if model_file.endswith('.off'):
            mesh = self.load_off_file(model_file)
        elif model_file.endswith('.obj'):
            mesh = self.load_obj_file(model_file)
        mesh = self.center_and_normalize_mesh(mesh)
        lidar_positions, lidar_orientations = self.place_virtual_lidars(num_lidars=8)
        points_list = []
        poses_list = []
        for i in range(len(lidar_positions)):
            points = self.simulate_lidar_scan(
                mesh,
                lidar_positions[i],
                lidar_orientations[i]
            )
            points_list.append(points)
            poses_list.append((lidar_positions[i], lidar_orientations[i]))
        relative_pose = self.get_relative_pose(poses_list[0], poses_list[1])
        if visualize:
            self.visualize_results(points_list, poses_list, mesh)
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        save_dir = os.path.join(self.save_path, category)
        os.makedirs(save_dir, exist_ok=True)
        for i, points in enumerate(points_list):
            save_file = os.path.join(save_dir, f"{model_name}_lidar{i}.ply")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(save_file, pcd)
        pose_file = os.path.join(save_dir, f"{model_name}_relative_pose.npy")
        np.save(pose_file, {
            'relative_position': relative_pose[0],
            'relative_rotation': relative_pose[1]
        })
        return points_list, poses_list, relative_pose
    def process_dataset(self, subset='train', visualize=False):
        subset_path = os.path.join(self.modelnet40_path, subset)
        categories = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        for category in categories:
            category_path = os.path.join(subset_path, category)
            model_files = [os.path.join(category_path, f) for f in os.listdir(category_path)
                           if f.endswith('.off') or f.endswith('.obj')]
            for model_file in tqdm(model_files):
                self.process_model(model_file, category, visualize=visualize)

    def visualize_results(self, points_list, poses_list, mesh=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if mesh is not None:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            vis.add_geometry(o3d_mesh)
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        for i, points in enumerate(points_list):
            if len(points) == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(colors[i % len(colors)])
            vis.add_geometry(pcd)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(poses_list[i][0])
            sphere.paint_uniform_color(colors[i % len(colors)])
            vis.add_geometry(sphere)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5,
            origin=[1, 1, 1]
        )
        vis.add_geometry(coordinate_frame)
        vis.run()
        vis.destroy_window()

def main():
    modelnet40_path = "/home/eryao/code/point/ModelNet40"
    save_path = "/home/eryao/code/point/data/lidar"
    simulator = ModelNet40LidarSimulator(modelnet40_path, save_path)
    # cat = "tent"
    # test_file = os.path.join(modelnet40_path, cat, "train", cat+"_0001.off")
    # simulator.process_model(test_file, cat, visualize=True)
    simulator.process_dataset(subset='train', visualize=False)
    simulator.process_dataset(subset='test', visualize=False)

if __name__ == "__main__":
    main()