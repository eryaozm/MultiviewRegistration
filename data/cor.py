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
        self.intensity_params = {
            'base_intensity': 0.5,
            'distance_factor': 0.4,
            'angle_factor': 0.3,
            'material_factor': 0.2,
            'noise_level': 0.05
        }

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

    def place_virtual_lidars(self, num_lidars=2, radius=3.0):
        lidar_positions = []
        lidar_orientations = []
        phi = np.linspace(np.pi / 8 * 3, np.pi / 8 * 3, num_lidars, endpoint=True)
        theta = np.linspace(np.pi / 8 * 5, np.pi / 8 * 7, num_lidars, endpoint=True)

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

    def calculate_intensity(self, hit_point, hit_normal, ray_origin, ray_direction, distance, material_id=0):
        intensity = self.intensity_params['base_intensity']
        distance_attenuation = 1.0 / (1.0 + self.intensity_params['distance_factor'] * distance ** 2)
        if hit_normal is not None and np.linalg.norm(hit_normal) > 0:
            hit_normal = hit_normal / np.linalg.norm(hit_normal)
            angle_factor = abs(np.dot(-ray_direction, hit_normal))
            angle_attenuation = angle_factor ** self.intensity_params['angle_factor']
        else:
            angle_attenuation = 1.0
        material_reflection = 0.7 + 0.3 * (material_id % 10) / 10.0
        material_effect = material_reflection ** self.intensity_params['material_factor']
        noise = 1.0 + np.random.uniform(-self.intensity_params['noise_level'],
                                        self.intensity_params['noise_level'])
        intensity = intensity * distance_attenuation * angle_attenuation * material_effect * noise
        intensity = np.clip(intensity, 0.0, 1.0)
        return intensity
    def get_face_normal(self, mesh, face_id):
        if face_id >= 0 and face_id < len(mesh.faces):
            vertices = mesh.vertices[mesh.faces[face_id]]
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            return normal
        return np.array([0, 0, 1])

    def simulate_lidar_scan(self, mesh, lidar_position, lidar_orientation,
                            horizontal_resolution=1500, vertical_resolution=64,
                            fov_h=180, fov_v=40, max_range=10.0):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        scene.add_triangles(mesh_t)
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
        primitive_ids = ans['primitive_ids'].numpy()
        valid_hit = hit != np.inf
        points_with_intensity = []
        for i in range(len(rays)):
            if valid_hit[i] and hit[i] <= max_range:
                point = rays[i] + directions[i] * hit[i]
                face_id = primitive_ids[i]
                face_normal = self.get_face_normal(mesh, face_id)
                intensity = self.calculate_intensity(
                    hit_point=point,
                    hit_normal=face_normal,
                    ray_origin=rays[i],
                    ray_direction=directions[i],
                    distance=hit[i],
                    material_id=face_id % 10
                )
                points_with_intensity.append(np.append(point, intensity))
        return np.array(points_with_intensity) if points_with_intensity else np.zeros((0, 4))
    def process_model(self, model_file, category, visualize=False):
        if model_file.endswith('.off'):
            mesh = self.load_off_file(model_file)
        elif model_file.endswith('.obj'):
            mesh = self.load_obj_file(model_file)
        mesh = self.center_and_normalize_mesh(mesh)
        lidar_positions, lidar_orientations = self.place_virtual_lidars(num_lidars=2)
        points_list = []
        poses_list = []

        for i in range(len(lidar_positions)):
            points_with_intensity = self.simulate_lidar_scan(
                mesh,
                lidar_positions[i],
                lidar_orientations[i]
            )
            points_list.append(points_with_intensity)
            poses_list.append((lidar_positions[i], lidar_orientations[i]))
        relative_pose = self.get_relative_pose(poses_list[0], poses_list[1])

        if visualize:
            self.visualize_results(points_list, poses_list, mesh)
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        save_dir = os.path.join(self.save_path, category)
        os.makedirs(save_dir, exist_ok=True)
        for i, points_with_intensity in enumerate(points_list):
            save_file = os.path.join(save_dir, f"{model_name}_lidar{i}.ply")
            pcd = o3d.geometry.PointCloud()
            if points_with_intensity.shape[0] > 0:
                points = points_with_intensity[:, :3]
                intensities = points_with_intensity[:, 3]
                pcd.points = o3d.utility.Vector3dVector(points)
                colors = np.zeros((len(intensities), 3))
                colors[:, 0] = intensities  # R
                colors[:, 1] = intensities  # G
                colors[:, 2] = intensities  # B
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(save_file, pcd)
            if points_with_intensity.shape[0] > 0:
                pcd_file = os.path.join(save_dir, f"{model_name}_lidar{i}.pcd")
                with open(pcd_file, 'w') as f:
                    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                    f.write(f"VERSION 0.7\n")
                    f.write(f"FIELDS x y z intensity\n")
                    f.write(f"SIZE 4 4 4 4\n")
                    f.write(f"TYPE F F F F\n")
                    f.write(f"COUNT 1 1 1 1\n")
                    f.write(f"WIDTH {points_with_intensity.shape[0]}\n")
                    f.write(f"HEIGHT 1\n")
                    f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
                    f.write(f"POINTS {points_with_intensity.shape[0]}\n")
                    f.write(f"DATA ascii\n")
                    for p in points_with_intensity:
                        f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")
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
            option = vis.get_render_option()
            if hasattr(option, 'mesh_show_back_face'):
                option.mesh_show_back_face = True
        lidar_markers = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        for i, points_with_intensity in enumerate(points_list):
            if len(points_with_intensity) == 0:
                continue
            points = points_with_intensity[:, :3]
            intensities = points_with_intensity[:, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = np.zeros((len(intensities), 3))
            for j, intensity in enumerate(intensities):
                colors[j, 0] = intensity
                colors[j, 1] = intensity * 0.5
                colors[j, 2] = 1.0 - intensity
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pcd)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(poses_list[i][0])
            sphere.paint_uniform_color(lidar_markers[i % len(lidar_markers)])
            vis.add_geometry(sphere)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5,
            origin=[1, 1, 1]
        )
        vis.add_geometry(coordinate_frame)
        if len(points_list) > 0 and len(points_list[0]) > 0:
            self.add_intensity_legend(vis)
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        vis.run()
        vis.destroy_window()

    def add_intensity_legend(self, vis, position=[0.8, 0.1, 0], size=0.5):
        num_samples = 20
        intensities = np.linspace(0, 1, num_samples)
        legend_points = []
        legend_colors = []
        for i, intensity in enumerate(intensities):
            point = np.array([position[0], position[1] + (i / num_samples) * size, position[2]])
            legend_points.append(point)
            color = np.array([intensity, intensity * 0.5, 1.0 - intensity])
            legend_colors.append(color)
        legend = o3d.geometry.PointCloud()
        legend.points = o3d.utility.Vector3dVector(legend_points)
        legend.colors = o3d.utility.Vector3dVector(legend_colors)
        vis.add_geometry(legend)


def main():
    modelnet40_path = "/home/eryao/code/point/ModelNet40"
    save_path = "/home/eryao/code/point/data/lidar"
    simulator = ModelNet40LidarSimulator(modelnet40_path, save_path)
    # test_file = os.path.join(modelnet40_path, "sofa", "train", "sofa_0001.off")
    # simulator.process_model(test_file, "sofa", visualize=True)
    simulator.process_dataset(subset='train', visualize=False)
    simulator.process_dataset(subset='test', visualize=False)


if __name__ == "__main__":
    main()