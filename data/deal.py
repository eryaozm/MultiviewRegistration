import open3d as o3d
import numpy as np


def simulate_low_reflectivity(point_cloud, reflectivity_threshold=0.3, removal_probability=0.7):
    processed_cloud = point_cloud.clone()
    points = np.asarray(processed_cloud.points)
    if not processed_cloud.has_colors():
        reflectivity = np.random.random(len(points))
        colors = np.zeros((len(points), 3))
        for i in range(len(points)):
            colors[i] = [reflectivity[i], reflectivity[i], reflectivity[i]]
        processed_cloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        colors = np.asarray(processed_cloud.colors)
        reflectivity = colors[:, 0]
    low_reflectivity_mask = reflectivity < reflectivity_threshold
    points_to_keep = []
    colors_to_keep = []

    for i in range(len(points)):
        if low_reflectivity_mask[i]:
            if np.random.random() > removal_probability:
                points_to_keep.append(points[i])
                colors_to_keep.append(np.asarray(processed_cloud.colors)[i])
        else:
            points_to_keep.append(points[i])
            colors_to_keep.append(np.asarray(processed_cloud.colors)[i])

    result_cloud = o3d.geometry.PointCloud()
    result_cloud.points = o3d.utility.Vector3dVector(np.array(points_to_keep))
    result_cloud.colors = o3d.utility.Vector3dVector(np.array(colors_to_keep))

    return result_cloud


def add_noise_to_point_cloud(point_cloud, noise_level=0.01):
    points = np.asarray(point_cloud.points)
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    noisy_cloud.points = o3d.utility.Vector3dVector(noisy_points)
    return noisy_cloud


def visualize_comparison(original_cloud, processed_cloud):
    original_cloud_moved = original_cloud.clone()
    processed_cloud_moved = processed_cloud.clone()
    original_center = original_cloud.get_center()
    original_cloud_moved.translate([-original_center[0] - 1, 0, 0])
    processed_cloud_moved.translate([-original_center[0] + 1, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Compare")
    vis.add_geometry(original_cloud_moved)
    vis.add_geometry(processed_cloud_moved)
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    vis.run()
    vis.destroy_window()

pcd = o3d.io.read_point_cloud("/home/eryao/code/point/data/lidar/sofa/sofa_0001_lidar0.ply") 
colors = np.zeros((len(pcd.points), 3))
points = np.asarray(pcd.points)
y_values = points[:, 1]
y_min, y_max = np.min(y_values), np.max(y_values)
for i in range(len(points)):
    reflectivity = (y_values[i] - y_min) / (y_max - y_min)
    colors[i] = [reflectivity, reflectivity, reflectivity]
pcd.colors = o3d.utility.Vector3dVector(colors)
noisy_pcd = add_noise_to_point_cloud(pcd, noise_level=0.02)
processed_pcd = simulate_low_reflectivity(
    noisy_pcd,
    reflectivity_threshold=0.4,
    removal_probability=0.8
)
visualize_comparison(noisy_pcd, processed_pcd)