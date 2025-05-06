import open3d as o3d
import numpy as np
import copy
import os
import glob
import time
from scipy.optimize import least_squares
import networkx as nx

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def pairwise_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result_fast = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_fast.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = result_icp.transformation
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result_icp.transformation)

    return transformation, information

def build_pose_graph(pcds, fpfhs, voxel_size):
    n_pcds = len(pcds)
    pose_graph = o3d.pipelines.registration.PoseGraph()

    for i in range(n_pcds):
        if i == 0:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
        else:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
    for i in range(n_pcds):
        for j in range(i + 1, n_pcds):
            transformation, information = pairwise_registration(
                pcds[i], pcds[j], fpfhs[i], fpfhs[j], voxel_size)
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                i, j, transformation, information, uncertain=False))
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                j, i, np.linalg.inv(transformation), information, uncertain=False))

    return pose_graph

def optimize_pose_graph(pose_graph, max_iterations=100):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        reference_node=0,
        preference_loop_closure=100.0)
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        criteria,
        option)
    return pose_graph

def create_spanning_tree_graph(pcds, fpfhs, voxel_size):
    n_pcds = len(pcds)
    pose_graph = o3d.pipelines.registration.PoseGraph()

    for i in range(n_pcds):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
    G = nx.Graph()
    for i in range(n_pcds):
        for j in range(i + 1, n_pcds):
            source_fpfh_data = np.asarray(fpfhs[i].data)
            target_fpfh_data = np.asarray(fpfhs[j].data)
            source_feature_dim = source_fpfh_data.shape[1]
            target_feature_dim = target_fpfh_data.shape[1]

            if source_feature_dim != target_feature_dim:
                min_dim = min(source_feature_dim, target_feature_dim)
                source_fpfh_data = source_fpfh_data[:, :min_dim]
                target_fpfh_data = target_fpfh_data[:, :min_dim]
            source_norm = np.linalg.norm(source_fpfh_data, axis=1, keepdims=True)
            target_norm = np.linalg.norm(target_fpfh_data, axis=1, keepdims=True)
            source_norm[source_norm == 0] = 1.0
            target_norm[target_norm == 0] = 1.0
            source_normalized = source_fpfh_data / source_norm
            target_normalized = target_fpfh_data / target_norm
            source_mean = np.mean(source_normalized, axis=0)
            target_mean = np.mean(target_normalized, axis=0)
            cos_sim = np.dot(source_mean, target_mean) / (np.linalg.norm(source_mean) * np.linalg.norm(target_mean))
            mean_dist = 1.0 - cos_sim

            G.add_edge(i, j, weight=mean_dist)
    mst = nx.minimum_spanning_tree(G)

    for i, j in mst.edges():
        transformation, information = pairwise_registration(
            pcds[i], pcds[j], fpfhs[i], fpfhs[j], voxel_size)
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i, j, transformation, information, uncertain=False))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            j, i, np.linalg.inv(transformation), information, uncertain=False))
    for i in range(n_pcds):
        for j in range(i + 1, n_pcds):
            if not mst.has_edge(i, j) and G[i][j]['weight'] < np.mean([G[e[0]][e[1]]['weight'] for e in mst.edges()]):
                transformation, information = pairwise_registration(
                    pcds[i], pcds[j], fpfhs[i], fpfhs[j], voxel_size)

                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    i, j, transformation, information, uncertain=False))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    j, i, np.linalg.inv(transformation), information, uncertain=False))

    return pose_graph

def multi_view_registration(point_cloud_files, voxel_size=0.01, use_spanning_tree=True, max_iterations=100):
    pcds = []
    for file in point_cloud_files:
        pcd = o3d.io.read_point_cloud(file)
        pcds.append(pcd)
    pcds_down = []
    fpfhs = []
    for i, pcd in enumerate(pcds):
        pcd_down, fpfh = preprocess_point_cloud(pcd, voxel_size)
        pcds_down.append(pcd_down)
        fpfhs.append(fpfh)
    if use_spanning_tree:
        pose_graph = create_spanning_tree_graph(pcds_down, fpfhs, voxel_size)
    else:
        pose_graph = build_pose_graph(pcds_down, fpfhs, voxel_size)
    optimized_pose_graph = optimize_pose_graph(pose_graph, max_iterations)
    pcds_transformed = []
    for i, pcd in enumerate(pcds):
        pcd_transformed = copy.deepcopy(pcd)
        pcd_transformed.transform(optimized_pose_graph.nodes[i].pose)
        pcds_transformed.append(pcd_transformed)
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds_transformed:
        merged_pcd += pcd
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return merged_pcd, pcds_transformed


def visualize_registration_result(merged_pcd, pcds_transformed, show_original=False):
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0.5, 0.5, 0.5],
        [1, 0.5, 0],
        [0.5, 0, 1],
        [0, 0.5, 0.5]
    ]

    for i, pcd in enumerate(pcds_transformed):
        color = colors[i % len(colors)]
        pcd.paint_uniform_color(color)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="多视角点云配准结果")
    if show_original:
        for pcd in pcds_transformed:
            vis.add_geometry(pcd)
    else:
        vis.add_geometry(merged_pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 黑色背景
    opt.point_size = 1.0
    vis.update_geometry(merged_pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    input_path = "/home/eryao/code/point/data/order"
    if os.path.isdir(input_path):
        point_cloud_files = sorted(glob.glob(os.path.join(input_path, "*.ply")))
    else:
        point_cloud_files = [
            "path/to/cloud1.ply",
            "path/to/cloud2.ply",
            "path/to/cloud3.ply",
        ]
    voxel_size = 0.01
    use_spanning_tree = True
    max_iterations = 100
    start_time = time.time()
    merged_pcd, pcds_transformed = multi_view_registration(
        point_cloud_files, voxel_size, use_spanning_tree, max_iterations)
    end_time = time.time()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    merged_pcd_path = os.path.join(output_dir, "merged_point_cloud.ply")
    o3d.io.write_point_cloud(merged_pcd_path, merged_pcd)
    for i, pcd in enumerate(pcds_transformed):
        pcd_path = os.path.join(output_dir, f"transformed_cloud_{i}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
    visualize_registration_result(merged_pcd, pcds_transformed, show_original=False)
    visualize_registration_result(merged_pcd, pcds_transformed, show_original=True)