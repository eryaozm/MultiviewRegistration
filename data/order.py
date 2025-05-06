import os
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_point_clouds(folder_path):
    point_clouds = []
    filenames = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.ply'):
            file_path = os.path.join(folder_path, file)
            pc = o3d.io.read_point_cloud(file_path)
            if len(pc.points) == 0:
                print(f"{file} is None.")
                continue
            point_clouds.append(pc)
            filenames.append(file)
    return point_clouds, filenames

def preprocess_point_cloud(pcd, voxel_size=0.05):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down


def extract_fpfh_features(pcd, voxel_size=0.05):
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def compute_similarity_matrix(point_clouds, features):
    n = len(point_clouds)
    similarity_matrix = np.zeros((n, n))
    correspondence_sets = {}
    for i in tqdm(range(n)):
        for j in range(n):
            if i == j:
                continue
            result = global_registration(
                point_clouds[i], point_clouds[j],
                features[i], features[j])
            similarity_matrix[i, j] = len(result.correspondence_set)
            correspondence_sets[(i, j)] = result.correspondence_set
    return similarity_matrix, correspondence_sets

def global_registration(source, target, source_feat, target_feat, distance_threshold=0.05):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feat, target_feat,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    return result


def find_optimal_order(similarity_matrix):
    max_sim = np.max(similarity_matrix)
    cost_matrix = max_sim - similarity_matrix
    np.fill_diagonal(cost_matrix, 0)
    G = nx.DiGraph()
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=cost_matrix[i, j])
    path = nx.approximation.traveling_salesman_problem(G, cycle=True)

    return path[:-1]
def refine_registration(source, target, init_transform=np.eye(4), distance_threshold=0.05):
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return result
def evaluate_registration(source, target, transformation, threshold=0.05):
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    dists = np.asarray(source_transformed.compute_point_cloud_distance(target))
    overlap_ratio = np.sum(dists < threshold) / len(dists)
    return overlap_ratio, np.mean(dists)


def visualize_registration(source, target, transformation=np.eye(4)):
    source_copy = copy.deepcopy(source)
    source_copy.transform(transformation)
    source_copy.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([source_copy, target])

def visualize_ordered_point_clouds(point_clouds, optimal_order, transformations=None):
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [0.5, 0.5, 0], [0.5, 0, 0.5]
    ]
    vis_pcd = []
    for i, idx in enumerate(optimal_order):
        pcd_copy = copy.deepcopy(point_clouds[idx])
        if transformations is not None and i > 0:
            cumulative_transform = np.eye(4)
            for j in range(i):
                from_idx = optimal_order[j]
                to_idx = optimal_order[(j + 1) % len(optimal_order)]
                cumulative_transform = np.dot(transformations[(from_idx, to_idx)], cumulative_transform)
            pcd_copy.transform(cumulative_transform)
        pcd_copy.paint_uniform_color(colors[i % len(colors)])
        vis_pcd.append(pcd_copy)
    o3d.visualization.draw_geometries(vis_pcd)


def main(folder_path):
    point_clouds, filenames = load_point_clouds(folder_path)
    n_clouds = len(point_clouds)
    processed_clouds = []
    features = []
    for i, pcd in enumerate(point_clouds):
        pcd_down = preprocess_point_cloud(pcd)
        processed_clouds.append(pcd_down)
        fpfh = extract_fpfh_features(pcd_down)
        features.append(fpfh)
    similarity_matrix, correspondence_sets = compute_similarity_matrix(processed_clouds, features)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Correspondence Count')
    plt.title('Point Cloud Similarity Matrix')
    plt.xlabel('Point Cloud Index')
    plt.ylabel('Point Cloud Index')
    plt.savefig('similarity_matrix.png')
    plt.close()
    optimal_order = find_optimal_order(similarity_matrix)
    print(f"{optimal_order}")

if __name__ == "__main__":
    import sys
    import copy

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "/home/eryao/code/point/data/order"

    main(folder_path)