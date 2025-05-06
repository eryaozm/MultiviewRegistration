import numpy as np
from sklearn.neighbors import KDTree
import random
import time
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex = ply_data['vertex']

    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])

    points = np.column_stack((x, y, z))
    return points


def distance_between_points(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def compute_rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(R, centroid_A)

    return R, t


def get_random_base(points, distance_threshold=0.1):
    n_points = len(points)
    max_iterations = 1000
    for _ in range(max_iterations):
        idx = random.sample(range(n_points), 4)
        base = points[idx]
        valid_base = True
        for i in range(4):
            for j in range(i + 1, 4):
                dist = distance_between_points(base[i], base[j])
                if dist < distance_threshold:
                    valid_base = False
                    break
            if not valid_base:
                break
        if valid_base:
            return idx, base


def find_congruent_set(target_points, kdtree, base_points, distance_ratio_threshold=0.05):
    source_distances = []
    for i in range(4):
        for j in range(i + 1, 4):
            source_distances.append(distance_between_points(base_points[i], base_points[j]))
    n_points = len(target_points)
    max_iterations = 100
    best_score = float('inf')
    best_match = None

    for _ in range(max_iterations):
        idx = random.sample(range(n_points), 4)
        candidate = target_points[idx]
        target_distances = []
        for i in range(4):
            for j in range(i + 1, 4):
                target_distances.append(distance_between_points(candidate[i], candidate[j]))
        distance_error = 0
        for i in range(len(source_distances)):
            ratio_diff = abs(source_distances[i] / max(source_distances) -
                             target_distances[i] / max(target_distances))
            distance_error += ratio_diff
        if distance_error < best_score:
            best_score = distance_error
            best_match = idx
    if best_score > distance_ratio_threshold * 6:
        return None
    return best_match


def compute_alignment_error(source_points, target_points, R, t, max_dist=1.0):
    transformed_source = np.dot(source_points, R.T) + t
    kdtree = KDTree(target_points)
    distances, _ = kdtree.query(transformed_source, k=1)
    distances = distances.flatten()
    valid_distances = distances[distances < max_dist]
    if len(valid_distances) == 0:
        return float('inf')
    return np.mean(valid_distances)


def four_pcs(source_points, target_points, n_iterations=1000, distance_threshold=0.1, error_threshold=0.5):
    best_error = float('inf')
    best_R = None
    best_t = None
    target_kdtree = KDTree(target_points)
    start_time = time.time()

    for i in range(n_iterations):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
        try:
            source_idx, source_base = get_random_base(source_points, distance_threshold)
            target_idx = find_congruent_set(target_points, target_kdtree, source_base)
            if target_idx is None:
                continue
            target_base = target_points[target_idx]
            R, t = compute_rigid_transform(source_base, target_base)
            error = compute_alignment_error(source_points, target_points, R, t)
            if error < best_error:
                best_error = error
                best_R = R
                best_t = t
                if best_error < error_threshold:
                    break
        except Exception as e:
            continue
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = best_R
    transform_matrix[:3, 3] = best_t
    return transform_matrix, best_error


def visualize_point_clouds(source, target, transformed_source=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c='r', s=1, alpha=0.5, label='Source')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='b', s=1, alpha=0.5, label='Traget')
    if transformed_source is not None:
        ax.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2],
                   c='g', s=1, alpha=0.5, label='Transformed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Result')
    plt.show()

def use_fpcs(source_path, target_path, visualize=False):
    source_points = read_ply(source_path)
    target_points = read_ply(target_path)
    transform_matrix, error = four_pcs(
        source_points,
        target_points,
        n_iterations=500,
        distance_threshold=0.05,
        error_threshold=0.01
    )

    if visualize:
        transformed_source = np.dot(source_points, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
        visualize_point_clouds(source_points, target_points, transformed_source)
    return transform_matrix
