import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

def load_point_cloud(file_path, downsample_voxel_size=0.05):
    pcd = o3d.io.read_point_cloud(file_path)
    points_raw = np.asarray(pcd.points)
    centroid = np.mean(points_raw, axis=0)
    max_bound = np.max(points_raw, axis=0)
    min_bound = np.min(points_raw, axis=0)
    scale = np.linalg.norm(max_bound - min_bound)
    if downsample_voxel_size > 0:
        bbox_size = max_bound - min_bound
        min_dimension = np.min(bbox_size[bbox_size > 0])
        adaptive_voxel_size = max(downsample_voxel_size, min_dimension / 1000)
        num_points = len(points_raw)
        if num_points > 1000000:
            adaptive_voxel_size = max(adaptive_voxel_size, min_dimension / 500)
    points = np.asarray(pcd.points)
    return points, pcd, centroid, scale

def normalize_point_cloud(points, centroid=None, scale=None):
    if centroid is None:
        centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    if scale is None:
        max_bound = np.max(centered_points, axis=0)
        min_bound = np.min(centered_points, axis=0)
        scale = np.max(np.linalg.norm(max_bound - min_bound))
        if scale < 1e-10:
            scale = 1.0
    normalized_points = centered_points / scale
    return normalized_points, centroid, scale


def denormalize_point_cloud(normalized_points, centroid, scale):
    return normalized_points * scale + centroid


def apply_random_transform(points, max_rotation=np.pi / 4, max_translation=0.5):
    angles = np.random.uniform(-max_rotation, max_rotation, size=3)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    t = np.random.uniform(-max_translation, max_translation, size=3)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (transform @ homogeneous.T).T[:, :3]
    return transformed_points, transform


class NormalizedPointCloudDataset(Dataset):
    def __init__(self, source_files, target_files=None, num_samples=1024, augment=True, normalize=True):
        self.source_files = source_files
        self.target_files = target_files if target_files else source_files
        self.num_samples = num_samples
        self.augment = augment
        self.normalize = normalize
        self.normalization_params = {}

    def __len__(self):
        return len(self.source_files)


    def __getitem__(self, idx):
        source_path = self.source_files[idx]
        source_points, source_pcd, source_centroid, source_scale = load_point_cloud(source_path)
        if self.source_files[idx] == self.target_files[idx]:
            target_points = source_points.copy()
            source_transformed, transform_gt = apply_random_transform(source_points)
            target_centroid, target_scale = source_centroid, source_scale
        else:
            target_path = self.target_files[idx]
            target_points, target_pcd, target_centroid, target_scale = load_point_cloud(target_path)
            source_transformed = source_points
            transform_gt = np.eye(4)
        self.normalization_params[idx] = {
            'source_centroid': source_centroid,
            'source_scale': source_scale,
            'target_centroid': target_centroid,
            'target_scale': target_scale
        }
        if self.normalize:
            source_points, _, _ = normalize_point_cloud(source_points, source_centroid, source_scale)
            target_points, _, _ = normalize_point_cloud(target_points, target_centroid, target_scale)
            if self.source_files[idx] == self.target_files[idx]:
                source_transformed, _, _ = normalize_point_cloud(source_transformed, source_centroid, source_scale)
                source_homogeneous = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
                if source_points.shape[0] >= 3:
                    source_pcd_norm = o3d.geometry.PointCloud()
                    source_pcd_norm.points = o3d.utility.Vector3dVector(source_points)

                    transformed_pcd_norm = o3d.geometry.PointCloud()
                    transformed_pcd_norm.points = o3d.utility.Vector3dVector(source_transformed)
                    result = o3d.pipelines.registration.registration_icp(
                        source_pcd_norm, transformed_pcd_norm, 0.05,
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint()
                    )
                    transform_gt = result.transformation
                else:
                    transform_gt = np.eye(4)
            else:
                source_transformed, _, _ = normalize_point_cloud(source_transformed, source_centroid, source_scale)
        if source_points.shape[0] > self.num_samples:
            idx = np.random.choice(source_points.shape[0], self.num_samples, replace=False)
            source_points = source_points[idx]
        elif source_points.shape[0] < self.num_samples:
            idx = np.random.choice(source_points.shape[0], self.num_samples - source_points.shape[0], replace=True)
            source_points = np.vstack([source_points, source_points[idx]])
        if target_points.shape[0] > self.num_samples:
            idx = np.random.choice(target_points.shape[0], self.num_samples, replace=False)
            target_points = target_points[idx]
        elif target_points.shape[0] < self.num_samples:
            idx = np.random.choice(target_points.shape[0], self.num_samples - target_points.shape[0], replace=True)
            target_points = np.vstack([target_points, target_points[idx]])
        if source_transformed.shape[0] > self.num_samples:
            idx = np.random.choice(source_transformed.shape[0], self.num_samples, replace=False)
            source_transformed = source_transformed[idx]
        elif source_transformed.shape[0] < self.num_samples:
            idx = np.random.choice(source_transformed.shape[0], self.num_samples - source_transformed.shape[0],
                                   replace=True)
            source_transformed = np.vstack([source_transformed, source_transformed[idx]])
        if self.augment:
            noise_level = 0.01 if self.normalize else 0.01 * source_scale
            source_points += np.random.normal(0, noise_level, size=source_points.shape)
            target_points += np.random.normal(0, noise_level, size=target_points.shape)
            source_transformed += np.random.normal(0, noise_level, size=source_transformed.shape)
        data = {
            'source_points': torch.FloatTensor(source_points),
            'target_points': torch.FloatTensor(target_points),
            'source_transformed': torch.FloatTensor(source_transformed),
            'transform_gt': torch.FloatTensor(transform_gt)
        }
        data['normalization_params'] = {
            'source_centroid': torch.FloatTensor(source_centroid),
            'source_scale': torch.tensor(source_scale, dtype=torch.float),
            'target_centroid': torch.FloatTensor(target_centroid),
            'target_scale': torch.tensor(target_scale, dtype=torch.float)
        }
        return data
    def get_normalization_params(self, idx):
        return self.normalization_params.get(idx, None)

def denormalize_prediction(pred_points, source_transform, target_transform=None):
    if target_transform is None:
        target_transform = source_transform
    source_centroid, source_scale = source_transform
    target_centroid, target_scale = target_transform
    return pred_points * source_scale + source_centroid

def adjust_transform_matrix(pred_transform, source_params, target_params):
    source_centroid, source_scale = source_params
    target_centroid, target_scale = target_params
    T_target_norm_to_orig = np.eye(4)
    T_target_norm_to_orig[:3, :3] *= target_scale
    T_target_norm_to_orig[:3, 3] = target_centroid
    T_source_orig_to_norm = np.eye(4)
    T_source_orig_to_norm[:3, :3] *= 1.0 / source_scale
    T_source_orig_to_norm[:3, 3] = -source_centroid / source_scale
    adjusted_transform = T_target_norm_to_orig @ pred_transform @ T_source_orig_to_norm

    return adjusted_transform

