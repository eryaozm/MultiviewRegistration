import torch
import torch.nn as nn
from typing import Dict
class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, source_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        assert source_points.dim() == 3 and target_points.dim() == 3
        assert source_points.shape[2] == 3 and target_points.shape[2] == 3
        batch_size = source_points.shape[0]
        source_expanded = source_points.unsqueeze(2)
        target_expanded = target_points.unsqueeze(1)
        dist = torch.sum((source_expanded - target_expanded) ** 2, dim=3)
        min_dist_source_to_target, _ = torch.min(dist, dim=2)
        min_dist_target_to_source, _ = torch.min(dist, dim=1)
        chamfer_dist = torch.mean(min_dist_source_to_target, dim=1) + torch.mean(min_dist_target_to_source, dim=1)
        return chamfer_dist

class TransformationLoss(nn.Module):
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        super(TransformationLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(self, pred_transform: torch.Tensor, gt_transform: torch.Tensor) -> torch.Tensor:
        pred_R = pred_transform[:, :3, :3]
        gt_R = gt_transform[:, :3, :3]
        pred_t = pred_transform[:, :3, 3]
        gt_t = gt_transform[:, :3, 3]
        rotation_loss = torch.norm(pred_R - gt_R, dim=(1, 2))
        translation_loss = torch.norm(pred_t - gt_t, dim=1)
        total_loss = self.rotation_weight * rotation_loss + self.translation_weight * translation_loss
        return total_loss

class CombinedRegistrationLoss(nn.Module):
    def __init__(self, chamfer_weight=1.0, transform_weight=1.0,
                 rotation_weight=1.0, translation_weight=1.0):
        super(CombinedRegistrationLoss, self).__init__()
        self.chamfer_loss = ChamferDistanceLoss()
        self.transform_loss = TransformationLoss(rotation_weight, translation_weight)
        self.chamfer_weight = chamfer_weight
        self.transform_weight = transform_weight

    def forward(self, pred_transform: torch.Tensor, gt_transform: torch.Tensor,
                source_points: torch.Tensor, target_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = source_points.shape[0]
        ones = torch.ones(batch_size, source_points.shape[1], 1, device=source_points.device)
        source_homogeneous = torch.cat([source_points, ones], dim=2)
        pred_transformed_source = torch.bmm(source_homogeneous, pred_transform.transpose(1, 2))[:, :, :3]
        gt_transformed_source = torch.bmm(source_homogeneous, gt_transform.transpose(1, 2))[:, :, :3]
        chamfer_loss = self.chamfer_loss(pred_transformed_source, target_points)
        transform_loss = self.transform_loss(pred_transform, gt_transform)
        transform_chamfer_loss = self.chamfer_loss(pred_transformed_source, gt_transformed_source)
        total_loss = (self.chamfer_weight * chamfer_loss +
                      self.transform_weight * transform_loss +
                      0.5 * transform_chamfer_loss)
        return {
            'total_loss': total_loss.mean(),
            'chamfer_loss': chamfer_loss.mean(),
            'transform_loss': transform_loss.mean(),
            'transform_chamfer_loss': transform_chamfer_loss.mean()
        }
