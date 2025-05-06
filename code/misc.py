import torch
import os
from tqdm import tqdm
from logger import logger
from loss import attention_weighted_chamfer_distance

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None):
    """加载检查点"""
    if not os.path.exists(filename):
        logger.warning(f"Checkpoint {filename} not found!")
        return None, 0

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))

    logger.info(f"Checkpoint loaded from {filename} (epoch {epoch})")
    return model, epoch

def evaluate(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    total_rotation_error = 0.0
    total_translation_error = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            source_points = batch['source_points'].to(device)
            target_points = batch['target_points'].to(device)
            source_transformed = batch['source_transformed'].to(device)
            transform_gt = batch['transform_gt'].to(device)
            pred_transform, selected_source, selected_target, cross_attn = model(source_transformed, target_points)
            source_pred = model.apply_transform(source_transformed, pred_transform)
            weighted_cd_loss = attention_weighted_chamfer_distance(source_pred, target_points, cross_attn)
            R_pred = pred_transform[:, :3, :3]
            R_gt = transform_gt[:, :3, :3]
            t_pred = pred_transform[:, :3, 3]
            t_gt = transform_gt[:, :3, 3]

            rotation_error = torch.norm(R_pred - R_gt, dim=(1, 2)).mean().item()
            translation_error = torch.norm(t_pred - t_gt, dim=1).mean().item()

            total_loss += weighted_cd_loss.item()
            total_rotation_error += rotation_error
            total_translation_error += translation_error
            batch_count += 1

    avg_loss = total_loss / batch_count
    avg_rotation_error = total_rotation_error / batch_count
    avg_translation_error = total_translation_error / batch_count

    logger.info(f"Evaluation - Avg Loss: {avg_loss:.4f}, "
                f"Rotation Error: {avg_rotation_error:.4f}, "
                f"Translation Error: {avg_translation_error:.4f}")

    return avg_loss, avg_rotation_error, avg_translation_error