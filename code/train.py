import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from datetime import datetime

from model import EnhancedPointCloudRegistration
from loss import CombinedRegistrationLoss
from dataset import NormalizedPointCloudDataset


def train(model, train_loader, val_loader, args):
    device = torch.device(args.device)
    model = model.to(device)
    loss_fn = CombinedRegistrationLoss(
        chamfer_weight=args.chamfer_weight,
        transform_weight=args.transform_weight,
        rotation_weight=args.rotation_weight,
        translation_weight=args.translation_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resumed training from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    print(f"Start training from epoch {start_epoch} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_train_losses = {'total_loss': 0.0, 'chamfer_loss': 0.0,
                              'transform_loss': 0.0, 'transform_chamfer_loss': 0.0}

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, batch_data in enumerate(pbar):
            source_points = batch_data['source_points'].to(device)
            target_points = batch_data['target_points'].to(device)
            gt_transform = batch_data['transform_gt'].to(device)
            pred_transform, selected_source, selected_target, _, _, _ = model(source_points, target_points)
            losses = loss_fn(pred_transform, gt_transform, source_points, target_points)
            total_loss = losses['total_loss']
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'chamfer': f"{losses['chamfer_loss'].item():.4f}",
                'transform': f"{losses['transform_loss'].item():.4f}"
            })
            for k, v in losses.items():
                epoch_train_losses[k] += v.item()
        for k in epoch_train_losses.keys():
            epoch_train_losses[k] /= len(train_loader)

        train_losses.append(epoch_train_losses['total_loss'])
        model.eval()
        epoch_val_losses = {'total_loss': 0.0, 'chamfer_loss': 0.0,
                            'transform_loss': 0.0, 'transform_chamfer_loss': 0.0}

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}")
            for batch_idx, batch_data in enumerate(pbar):
                source_points = batch_data['source_points'].to(device)
                target_points = batch_data['target_points'].to(device)
                gt_transform = batch_data['transform_gt'].to(device)
                pred_transform, selected_source, selected_target, _, _, _= model(source_points, target_points)
                losses = loss_fn(pred_transform, gt_transform, source_points, target_points)
                pbar.set_postfix({
                    'val_loss': f"{losses['total_loss'].item():.4f}",
                    'val_chamfer': f"{losses['chamfer_loss'].item():.4f}",
                    'val_transform': f"{losses['transform_loss'].item():.4f}"
                })
                for k, v in losses.items():
                    epoch_val_losses[k] += v.item()
        for k in epoch_val_losses.keys():
            epoch_val_losses[k] /= len(val_loader)
        val_losses.append(epoch_val_losses['total_loss'])
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {epoch_train_losses['total_loss']:.4f}, "
              f"Val Loss: {epoch_val_losses['total_loss']:.4f}")
        scheduler.step(epoch_val_losses['total_loss'])
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        if epoch_val_losses['total_loss'] < best_val_loss:
            best_val_loss = epoch_val_losses['total_loss']
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
    plt.close()

    print("Training completed!")
    return model, best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration Training and Inference')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--k_correspondences', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--chamfer_weight', type=float, default=1.0)
    parser.add_argument('--transform_weight', type=float, default=0.1)
    parser.add_argument('--rotation_weight', type=float, default=1.0)
    parser.add_argument('--translation_weight', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--test_output', type=str, default='test_results')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    args.data_dir = "/home/eryao/code/point/data/tmp"
    args.save_dir = "/home/eryao/code/point/data/checkpoints"
    args.batch_size = 2
    args.lr = 1e-3
    args.transformer_layers = 8
    args.epochs = 30
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    args.device = device
    model = EnhancedPointCloudRegistration(
        feature_dim=args.feature_dim,
        transformer_layers=args.transformer_layers,
        k_correspondences=args.k_correspondences
    )
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.ply")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pcd")))
    np.random.shuffle(all_files)
    train_ratio = 0.8
    train_size = int(len(all_files) * train_ratio)

    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    train_dataset = NormalizedPointCloudDataset(
        source_files=train_files,
        num_samples=args.num_points,
        augment=True,
        normalize=True
    )

    val_dataset = NormalizedPointCloudDataset(
        source_files=val_files,
        num_samples=args.num_points,
        augment=False,
        normalize=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    model, best_val_loss = train(model, train_loader, val_loader, args)
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()