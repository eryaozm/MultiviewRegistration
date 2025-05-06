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
from loss import ChamferDistanceLoss
from dataset import NormalizedPointCloudDataset

def batch_inference(model, test_loader, device='cuda'):
    model.eval()
    results = []
    chamfer_distances = []
    chamfer_loss_fn = ChamferDistanceLoss()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Inference")):
            source_points = batch_data['source_points'].to(device)
            target_points = batch_data['target_points'].to(device)
            gt_transform = batch_data.get('transform_gt')
            pred_transform, selected_source, selected_target, _, _, _ = model(source_points, target_points)
            transformed_source = model.apply_transform(source_points, pred_transform)
            chamfer_dist = chamfer_loss_fn(transformed_source, target_points)
            chamfer_distances.extend(chamfer_dist.cpu().numpy())
            for i in range(source_points.size(0)):
                result = {
                    'source_points': source_points[i].cpu().numpy(),
                    'target_points': target_points[i].cpu().numpy(),
                    'pred_transform': pred_transform[i].cpu().numpy(),
                    'transformed_source': transformed_source[i].cpu().numpy(),
                    'chamfer_distance': chamfer_dist[i].item()
                }

                if gt_transform is not None:
                    result['gt_transform'] = gt_transform[i].cpu().numpy()

                results.append(result)
    avg_chamfer_distance = np.mean(chamfer_distances)
    print(f"Average Chamfer Distance: {avg_chamfer_distance:.6f}")
    return results, avg_chamfer_distance
def visualize_registration(source_points, target_points, transformed_source, title=None):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.paint_uniform_color([1, 0, 0])

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.paint_uniform_color([0, 1, 0]) 

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_source)
    transformed_pcd.paint_uniform_color([0, 0, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window(title if title else "Point Cloud Registration")
    vis.add_geometry(source_pcd)
    vis.add_geometry(target_pcd)
    vis.add_geometry(transformed_pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    vis.update_geometry(source_pcd)
    vis.update_geometry(target_pcd)
    vis.update_geometry(transformed_pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration Training and Inference')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--source_dir', type=str, default='data')
    parser.add_argument('--target_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--transformer_layers', type=int, default=8)
    parser.add_argument('--k_correspondences', type=int, default=512)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--chamfer_weight', type=float, default=1.0)
    parser.add_argument('--transform_weight', type=float, default=0.1)
    parser.add_argument('--rotation_weight', type=float, default=1.0)
    parser.add_argument('--translation_weight', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')

    # 测试参数
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--test_output', type=str, default='test_results')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    args.source_dir = "/home/eryao/code/point/data/test/source"
    args.target_dir = "/home/eryao/code/point/data/test/target"
    args.save_dir = "/home/eryao/code/point/data/checkpoints"
    args.batch_size = 1
    args.visualize = True
    args.num_points = 2048
    args.transformer_layers = 8
    # args.resume = "/home/eryao/code/point/data/checkpoints/best_model.pth"
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    args.device = device
    print(f"Using device: {device}")
    model = EnhancedPointCloudRegistration(
        feature_dim=args.feature_dim,
        transformer_layers=args.transformer_layers,
        k_correspondences=args.k_correspondences
    )
    if not args.model_path:
        if os.path.exists(os.path.join(args.save_dir, "best_model.pth")):
            args.model_path = os.path.join(args.save_dir, "best_model.pth")
        else:
            raise ValueError("Model path must be specified for test mode")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    target_files = sorted(glob.glob(os.path.join(args.target_dir, "*.ply")))
    if not target_files:
        target_files = sorted(glob.glob(os.path.join(args.tartet_dir, "*.pcd")))
    if not target_files:
        raise ValueError(f"No point cloud files found in {args.tartet_dir}")
    print(f"Found {len(target_files)} target files")
    source_files = sorted(glob.glob(os.path.join(args.source_dir, "*.ply")))
    if not source_files:
        source_files = sorted(glob.glob(os.path.join(args.source_dir, "*.pcd")))

    if not source_files:
        raise ValueError(f"No point cloud files found in {args.source_dir}")

    print(f"Found {len(source_files)} source files")
    test_dataset = NormalizedPointCloudDataset(
        source_files=target_files,
        target_files=source_files,
        num_samples=args.num_points,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    os.makedirs(args.test_output, exist_ok=True)
    print("Starting inference...")
    results, avg_chamfer = batch_inference(model, test_loader, device)
    print(f"Inference completed with average Chamfer distance: {avg_chamfer:.6f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.test_output, f"results_{timestamp}.pth")
    torch.save({
        'results': results,
        'avg_chamfer_distance': avg_chamfer,
        'model_path': args.model_path,
        'timestamp': timestamp
    }, result_file)
    print(f"Results saved to {result_file}")


    if args.visualize:
        for i, result in enumerate(results[:1]):
            source_points = result['source_points']
            target_points = result['target_points']
            transformed_source = result['transformed_source']
            chamfer_dist = result['chamfer_distance']

            title = f"Registration Result {i + 1} - Chamfer Distance: {chamfer_dist:.6f}"
            visualize_registration(source_points, target_points, transformed_source, title)

def infer(source_path,target_path):
    test_dataset = NormalizedPointCloudDataset(
        source_files=[source_path],
        target_files=[target_path],
        num_samples=2048,
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    model = EnhancedPointCloudRegistration(
        feature_dim=256,
        transformer_layers=8,
        k_correspondences=512
    )
    checkpoint = torch.load("/home/eryao/code/point/data/checkpoints/best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to("cuda")
    results, _ = batch_inference(model, test_loader, "cuda")
    pred_transform = results[0].get('pred_transform')
    return pred_transform

if __name__ == '__main__':
    main()