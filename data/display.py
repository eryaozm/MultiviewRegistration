import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/eryao/code/point/data/tnt/truck_scan.ply")
points = np.asarray(pcd.points)
z_min, z_max = points[:, 1].min(), points[:, 1].max()
z_normalized = (points[:, 1] - z_min) / (z_max - z_min)
colors = np.zeros((len(points), 3))
colors[:, 0] = z_normalized
colors[:, 1] = 1 - z_normalized
colors[:, 2] = 0.8
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd],
                                  window_name="vis",
                                  width=800,
                                  height=600,
                                  zoom=0.5,
                                  front=[0, -1, -0.5],
                                  lookat=pcd.get_center(),
                                  up=[0, 1, 0])