import open3d as o3d
import numpy as np
mesh = o3d.io.read_triangle_mesh("/home/eryao/code/point/data/truck_m.ply")
o3d.visualization.draw_geometries([mesh])