import open3d as o3d
import numpy as np

point_cloud = o3d.io.read_point_cloud("/home/eryao/code/point/data/truck_pc.ply")
if not point_cloud.has_normals():
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    point_cloud.orient_normals_consistent_tangent_plane(k=20)
o3d.visualization.draw_geometries([point_cloud])
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    point_cloud, depth=10, width=0, scale=1.1, linear_fit=False
)[0]
vertices_to_remove = np.where(np.asarray(mesh.vertex_colors)[:, 0] < 0.1)[0]
mesh.remove_vertices_by_index(vertices_to_remove)
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()

mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("/home/eryao/code/point/data/truck_m.ply", mesh)

def display_comparison(point_cloud, mesh):
    pc_temp = o3d.geometry.PointCloud(point_cloud)
    pc_temp.paint_uniform_color([1, 0, 0])
    mesh.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pc_temp, mesh])
