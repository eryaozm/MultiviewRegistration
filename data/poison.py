import open3d as o3d
import numpy as np
import os

def poisson_reconstruction(pcd_path, output_path, depth=8, width=0, scale=1.1, linear_fit=False):
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])
    if len(pcd.normals) == 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
    densities = np.asarray(densities)
    density_colors = plt_colormap(densities, "jet")
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.io.write_triangle_mesh(output_path, mesh)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([density_mesh], mesh_show_back_face=True)
    return mesh

def plt_colormap(values, colormap_name):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    norm = plt.Normalize(values.min(), values.max())
    colormap = cm.get_cmap(colormap_name)
    colors = colormap(norm(values))[:, :3]
    return colors

if __name__ == "__main__":
    input_file = "/home/eryao/code/point/data/truck_pc.ply"
    output_file = "/home/eryao/code/point/data/truck_m.ply"
    poisson_mesh = poisson_reconstruction(
        input_file,
        output_file,
        depth=10,
        width=0,
        scale=1.1,
        linear_fit=True
    )
