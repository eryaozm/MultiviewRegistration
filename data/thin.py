import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("/home/eryao/code/point/data/m60_mesh.ply")

def simplify_mesh(mesh, target_reduction):
    mesh_simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(len(mesh.triangles) * target_reduction)
    )
    return mesh_simplified


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh

def save_binary_ply(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh, write_ascii=False, compressed=True)
    return filename

def quantize_vertices(mesh, precision=1000):
    vertices = np.asarray(mesh.vertices)
    vertices = np.round(vertices * precision) / precision
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

mesh_reduced = simplify_mesh(mesh, 0.3)
mesh_reduced = clean_mesh(mesh_reduced)
mesh_reduced = quantize_vertices(mesh_reduced)
mesh_reduced.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_reduced])

output_file = save_binary_ply(mesh_reduced, "/home/eryao/code/point/data/m60_thinmesh.ply")