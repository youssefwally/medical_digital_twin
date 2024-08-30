#imports
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch

import open3d as o3d

import dgl
    
############################################################################################

def triangle_mesh_to_adjacency_matrix(mesh):
    # Get the vertices and triangles of the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create an empty adjacency matrix
    n_vertices = len(vertices)
    adjacency_matrix = sp.lil_matrix((n_vertices, n_vertices), dtype=np.float32)

    # Iterate through the triangles and add edges to the adjacency matrix
    for tri in triangles:
        adjacency_matrix[tri[0], tri[1]] = 1.0
        adjacency_matrix[tri[1], tri[0]] = 1.0
        adjacency_matrix[tri[1], tri[2]] = 1.0
        adjacency_matrix[tri[2], tri[1]] = 1.0
        adjacency_matrix[tri[2], tri[0]] = 1.0
        adjacency_matrix[tri[0], tri[2]] = 1.0

    # Convert the adjacency matrix to a more efficient sparse matrix representation
    adjacency_matrix = adjacency_matrix.tocsr()
    
    return adjacency_matrix

#############################################################################################################

def open3d_to_dgl_graph(path, open3d_geometry):
    intensity_path = path.replace("registered_meshes","organ_decimations_ply")
    intensity_mesh = o3d.io.read_triangle_mesh(intensity_path)
    open3d_geometry.compute_vertex_normals()

    # Extract points, normals and adjacency information
    points = open3d_geometry.vertices
    adjacency_matrix = triangle_mesh_to_adjacency_matrix(open3d_geometry)
    # Create a DGL graph from the adjacency matrix
    dgl_graph = dgl.from_scipy(adjacency_matrix)

    # Add node features (e.g., point coordinates) to the DGL graph
    points_np = np.array(open3d_geometry.vertices)
    normals_np = np.array(open3d_geometry.vertex_normals)
    intensities_np = np.array(intensity_mesh.vertex_colors)
    # features = np.concatenate((points_np, normals_np, intensities_np), axis=1)
    features = points_np
    
    dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)

    return dgl_graph

#############################################################################################################

def get_data(path, organ):
    train_ids_path = "../data/NonNa_organs_split_train.txt"
    val_ids_path = "../data/NonNa_organs_split_val.txt"
    test_ids_path = "../data/NonNa_organs_split_test.txt"

    train_dirs = np.loadtxt(train_ids_path, delimiter=",", dtype=str)
    val_dirs = np.loadtxt(val_ids_path, delimiter=",", dtype=str)
    test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)

    train_graphs = []
    val_graphs = []
    test_graphs = []

    #During Tests
    # train_dirs = train_dirs[:5]
    # val_dirs = val_dirs[:5]
    # test_dirs = test_dirs[:5]

    for dir in train_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        dgl_graph = open3d_to_dgl_graph(f'{path}{str(dir)}/{organ}',mesh)
        dgl_graph = dgl_graph.remove_self_loop()
        dgl_graph = dgl_graph.add_self_loop()
        train_graphs.append(dgl_graph)

    for dir in val_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        dgl_graph = open3d_to_dgl_graph(f'{path}{str(dir)}/{organ}', mesh)
        dgl_graph = dgl_graph.remove_self_loop()
        dgl_graph = dgl_graph.add_self_loop()
        val_graphs.append(dgl_graph)

    for dir in test_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        dgl_graph = open3d_to_dgl_graph(f'{path}{str(dir)}/{organ}', mesh)
        dgl_graph = dgl_graph.remove_self_loop()
        dgl_graph = dgl_graph.add_self_loop()
        test_graphs.append(dgl_graph)

    return train_graphs, val_graphs, test_graphs

############################################################################################