#imports
import os
import argparse
import ast
import logging
from tqdm import tqdm
import numpy as np
import random
import pickle as pkl
import pandas as pd
import scipy.sparse as sp

import open3d as o3d
# import wandb

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch_geometric
from torch_geometric.data import Data

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

from graphmae.models import build_model
from graphmae.evaluation import linear_probing_for_inductive_node_classiifcation, LogisticRegression
    
############################################################################################

#Taking arguments from user
#python autoencoder.py --use_input_encoder False --encoder_features 495 --activation ReLU --layer gat --dropout 0.001 -lr 
#0.000009 --weight_decay 0.006
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--model_path", type=str, default="../models/liver_mesh.ply_graphmae_no_feat.pt")
    parser.add_argument("--meshes_path", type=str, default="../../../../../../vol/space/projects/ukbb/projects/silhouette/gender_registered_1k/")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/body_vertices_prediction")
    # parser.add_argument("--subject", type=str, default="1000071")
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args
    
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

def open3d_to_dgl_graph(open3d_geometry):
    open3d_geometry.compute_vertex_normals()

    # Extract points, normals and adjacency information
    adjacency_matrix = triangle_mesh_to_adjacency_matrix(open3d_geometry)
    # Create a DGL graph from the adjacency matrix
    dgl_graph = dgl.from_scipy(adjacency_matrix)

    # Add node features (e.g., point coordinates) to the DGL graph
    points_np = np.array(open3d_geometry.vertices)
    normals_np = np.array(open3d_geometry.vertex_normals)
    # features = np.concatenate((points_np, normals_np), axis=1)
    features = points_np
    
    dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)

    return dgl_graph

#############################################################################################################

def get_single_subject(path, subject):

    mesh = o3d.io.read_triangle_mesh(f'{path}{str(subject)}')
    dgl_graph = open3d_to_dgl_graph(mesh)
    dgl_graph = dgl_graph.remove_self_loop()
    dgl_graph = dgl_graph.add_self_loop()

    return dgl_graph

############################################################################################

if __name__ == '__main__':
    args = build_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.determinstic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs = next(os.walk(args.meshes_path))[2]

    model = torch.load(args.model_path, map_location=torch.device('cpu'))
    model = model.to(device)

    # run = wandb.init(
    #     project="digital_twin_gae",
    #     entity="yussufwaly",
    #     notes="GAE",
    #     tags=[],
    #     config=args,
    #     )
    print(args.meshes_path)
    
    for dir in dirs:
        dgl_graph = get_single_subject(args.meshes_path, str(dir))
        dgl_graph = dgl_graph.to(device)

        z = model.embed(dgl_graph, dgl_graph.ndata['feat'], True)
        rep, pred = model.decode(dgl_graph, z)
        pred = pred.detach().cpu().numpy()
        rep = rep.detach().cpu().numpy()

        z = z.detach().cpu().numpy()
        # print(f'latent space: {z}')
        # print(f'latent space shape: {z.shape}')
    
        # print(f'prediction: {pred}')
        # print(f'prediction shape: {pred.shape}')
    
        # print(f'rep: {rep}')
        # print(f'rep shape: {rep.shape}')

        # #WRITING
        # with open('../outputs/graphmae_latent_space', "wb") as fp:
        #     pkl.dump(z, fp)
        # fp.close()
        # #WRITING
        # with open('../outputs/graphmae_pred', "wb") as fp:
        #     pkl.dump(pred, fp)
        # fp.close()
        # #WRITING
        # with open('../outputs/graphmae_rep', "wb") as fp:
        #     pkl.dump(rep, fp)
        # fp.close()

        #WRITING
        with open(f'{args.save_path}/{dir[:-4]}', "wb") as fp:
            pkl.dump(rep, fp)
        fp.close()
        
    print("---------------------------------------------", flush=True)