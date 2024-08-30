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
import wandb

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
    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/2000/")
    parser.add_argument("--organ", type=str, default="right_kidney_mesh.ply")

    
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
    # Extract points and adjacency information
    points = open3d_geometry.vertices
    adjacency_matrix = triangle_mesh_to_adjacency_matrix(open3d_geometry)
    # Create a DGL graph from the adjacency matrix
    dgl_graph = dgl.from_scipy(adjacency_matrix)

    # Add node features (e.g., point coordinates) to the DGL graph
    dgl_graph.ndata['feat'] = torch.tensor(points, dtype=torch.float32)

    return dgl_graph

#############################################################################################################

def evaluate(model, dataloader, device):
    logging.info("start test..")

    model.eval()
    test_loss_list = []
    for subgraph in dataloader:
        subgraph = subgraph.to(device)
        loss, loss_dict = model(subgraph, subgraph.ndata["feat"])
        test_loss_list.append(loss.item())
    
    mean_test_loss = np.mean(test_loss_list)
    median_test_loss = np.median(test_loss_list)

    wandb.log({'mean_test_loss': mean_test_loss, 'median_test_loss': median_test_loss})
    wandb.watch(model)

    return model

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

    for dir in train_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        dgl_graph = open3d_to_dgl_graph(mesh)
        dgl_graph = dgl_graph.remove_self_loop()
        dgl_graph = dgl_graph.add_self_loop()
        train_graphs.append(dgl_graph)

    for dir in val_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        dgl_graph = open3d_to_dgl_graph(mesh)
        dgl_graph = dgl_graph.remove_self_loop()
        dgl_graph = dgl_graph.add_self_loop()
        val_graphs.append(dgl_graph)

    for dir in test_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        dgl_graph = open3d_to_dgl_graph(mesh)
        dgl_graph = dgl_graph.remove_self_loop()
        dgl_graph = dgl_graph.add_self_loop()
        test_graphs.append(dgl_graph)

    all_graphs = train_graphs + val_graphs + test_graphs
    return all_graphs

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

    run = wandb.init(
        project="digital_twin_graphmae",
        entity="yussufwaly",
        notes="GAE",
        tags=[],
        config=args,
        )
    
    graphs = get_data(wandb.config.path, wandb.config.organ)
 
    scheduler = None
    logger = None
    wandb.config.update( {'device': device }, allow_val_change=True)

    #Model
    model = torch.load(wandb.config.model_path, map_location=torch.device('cpu'))
    model = model.to(device)

    model = evaluate(model, graphs, device)
