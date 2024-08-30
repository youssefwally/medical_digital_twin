#imports
import os
import argparse
import ast
import pickle as pkl
from itertools import tee

import wandb
import numpy as np
import open3d as o3d

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE, LayerNorm, Linear
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
    
############################################################################################

#Taking arguments from user
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/2000/1717978/") #optimised on 1000180, female opt test 3227250, male opt test 1853017
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args
    
############################################################################################

#Data preprocessing
def process_data(organ, path, save):
    registered_mesh = []
    mesh = o3d.io.read_triangle_mesh(os.path.join(path, organ))

    vertices_data = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices = torch.from_numpy(vertices_data).double()
    edges = []
    for triangle in triangles:
        edges.append([triangle[0], triangle[1]])
        edges.append([triangle[0], triangle[2]])
        edges.append([triangle[1], triangle[2]])
        
    edges_torch = [[],[]]
    edges =np.unique(np.array(edges), axis=0)
    for edge in edges:
        edges_torch[0].append(edge[0])
        edges_torch[1].append(edge[1])

    edges_torch = torch.from_numpy(np.asarray(edges_torch)).long()

    registered_mesh.append((vertices.type(torch.float32), edges_torch))
    data = Data(x=registered_mesh[0][0], y=registered_mesh[0][1], edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
    all_edge_index = data.edge_index
    data = train_test_split_edges(data)
    # print(data)

    if(save):
        with open('../data/gae/liver/data', 'wb') as f:
            pkl.dump(data, f)

    return data

############################################################################################

if __name__ == '__main__':
    args = build_args()

    process_data(args.organ, args.path, args.save)
    