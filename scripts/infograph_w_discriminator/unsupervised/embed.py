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
from torch_geometric.data import DataLoader

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score

from main import InfoGraph
    
############################################################################################

#Taking arguments from user
#python autoencoder.py --use_input_encoder False --encoder_features 495 --activation ReLU --layer gat --dropout 0.001 -lr 
#0.000009 --weight_decay 0.006
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--model_path", type=str, default="../models/liver_mesh.ply_infograph.pt")
    parser.add_argument("--meshes_path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    parser.add_argument("--save_path", type=str, default="../../../../../../vol/aimspace/users/wyo/latent_spaces/infograph_vertices_prediction")
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    # parser.add_argument("--subject", type=str, default="1000071")
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    args = parser.parse_args()
    return args
    
############################################################################################
def get_single_subject(path, organ, subject, save=False):
    registered_mesh = []    
    
    mesh = o3d.io.read_triangle_mesh(f'{path}{subject}/{organ}')

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
    data = Data(x=registered_mesh[0][0], y=registered_mesh[0][0].flatten(), edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))

    return data

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
    dirs = next(os.walk(args.meshes_path))[1]

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
        graph = get_single_subject(args.meshes_path, args.organ, str(dir))
        graph = graph.to(device)
        batch = torch.from_numpy(np.asarray([0]))
        batch = batch.to(device)
        dataloader = DataLoader([graph], batch_size=1)

        # y, M = model.embed(graph.x, graph.edge_index, batch)  
        emb, y = model.encoder.get_embeddings(dataloader)

        #Comment if pool
        # rep, pred = model.decode(dgl_graph, z)
        # pred = pred.detach().cpu().numpy()
        # rep = rep.detach().cpu().numpy()

        # emb = emb.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        # M = M.detach().cpu().numpy()

        #WRITING
        #z if pool
        with open(f'{args.save_path}/{args.organ[:-9]}/{dir}', "wb") as fp:
            pkl.dump(emb, fp)
        fp.close()
        
    print("---------------------------------------------", flush=True)