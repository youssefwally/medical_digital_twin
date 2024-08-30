#imports
import os
import argparse
import ast
import pickle as pkl
from itertools import tee

import wandb
import numpy as np
import pandas as pd
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

def get_data(path, organ, label, save=False):
    registered_mesh = []
    test_ids_path = "../data/NonNa_organs_split_test.txt"
    test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)
    dirs = next(os.walk(path))[1]
    train_dataset = []
    test_dataset = []

    #In Test
    # dirs = dirs[:500]
    body_fields = ["eid", "22407-2.0", "22408-2.0", "31-0.0"]
    full_ukbb_data = pd.read_csv("../../../../../../vol/aimspace/projects/ukbb/data/tabular/ukb668815_imaging.csv", usecols=body_fields)
    full_ukbb_data_new_names = {'22407-2.0':'VAT', '22408-2.0':'ASAT', '31-0.0':'sex'}
    full_ukbb_data = full_ukbb_data.rename(index=str, columns=full_ukbb_data_new_names)
    
    basic_features = pd.read_csv("../data/basic_features.csv")
    basic_features_new_names = {'21003-2.0':'age', '31-0.0':'sex', '21001-2.0':'bmi', '21002-2.0':'weight','50-2.0':'height'}
    basic_features = basic_features.rename(index=str, columns=basic_features_new_names)
    print(f'Number of samples used: {len(dirs)}, with label: {label}', flush=True)

    if(label == 'sex' or label == 'VAT' or label == 'ASAT'):
        features = full_ukbb_data
    else:
        features = basic_features
    
    for dir in dirs:
        registered_mesh = []
        mesh = o3d.io.read_triangle_mesh(f'{path}{dir}/{organ}')
    
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

        cur_patient_feature = features[features['eid'] == int(dir)]
        if(len(cur_patient_feature[label]) == 1):
            if(not pd.isnull(cur_patient_feature[label].item())):
                cur_patient_feature_tensor = torch.tensor(cur_patient_feature[label].item())
                registered_mesh.append((vertices.type(torch.float32), edges_torch, cur_patient_feature_tensor.type(torch.float32)))
                data = Data(x=registered_mesh[0][0], y=registered_mesh[0][2], edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
                if(dir in test_dirs):
                    test_dataset.append(data)
                else:
                    train_dataset.append(data)
                # data = train_test_split_edges(data)
                # print(data)
    
    if(save):
        with open(f'../data/infograph/{organ}/data', 'wb') as f:
            pkl.dump(train_dataset, f)

    return train_dataset, test_dataset
############################################################################################
    
