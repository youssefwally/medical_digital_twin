#imports
import os
import argparse
import numpy as np
from itertools import tee

import torch
import open3d as o3d
from collections import defaultdict

import torch_geometric

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE, LayerNorm, Linear

import gae_data
    
############################################################################################

def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--model_path", type=str, default="../models/liver_mesh.ply_global_gae_gat.pt")
    parser.add_argument("--id", type=str, default="4876166") #worst: 3655108, best: 4876166
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    
    
    args = parser.parse_args()
    return args
    
############################################################################################

#Iterate over all pairs of consecutive items in a list
def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

############################################################################################

#Generating GNN layers
def get_gnn_layers(num_conv_layers: int, hidden_channels, num_inp_features:int, 
                 gnn_layer, activation=nn.ReLU, normalization=None, dropout = None):
    """Creates GNN layers"""
    layers = nn.ModuleList()

    for i in range(num_conv_layers):
        if i == 0:
            layers.append(gnn_layer(num_inp_features, hidden_channels[i]))
            layers.append(activation())
            if normalization is not None:
                layers.append(normalization(hidden_channels[i]))
        else:
            layers.append(gnn_layer(hidden_channels[i-1], hidden_channels[i]))
            layers.append(activation())
            if normalization is not None:
                layers.append(normalization(hidden_channels[i]))

    return nn.ModuleList(layers)
    
############################################################################################

#Making multilayer perceptron layers 
def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    #print('Output activation ',output_activation)
    return nn.Sequential(*layers)
    
    
############################################################################################

#Encoder
class GNN(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, activation, normalization, num_conv_layers=4, layer='gcn',
                 use_input_encoder=True, encoder_features=128, apply_batch_norm=True,
                 apply_dropout_every=True, dropout = 0):
        super(GNN, self).__init__()
        torch.manual_seed(42)
        
        self.fc = torch.nn.ModuleList()
        self.layer_type = layer
        self.use_input_encoder = use_input_encoder
        self.apply_batch_norm = apply_batch_norm
        self.dropout = dropout
        self.normalization_bool = normalization
        self.activation = activation
        self.apply_dropout_every = apply_dropout_every

        if self.normalization_bool:
            self.normalization = LayerNorm
        else:
            self.normalization = None

        if self.use_input_encoder :
            self.input_encoder = get_mlp_layers(
                channels=[in_features, encoder_features],
                activation=nn.ELU,
            )
            in_features = encoder_features

        if layer == 'gcn':
            self.layers = get_gnn_layers(num_conv_layers, hidden_channels, num_inp_features=in_features,
                                        gnn_layer=GCNConv,activation=activation,normalization=self.normalization )
        elif layer == 'sageconv':
            self.layers = get_gnn_layers(num_conv_layers, hidden_channels,in_features,
                                        gnn_layer=SAGEConv,activation=activation,normalization=self.normalization )
        elif layer == 'gat':
            self.layers = get_gnn_layers(num_conv_layers, hidden_channels,in_features,
                                        gnn_layer=GATConv,activation=activation,normalization=self.normalization )        

    def forward(self, x, edge_index):

        if self.use_input_encoder:
            x = self.input_encoder(x)

        if self.normalization is None:
            for i, layer in enumerate(self.layers):
                # Each GCN consists 2 modules GCN -> Activation 
                # GCN send edge index
                if i% 2 == 0:
                    x = layer(x, edge_index)
                else:
                    x = layer(x)

                if self.apply_dropout_every:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, layer in enumerate(self.layers):
                # Each GCN consists 3 modules GCN -> Activation ->  Normalization 
                # GCN send edge index
                if i% 3 == 0:
                    x = layer(x, edge_index)
                else:
                    x = layer(x)

                if self.apply_dropout_every:
                    x = F.dropout(x, p=self.dropout, training=self.training)        

        return x
    
############################################################################################

def autoencode_patient(id, organ):    
    torch_geometric.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = "../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/2000/"
    model_path = "../models/liver_mesh.ply_global_gae_gat.pt"
    dirs = next(os.walk(path))[1]
    
    path_with_id = os.path.join(path, str(id))
    data = gae_data.process_data(organ, path_with_id, False)

    x = data.x.to(device)
    y = data.y.to(device)

    model = torch.load(model_path)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        Z = model.encode(x, y)
        x_hat = model.decode(Z, y)

    temp_preds = y.detach().cpu().numpy()
    x_hat_np = x_hat.detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    preds = [[], []]

    for i,_ in enumerate(temp_preds[0]):
        if(x_hat_np[i] > 0.5):
            preds[0].append(temp_preds[0][i])
            preds[1].append(temp_preds[1][i])    

    return x, np.asarray(preds), y

##################################################################################

def edge_index_to_edge_list(edge_index):
    preds_edges = []
    for i,_ in enumerate(edge_index[0]):
        temp = [edge_index[0][i], edge_index[1][i]]
        preds_edges.append(temp)
    preds_edges = np.asarray(preds_edges)
    return preds_edges

##################################################################################

def edges_to_faces(edges):
    face_list = set()
    edge_list = set()
    connections = defaultdict(set)
    for a, b in edges:
        connections[a] |= {b}
        connections[b] |= {a}
        common = connections[a] & connections[b]
        for x in common:
            forward_order = {(x, a),(a, b),(b, x)}
            reverse_order = {(a, x),(x, b),(b, a)}
            if not forward_order & edge_list:
                face_list.add((x, a, b))
                edge_list |= forward_order
            elif not reverse_order & edge_list:
                face_list.add((a, x, b))
                edge_list |= reverse_order
            else:
                face_list.add((b, x, a))
                edge_list.update([(b, x),(x, a),(a, b)])
  
    return np.array(list(face_list))

##################################################################################

def create_triangle_mesh(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh

##################################################################################

if __name__ == '__main__':
    args = build_args()

    x, predicted_edges, y = autoencode_patient(args.id, args.organ) #worst: 3655108, best: 4876166
    edges = edge_index_to_edge_list(predicted_edges)
    faces = edges_to_faces(edges)
    output = create_triangle_mesh(x, faces)

    save_path = f'../outputs/gae_{args.id}_{args.organ}'
    o3d.io.write_triangle_mesh(save_path, output)