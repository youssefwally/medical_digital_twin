#imports
import os
import argparse
import ast
import pickle as pkl
from itertools import tee
import scipy.sparse as sp
import logging
import random
from tqdm import tqdm

import wandb
import numpy as np
import open3d as o3d

import dgl

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE, LayerNorm, Linear
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import gae_data
    
############################################################################################

#Taking arguments from user
#python my_verts_autoencoder.py --use_input_encoder False --encoder_features 495 --activation ReLU --layer gat --dropout 0.001 -lr 
#0.000009 --weight_decay 0.006
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use_input_encoder", type=ast.literal_eval, default=True)
    parser.add_argument("--in_features", type=int, default=3)
    parser.add_argument("--encoder_features", type=int, default=64)
    parser.add_argument("--hidden_channels", nargs='+', type = int, default=[1024, 512, 256, 128, 64])
    parser.add_argument("--activation", type=str, default="LeakyReLU")
    parser.add_argument("--normalization", type=ast.literal_eval, default=True)
    parser.add_argument("--layer", type=str, default="sageconv")
    parser.add_argument("--dropout", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.00007)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--optimizer", type=str, default="adam")
    
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
    vertices = torch.from_numpy(vertices).type(torch.float32)
    adjacency_matrix = adjacency_matrix.tocsr()
    adjacency_matrix = torch.tensor(adjacency_matrix.toarray())
    
    return vertices, adjacency_matrix

#############################################################################################################

def get_data(path, organ):
    train_ids_path = "../data/NonNa_organs_split_train.txt"
    val_ids_path = "../data/NonNa_organs_split_val.txt"
    test_ids_path = "../data/NonNa_organs_split_test.txt"

    train_dirs = np.loadtxt(train_ids_path, delimiter=",", dtype=str)
    val_dirs = np.loadtxt(val_ids_path, delimiter=",", dtype=str)
    test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)

    train_dirs = train_dirs[:5]
    val_dirs = val_dirs[:5]
    test_dirs =  test_dirs[:5]

    train_graphs = []
    val_graphs = []
    test_graphs = []

    for dir in train_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        vertices, adjacency_matrix = triangle_mesh_to_adjacency_matrix(mesh)
        graph = Data(x=vertices, y=adjacency_matrix)
        train_graphs.append(graph)

    for dir in val_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        vertices, adjacency_matrix = triangle_mesh_to_adjacency_matrix(mesh)
        graph = Data(x=vertices, y=adjacency_matrix)
        val_graphs.append(graph)

    for dir in test_dirs:
        mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}/{organ}')
        vertices, adjacency_matrix = triangle_mesh_to_adjacency_matrix(mesh)
        graph = Data(x=vertices, y=adjacency_matrix)
        test_graphs.append(graph)

    return train_graphs, val_graphs, test_graphs

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

def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]

############################################################################################

#Encoder
class GNN(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, activation, normalization, num_conv_layers=4, layer='sageconv',
                 use_input_encoder=True, encoder_features=256, apply_batch_norm=True,
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

    def forward(self, x, a):

        if self.use_input_encoder:
            x = self.input_encoder(x)

        if self.normalization is None:
            for i, layer in enumerate(self.layers):
                # Each GCN consists 2 modules GCN -> Activation 
                # GCN send edge index
                if i% 2 == 0:
                    x = layer(x, a)
                else:
                    x = layer(x)

                if self.apply_dropout_every:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, layer in enumerate(self.layers):
                # Each GCN consists 3 modules GCN -> Activation ->  Normalization 
                # GCN send edge index
                if i% 3 == 0:
                    x = layer(x, a)
                else:
                    x = layer(x)

                if self.apply_dropout_every:
                    x = F.dropout(x, p=self.dropout, training=self.training)        

        return x
    
############################################################################################

#Train function
def train(model, optimizer, dataloaders, max_epoch, device):
    logging.info("start training..")
    train_loader, val_loader, test_loader, eval_train_loader = dataloaders
    epoch_iter = tqdm(range(max_epoch))

    if isinstance(train_loader, list) and len(train_loader) ==1:
        train_loader = [train_loader[0].to(device)]
        eval_train_loader = train_loader
    if isinstance(val_loader, list) and len(val_loader) == 1:
        val_loader = [val_loader[0].to(device)]
        test_loader = val_loader
    
    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for subgraph in train_loader:
            subgraph = subgraph.to(device)

            optimizer.zero_grad()
            z = model.encode(subgraph.x, subgraph.y)
            loss = model.recon_loss(z, subgraph.y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
            
        model.eval()
        val_loss_list = []
        for subgraph in val_loader:
            subgraph = subgraph.to(device)
            with torch.no_grad():
                z = model.encode(subgraph.x, subgraph.y)

            loss = model.recon_loss(z, subgraph.y)

            val_loss_list.append(loss.item())
            val_loss = np.mean(val_loss_list)

        model.eval()
        test_ap_list = []
        test_auc_list = []
        test_loss_list = []
        for subgraph in test_loader:
            subgraph = subgraph.to(device)
            with torch.no_grad():
                z = model.encode(subgraph.x, subgraph.y)

            auc, ap = model.test(subgraph.x, subgraph.y)

            loss = model.recon_loss(subgraph.x, subgraph.y)

            test_ap_list.append(ap.item())
            test_auc_list.append(auc.item())
            test_loss_list.append(loss.item())
            test_ap = np.mean(test_ap_list)
            test_auc = np.mean(test_auc_list)
            test_loss = np.mean(test_loss_list)

        wandb.log({'train_loss': train_loss,'val_loss': val_loss,'test_loss': test_loss, 'epoch': epoch})
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | test_loss: {test_loss:.4f} | test_ap: {test_ap:.4f} | test_auc: {test_auc:.4f}")

        wandb.watch(model)
    return model
    
############################################################################################

#Optimizer
def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer
    
############################################################################################

if __name__ == '__main__':
    args = build_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch_geometric.seed_everything(42)
    torch.backends.cudnn.determinstic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = wandb.init(
        project="digital_twin_graphsage",
        entity="yussufwaly",
        notes="GAE",
        tags=[],
        config=args,
        )
    
    train_graphs, val_graphs, test_graphs = get_data(wandb.config.path, wandb.config.organ)

    #Model Parameters
    activation = getattr(nn, wandb.config.activation)
    model_params = dict(
            use_input_encoder = wandb.config.use_input_encoder,
            in_features= wandb.config.in_features, 
            encoder_features = wandb.config.encoder_features,
            hidden_channels= wandb.config.hidden_channels,
            activation=activation,
            normalization = wandb.config.normalization,
            layer = wandb.config.layer,
            num_conv_layers = len(wandb.config.hidden_channels),
            dropout = wandb.config.dropout)
    
    scheduler = None
    logger = None
    wandb.config.update( {'device': device }, allow_val_change=True)    

    # model
    model = GAE(GNN(**model_params))

    print(model)

    # move to GPU (if available)
    model = model.to(device)
    wandb.config.update( {'device': device }, allow_val_change=True)

    # inizialize the optimizer
    optimizer = build_optimizer(model, wandb.config.optimizer, wandb.config.lr, wandb.config.weight_decay)
    
    model = train(model, optimizer, (train_graphs, val_graphs, test_graphs, train_graphs), wandb.config.epochs, device)

    if(wandb.config.save):
        torch.save(model, f'../models/{wandb.config.organ}_my_verts_gae_{wandb.config.layer}.pt')

    print("done")