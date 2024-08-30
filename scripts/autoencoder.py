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
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE, LayerNorm, InnerProductDecoder
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
    
############################################################################################

#Taking arguments from user
#python autoencoder.py --use_input_encoder False --encoder_features 495 --activation ReLU --layer gat --dropout 0.001 -lr 
#0.000009 --weight_decay 0.006
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/2000/1000180/")
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use_input_encoder", type=ast.literal_eval, default=True)
    parser.add_argument("--in_features", type=int, default=3)
    parser.add_argument("--encoder_features", type=int, default=516)
    parser.add_argument("--hidden_channels", nargs='+', type = int, default=[1024, 1024, 512, 512, 256, 128, 64])
    parser.add_argument("--activation", type=str, default="LeakyReLU")
    parser.add_argument("--normalization", type=ast.literal_eval, default=True)
    parser.add_argument("--layer", type=str, default="gat")
    parser.add_argument("--dropout", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.00007)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    args = parser.parse_args()
    return args
    
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

#Train function
def train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    #cumm loss and div by len
    optimizer.step()
    return float(loss)
    
############################################################################################

#Validation function
def calculate_val_loss(model, x, val_pos_edge_index, val_neg_edge_index):
    model.eval()
    
    with torch.no_grad():
        z = model.encode(x, val_pos_edge_index)
    
    loss = model.recon_loss(z, val_pos_edge_index, val_neg_edge_index)
    #cumm loss and div by len
    
    return loss
    
############################################################################################

#Test function
def test(model, x, pos_edge_index, neg_edge_index):
    model.eval()
    
    with torch.no_grad():
        z = model.encode(x, pos_edge_index)

    auc, ap = model.test(z, pos_edge_index, neg_edge_index)

    loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
    #cumm loss and div by len

    return loss, auc, ap
    
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
    torch_geometric.seed_everything(42)

    registeration_path = args.path
    with open('../data/gae/liver/data', 'rb') as f:
        data = pkl.load(f)

    
    print(data)

    run = wandb.init(
        project="digital_twin_gae",
        entity="yussufwaly",
        notes="GAE",
        tags=[],
        config=args,
        )

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

    

    # model
    model = GAE(GNN(**model_params))

    print(model)

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    #WRITING
    # with open("../../../../../../vol/aimspace/users/wyo/outputs/x", "wb") as fp:
    #     pkl.dump(data.x, fp)
    # fp.close()

    x = data.x.to(device)
    all_edge_index = data.y.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    val_pos_edge_index = data.val_pos_edge_index.to(device)
    val_neg_edge_index = data.val_neg_edge_index.to(device)
    wandb.config.update( {'device': device }, allow_val_change=True)

    # inizialize the optimizer
    optimizer = build_optimizer(model, wandb.config.optimizer, wandb.config.lr, wandb.config.weight_decay)

    
    for epoch in range(1, wandb.config.epochs + 1):
        loss = train(model, optimizer, x, train_pos_edge_index)
        val_loss = calculate_val_loss(model, x, val_pos_edge_index, val_neg_edge_index)
        wandb.log({'train_loss': loss, 'validation_loss': val_loss, 'epoch': epoch})

        test_pos_edge_index = data.test_pos_edge_index.to(device)
        test_neg_edge_index = data.test_neg_edge_index.to(device)

        test_loss, auc, ap = test(model, x, test_pos_edge_index, test_neg_edge_index)
        wandb.log({'test_loss': test_loss, 'AUC': auc, 'AP': ap, 'epoch': epoch})
        
        wandb.watch(model)

    if(wandb.config.output):
        model.eval()
        with torch.no_grad():
            Z = model.encode(x, all_edge_index)
            x_hat = model.decode(Z, all_edge_index)

        temp_preds = all_edge_index.detach().cpu().numpy()
        x_hat_np = x_hat.detach().cpu().numpy()
        preds = [[], []]

        for i,_ in enumerate(temp_preds[0]):
            if(x_hat_np[i] > 0.5):
                preds[0].append(temp_preds[0][i])
                preds[1].append(temp_preds[1][i])

        preds_tensor = torch.from_numpy(np.asarray(preds))

        preds_adj = SparseTensor(
            row=preds_tensor[0],
            col=preds_tensor[1],
            sparse_sizes=(x.shape[0], x.shape[0])
        ).to_dense()

        all_edge_index_adj = SparseTensor(
            row=all_edge_index[0],
            col=all_edge_index[1],
            sparse_sizes=(x.shape[0], x.shape[0])
        ).to_dense()

        all_edge_index_adj = all_edge_index_adj.detach().cpu().numpy()
        preds_adj = preds_adj.detach().cpu().numpy()
        x_file = x.detach().cpu().numpy()
        z_file = Z.detach().cpu().numpy()

        #WRITING
        with open('../data/gae/liver/x', "wb") as fp:
            pkl.dump(x_file, fp)
        fp.close()
        #WRITING
        with open('../data/gae/liver/z', "wb") as fp:
            pkl.dump(z_file, fp)
        fp.close()
        #WRITING
        with open('../data/gae/liver/x_hat', "wb") as fp:
            pkl.dump(x_hat_np, fp)
        fp.close()
        #WRITING
        with open('../data/gae/liver/preds_O', "wb") as fp:
            pkl.dump(preds, fp)
        fp.close()
        #WRITING
        with open('../data/gae/liver/preds', "wb") as fp:
            pkl.dump(preds_adj, fp)
        fp.close()
        #WRITING
        with open('../data/gae/liver/labels', "wb") as fp:
            pkl.dump(all_edge_index_adj, fp)
        fp.close()

        print(np.all(preds == data.y.numpy()))

    if(wandb.config.save):
        torch.save(model, f'../models/{wandb.config.organ}_gae_{wandb.config.layer}.pt')