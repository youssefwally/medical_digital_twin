#imports
import os
import argparse
import ast
import pickle as pkl
from itertools import tee

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE, LayerNorm, Linear
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import gae_data
    
############################################################################################

#Taking arguments from user
#python autoencoder.py --use_input_encoder False --encoder_features 495 --activation ReLU --layer gat --dropout 0.001 -lr 
#0.000009 --weight_decay 0.006
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--model_path", type=str, default="../models/liver_mesh.ply_global_gae_gat.pt")
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

if __name__ == '__main__':
    args = build_args()
    torch_geometric.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = "../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/2000/"
    dirs = next(os.walk(path))[1]

    losses = []
    edges_count = []
    ordered_ids = []

    for dir in dirs:
        path_with_id = os.path.join(path, str(dir))
        data = gae_data.process_data(args.organ, path_with_id, False)

        # print(data)

        x = data.x.to(device)
        y = data.y.to(device)

        model = torch.load(args.model_path)
        model = model.to(device)

        model.eval()

        with torch.no_grad():
            Z = model.encode(x, y)

            # auc, ap = model.test(Z, y)
            loss = model.recon_loss(Z, y)

            x_hat = model.decode(Z, y)

        # print(f'Loss: {loss}')

        # print(f'Z: {Z}, Z size: {Z.shape}')
        losses.append(loss.detach().cpu())
        edges_count.append(data.y.shape[1])
        ordered_ids.append(dir)

        #WRITING
        with open(f'../../../../../../vol/aimspace/users/wyo/latent_spaces/edge_prediction/{args.organ[:-9]}/{dir}', "wb") as fp:
            pkl.dump(Z.detach().cpu().numpy(), fp)
        fp.close()

    losses = np.asarray(losses)
    avg_loss = np.average(losses)
    median_loss = np.median(losses)

    print(f'Average Loss: {avg_loss}, Median Loss: {median_loss}')

    #WRITING
    with open('../outputs/liver_gat_losses', "wb") as fp:
        pkl.dump(losses, fp)
    fp.close()
    #WRITING
    with open('../outputs/liver_gat_edges_count', "wb") as fp:
        pkl.dump(edges_count, fp)
    fp.close()
    #WRITING
    with open('../outputs/liver_gat_ids', "wb") as fp:
        pkl.dump(ordered_ids, fp)
    fp.close()