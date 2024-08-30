import os
import pickle as pkl
from itertools import tee

import numpy as np
import open3d as o3d

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, LayerNorm, Linear
from torch_geometric.nn.models.autoencoder import ARGVA
from torch_geometric.nn import global_mean_pool, global_max_pool

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

def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

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
        
        if layer == 'gcn':
            self.conv_mu = GCNConv( hidden_channels[len(hidden_channels)-1], int((hidden_channels[len(hidden_channels)-1])/2) , cached=True)
            self.conv_logstd = GCNConv(hidden_channels[len(hidden_channels)-1], int((hidden_channels[len(hidden_channels)-1])/2), cached=True)
        elif layer == 'sageconv':
            self.conv_mu = SAGEConv( hidden_channels[len(hidden_channels)-1], int((hidden_channels[len(hidden_channels)-1])/2) , cached=True)
            self.conv_logstd = SAGEConv(hidden_channels[len(hidden_channels)-1], int((hidden_channels[len(hidden_channels)-1])/2), cached=True)
        elif layer == 'gat':
            self.conv_mu = GATConv( hidden_channels[len(hidden_channels)-1], int((hidden_channels[len(hidden_channels)-1])/2) , cached=True)
            self.conv_logstd = GATConv(hidden_channels[len(hidden_channels)-1], int((hidden_channels[len(hidden_channels)-1])/2), cached=True)   

           

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
        
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

def train(model, discriminator, encoder_optimizer, discriminator_optimizer, num_nodes, x, train_pos_edge_index):
    model.train()
    encoder_optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)

    for i in range(50):
        idx = range(num_nodes)  
        discriminator.train()
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z[idx]) 
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

    loss = 0
    loss = loss + model.reg_loss(z)  
    
    loss = loss + model.recon_loss(z, train_pos_edge_index)
    loss = loss + (1 / num_nodes) * model.kl_loss()
    loss.backward()

    encoder_optimizer.step()

    return loss


@torch.no_grad()
def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    z = model.encode(x, train_pos_edge_index)

    input = z.cpu().numpy()

    auc, ap = model.test(z, pos_edge_index, neg_edge_index)

    return auc, ap


registeration_path = "../../../../../../vol/aimspace/users/wyo/organ_decimations_ply/2000/1000180/"

registered_mesh = []
organ = "liver_mesh.ply"
mesh = o3d.io.read_triangle_mesh(os.path.join(registeration_path, organ))

vertices_data = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
vertices = torch.from_numpy(vertices_data).double()
edges = []
for triangle in triangles:
    edges.append([triangle[0], triangle[1]])
    edges.append([triangle[0], triangle[2]])
    edges.append([triangle[1], triangle[2]])
edges_torch = torch.from_numpy(np.unique(np.array(edges), axis=0).reshape(2,-1)).long()

registered_mesh.append((vertices.type(torch.float32), edges_torch))
data = Data(x=registered_mesh[0][0], edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
data = train_test_split_edges(data)

activation = getattr(nn, "ReLU")
model_params = dict(
        use_input_encoder = True,
        in_features= 3, 
        encoder_features = 128,
        hidden_channels= [512, 256, 256, 128],
        activation=activation,
        normalization = True,
        layer = "sageconv",
        num_conv_layers = 4,
        dropout = 0.5)
epochs = 100

# model
encoder = GNN(**model_params)
discriminator = Discriminator(in_channels=64, hidden_channels=128, 
                              out_channels=1) 

model = ARGVA(encoder, discriminator)

print(model)

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
num_nodes = data.num_nodes

# inizialize the optimizer
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)

for epoch in range(1, epochs + 1):
    loss = train(model, discriminator, encoder_optimizer, discriminator_optimizer, num_nodes, x, train_pos_edge_index)

    test_pos_edge_index = data.test_pos_edge_index.to(device)
    test_neg_edge_index = data.test_neg_edge_index.to(device)

    auc, ap = test(model, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index)
    if(epoch % 10 == 0):
        print((f'Epoch: {epoch:03d}, Loss: {loss:.3f}, AUC: {auc:.3f}, '
           f'AP: {ap:.3f}'))

Z = model.encode(x, train_pos_edge_index)
print(f'latent space: {Z}')
