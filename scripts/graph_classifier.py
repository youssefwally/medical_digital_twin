#imports
import os
import argparse
import ast
import pickle as pkl
from itertools import tee
import random
import wandb
import numpy as np
import pandas as pd
import open3d as o3d
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from torchmetrics import R2Score
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch_sparse import SparseTensor

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
    
############################################################################################
#Taking arguments from user
#python autoencoder.py --use_input_encoder False --encoder_features 495 --activation ReLU --layer gat --dropout 0.001 -lr 
#0.000009 --weight_decay 0.006
def build_args():
    parser = argparse.ArgumentParser(description='GAE')
    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    # parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/original_meshes/")
    # parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/vertex_clustering/")
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    parser.add_argument("--label", type=str, default="VAT")
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument('--batchs', dest='batchs', type=int, help='Batches.', default=128)
    parser.add_argument('--alpha', dest='alpha', type=float, help='alpha.', default=0.25)
    parser.add_argument('--gamma', dest='gamma', type=float, help='gamma.', default=2)
    parser.add_argument('--threshold', dest='threshold', type=float, help='threshold.', default=0.5)
    parser.add_argument("--use_input_encoder", type=ast.literal_eval, default=True)
    parser.add_argument("--in_features", type=int, default=3)
    parser.add_argument("--encoder_features", type=int, default=516)
    parser.add_argument("--hidden_channels", nargs='+', type = int, default=[1024, 1024, 512, 512, 256, 256, 256])
    parser.add_argument("--num_conv_layers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--activation", type=str, default="LeakyReLU")
    parser.add_argument("--normalization", type=ast.literal_eval, default=True)
    parser.add_argument("--layer", type=str, default="gat")
    parser.add_argument("--dropout", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.00007)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="StepLR")

    parser.add_argument('--step_size', dest='step_size', type=float,
            help='step_size.', default=1.00)
    parser.add_argument('--scheduler_gamma', dest='scheduler_gamma', type=float,
            help='scheduler_gamma.', default=0.7)
    
    args = parser.parse_args()
    return args
############################################################################################
#Data

class MeshDataset(Dataset):
    def __init__(self, dirs_path, root, organ, label, features, transform=None, pre_transform=None):
        self.dirs_path = dirs_path
        self.organ = organ
        self.label = label
        self.features = features
        with open(self.dirs_path, 'r') as f:
            self.dirs = [line.strip() for line in f]
        labels_path = "../data/liver_diseases.csv"
        self.labels = pd.read_csv(labels_path, delimiter=",", dtype=str, index_col=0)
        super(MeshDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self.dirs
    
    def process(self):
        done = True
        done_data = next(os.walk('../../../../../../vol/aimspace/users/wyo/proccessed_data'))[2]
        for dir in self.dirs:
            if(not str(dir + ".pt") in done_data):
                done = False
        if(done):
            print("Data Already Done!", flush=True)
            pass        
        else:
            for dir in self.dirs:
                print(f"Currently doing subject {dir}", flush=True)
                registered_mesh = []
                labels = self.labels

                mesh = o3d.io.read_triangle_mesh(f'{self.root}/{str(dir)}/{self.organ}')

                vertices_data = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                vertices = torch.from_numpy(vertices_data).double()
                edges = []
                for triangle in triangles:
                    edges.append([triangle[0], triangle[1]])
                    edges.append([triangle[0], triangle[2]])
                    edges.append([triangle[1], triangle[2]])
                edges_torch = [[],[]]
                edges = np.unique(np.array(edges), axis=0)
                for edge in edges:
                    edges_torch[0].append(edge[0])
                    edges_torch[1].append(edge[1])
                edges_torch = torch.from_numpy(np.asarray(edges_torch)).long()
                registered_mesh.append((vertices.type(torch.float32), edges_torch))
                data = None

                if(self.label == 'disease'):
                    label = 1 if int(dir) in labels.index else 0
                    label_tensor = label
                    data = Data(x=registered_mesh[0][0], y=label_tensor, edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
                else:
                    cur_patient_feature = self.features[self.features['eid'] == int(dir)]
                    if(len(cur_patient_feature[self.label]) == 1):
                        if(not pd.isnull(cur_patient_feature[self.label].item())):
                            cur_patient_feature_tensor = torch.tensor(cur_patient_feature[self.label].item())
                            data = Data(x=registered_mesh[0][0], y=cur_patient_feature_tensor, edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))

                if data is None:
                    raise ValueError(f"No data found at index {dir}")

                torch.save(data, f'../../../../../../vol/aimspace/users/wyo/proccessed_data/{dir}.pt')

    
    def len(self):
        return len(self.dirs)
    
    def get(self, idx): 
        dir = self.processed_file_names[idx]
        
        data = torch.load(f'../../../../../../vol/aimspace/users/wyo/proccessed_data/{dir}.pt')

        if(self.label == 'disease'):
                label = 1 if int(dir) in labels.index else 0
                label_tensor = label
        else:
            cur_patient_feature = self.features[self.features['eid'] == int(dir)]
            if(len(cur_patient_feature[self.label]) == 1):
                if(not pd.isnull(cur_patient_feature[self.label].item())):
                    cur_patient_feature_tensor = torch.tensor(cur_patient_feature[self.label].item())
                    label_tensor = cur_patient_feature_tensor

        data.y = label_tensor

        return data

############################################################################################

class MeshInMemoryDataset(InMemoryDataset):
    def __init__(self, dirs_path, root, organ, label, features, transform=None, pre_transform=None):
        self.dirs_path = dirs_path
        self.organ = organ
        self.label = label
        self.features = features
        with open(self.dirs_path, 'r') as f:
            self.dirs = [line.strip() for line in f]
        labels_path = "../data/liver_diseases.csv"
        self.labels = pd.read_csv(labels_path, delimiter=",", dtype=str, index_col=0)
        super(InMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return self.dirs
    
    @property
    def processed_file_names(self):
        return self.dirs

    def __len__(self):
        return len(self.dirs)
    
    def process(self): 
        data_list = []
        labels = self.labels

        for idx in range(len(self.raw_file_names)):
            print(f"Currently doing subject {self.raw_file_names[idx]}", flush=True)
            registered_mesh = []
            mesh = o3d.io.read_triangle_mesh(f'{self.root}/{str(self.raw_file_names[idx])}/{self.organ}')

            vertices_data = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            vertices = torch.from_numpy(vertices_data).double()
            edges = []
            for triangle in triangles:
                edges.append([triangle[0], triangle[1]])
                edges.append([triangle[0], triangle[2]])
                edges.append([triangle[1], triangle[2]])
            edges_torch = [[],[]]
            edges = np.unique(np.array(edges), axis=0)
            for edge in edges:
                edges_torch[0].append(edge[0])
                edges_torch[1].append(edge[1])
            edges_torch = torch.from_numpy(np.asarray(edges_torch)).long()
            registered_mesh.append((vertices.type(torch.float32), edges_torch))
            data = None

            if(self.label == 'disease'):
                label = 1 if int(dir) in labels.index else 0
                label_tensor = label
                data = Data(x=registered_mesh[0][0], y=label_tensor, edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
            else:
                cur_patient_feature = self.features[self.features['eid'] == int(self.raw_file_names[idx])]
                if(len(cur_patient_feature[self.label]) == 1):
                    if(not pd.isnull(cur_patient_feature[self.label].item())):
                        cur_patient_feature_tensor = torch.tensor(cur_patient_feature[self.label].item())
                        data = Data(x=registered_mesh[0][0], y=cur_patient_feature_tensor, edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
            if data is None:
                raise ValueError(f"No data found at index {self.raw_file_names[idx]}")

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


############################################################################################
# @DeprecationWarning
def get_data(path, organ, wanted_label='disease'):
    registered_mesh = []
    labels_path = "../data/liver_diseases.csv"
    train_ids_path = "../data/NonNa_organs_split_train.txt"
    val_ids_path = "../data/NonNa_organs_split_val.txt"
    test_ids_path = "../data/NonNa_organs_split_test.txt"
    labels = pd.read_csv(labels_path, delimiter=",", dtype=str, index_col=0)    
    train_dirs = np.loadtxt(train_ids_path, delimiter=",", dtype=str)
    val_dirs = np.loadtxt(val_ids_path, delimiter=",", dtype=str)
    test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)
    dirs = next(os.walk(path))[1]
    train_dataset = []
    val_dataset = []
    test_dataset = []
    errors = []
    #In Test
    dirs = dirs[:5000]
    print(f'Number of samples used: {len(dirs)}', flush=True)

    body_fields = ["eid", "22407-2.0", "22408-2.0", "31-0.0"]
    full_ukbb_data = pd.read_csv("../../../../../../vol/aimspace/projects/ukbb/data/tabular/ukb668815_imaging.csv", usecols=body_fields)
    full_ukbb_data_new_names = {'22407-2.0':'VAT', '22408-2.0':'ASAT', '31-0.0':'sex'}
    full_ukbb_data = full_ukbb_data.rename(index=str, columns=full_ukbb_data_new_names)
    
    basic_features = pd.read_csv("../data/basic_features.csv")
    basic_features_new_names = {'21003-2.0':'age', '31-0.0':'sex', '21001-2.0':'bmi', '21002-2.0':'weight','50-2.0':'height'}
    basic_features = basic_features.rename(index=str, columns=basic_features_new_names)
    print(f'Number of samples used: {len(dirs)}, with label: {wanted_label}', flush=True)

    if(wanted_label == 'sex' or wanted_label == 'VAT' or wanted_label == 'ASAT'):
        features = full_ukbb_data
    else:
        features = basic_features
    
    for dir in dirs:
        registered_mesh = []
        try: 
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
            edges = np.unique(np.array(edges), axis=0)
            for edge in edges:
                edges_torch[0].append(edge[0])
                edges_torch[1].append(edge[1])
            edges_torch = torch.from_numpy(np.asarray(edges_torch)).long()
            registered_mesh.append((vertices.type(torch.float32), edges_torch))

            # label = labels.loc[int(dir)].to_list()
            # label = [int(id) for id in label]
            # label_np = np.asarray(label)
            # label_tensor = torch.from_numpy(label_np)

            if(wanted_label == 'disease'):
                label = 1 if int(dir) in labels.index else 0
                label_tensor = label
                data = Data(x=registered_mesh[0][0], y=label_tensor, edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
                if(dir in train_dirs):
                    train_dataset.append(data)
                if(dir in val_dirs):
                    val_dataset.append(data)
                elif(dir in test_dirs):
                    test_dataset.append(data)
            else:
                cur_patient_feature = features[features['eid'] == int(dir)]
                if(len(cur_patient_feature[wanted_label]) == 1):
                    if(not pd.isnull(cur_patient_feature[wanted_label].item())):
                        cur_patient_feature_tensor = torch.tensor(cur_patient_feature[wanted_label].item())
                        data = Data(x=registered_mesh[0][0], y=cur_patient_feature_tensor, edge_index=registered_mesh[0][1], num_nodes= len(registered_mesh[0][0]))
                        if(dir in train_dirs):
                            train_dataset.append(data)
                        if(dir in val_dirs):
                            val_dataset.append(data)
                        elif(dir in test_dirs):
                            test_dataset.append(data)

        except:
            errors.append(dir)
    print(f'#train subjects: {len(train_dataset)}, #val subjects: {len(val_dataset)}, #test subjects: {len(test_dataset)}')    
    return train_dataset, val_dataset, test_dataset
    
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

class GNN(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, activation, normalization, num_classes, num_conv_layers=4, layer='gcn',
                 use_input_encoder=True, encoder_features=128, apply_batch_norm=True,
                 apply_dropout_every=True, dropout = 0):
        super(GNN, self).__init__()
        torch.manual_seed(42)
        
        self.fc = torch.nn.ModuleList()
        self.layer_type = layer
        self.num_classes = num_classes
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

        for i in range((len(hidden_channels)-num_conv_layers)):
            self.fc.append(Linear(hidden_channels[i+num_conv_layers-1], hidden_channels[i+num_conv_layers]))
        
        self.pred_layer = Linear(hidden_channels[len(hidden_channels)-1], self.num_classes)

        if(self.num_classes == -1):
            self.pred_layer = Linear(hidden_channels[len(hidden_channels)-1], 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

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

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)

       
        for i in range(len(self.fc)):
           x = self.fc[i](x)
           x = torch.tanh(x)
           x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pred_layer(x) 
        # x = torch.nn.Sigmoid()(x) 

        return x
    
############################################################################################

def reduce_loss(loss, reduction):
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()

############################################################################################

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

############################################################################################

def focal_loss(pred, target, alpha = 1, gamma = 2, weight=None, reduction='mean', avg_factor=None):
    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

#Train function
def train(model, optimizer, dataloader, alpha, gamma, threshold = 0.5, loss_fn = nn.BCEWithLogitsLoss()):
    """Train network on training dataset."""
    model.train()
    cumulative_loss = 0.0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out.squeeze(1), data.y.float())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(dataloader)
    
############################################################################################

#Validation function
def calculate_val_loss(model, dataloader, alpha, gamma, threshold = 0.5, loss_fn = nn.BCEWithLogitsLoss()):
    model.eval()
    cumulative_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out.squeeze(1), data.y.float())
            cumulative_loss += loss.item()
    return cumulative_loss / len(dataloader)
    
############################################################################################

#Test function
def test(model, dataloader, alpha, gamma, threshold = 0.5, loss_fn = nn.BCEWithLogitsLoss(), label="disease"):    
    model.eval()
    prediction_accuracies = []
    prediction_f1 = []
    prediction_accuracies = []
    measure_score = R2Score().to(device)

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            predictions = model(data)

            if(label == 'disease' or label == 'sex'):
                # predicted_class_labels = torch.nn.Sigmoid()(predictions)
                predicted_class_labels = predictions.squeeze(1)
                predicted_class_labels = torch.nn.Sigmoid()(predicted_class_labels) 
                predicted_class_labels = torch.round(predicted_class_labels)

                correct_assignments = (predicted_class_labels == data.y.float()).sum()
                num_assignemnts = predicted_class_labels.shape[0]
                prediction_accuracies.append(float(correct_assignments / num_assignemnts))
                f1_score = sklearn.metrics.f1_score(predicted_class_labels.int().detach().cpu(), data.y.int().detach().cpu(), average='weighted')
                prediction_f1.append((f1_score))
            else:
                predicted_label = predictions.squeeze(1)
                prediction_accuracies.append(measure_score(predicted_label, data.y))
                prediction_f1.append(0)

            if(label == 'disease'):
                # print(f'gt: {data.y.float().detach().cpu()}, len gt: {len(np.asarray(data.y.float().detach().cpu()))}, pred: {predicted_class_labels.detach().cpu()}, len pred: {len(np.asarray(predicted_class_labels.detach().cpu()))}')
                cm = confusion_matrix(data.y.float().detach().cpu(), predicted_class_labels.detach().cpu())
                wandb.log({"conf_mat" : wandb.plot.confusion_matrix( 
                    preds=np.asarray(predicted_class_labels.detach().cpu()), y_true=np.asarray(data.y.float().detach().cpu()),
                    class_names=['healthy', 'unhealthy'])})
            elif(label == 'sex'):
                # print(f'gt: {data.y.float().detach().cpu()}, len gt: {len(np.asarray(data.y.float().detach().cpu()))}, pred: {predicted_class_labels.detach().cpu()}, len pred: {len(np.asarray(predicted_class_labels.detach().cpu()))}')
                cm = confusion_matrix(data.y.float().detach().cpu(), predicted_class_labels.detach().cpu())
                wandb.log({"conf_mat" : wandb.plot.confusion_matrix( 
                    preds=np.asarray(predicted_class_labels.detach().cpu()), y_true=np.asarray(data.y.float().detach().cpu()),
                    class_names=['female', 'male'])})

    return sum(prediction_accuracies) / len(dataloader), sum(prediction_f1) / len(dataloader)
    
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

    run = wandb.init(
        project="digital_twin_graph_classifier",
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
            num_classes= wandb.config.num_classes,
            activation=activation,
            normalization = wandb.config.normalization,
            layer = wandb.config.layer,
            num_conv_layers = wandb.config.num_conv_layers,
            dropout = wandb.config.dropout)

    

    # model
    model = GNN(**model_params)

    print(model)

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model= nn.DataParallel(model)
    model = model.to(device)
    
    #WRITING
    # with open("../../../../../../vol/aimspace/users/wyo/outputs/x", "wb") as fp:
    #     pkl.dump(data.x, fp)
    # fp.close()
    print(wandb.config.path, flush=True)

    print("loading IDs", flush=True)
    train_ids_path = "../data/NonNa_organs_split_train.txt"
    val_ids_path = "../data/NonNa_organs_split_val.txt"
    test_ids_path = "../data/NonNa_organs_split_test.txt"

    train_dirs = np.loadtxt(train_ids_path, delimiter=",", dtype=str)
    val_dirs = np.loadtxt(val_ids_path, delimiter=",", dtype=str)
    test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)
    
    label_based_train_ids_path = "../data/label_based_organs_split_train.txt"
    label_based_val_ids_path = "../data/label_based_organs_split_val.txt"
    label_based_test_ids_path = "../data/label_based_organs_split_test.txt"

    # temp_train_dirs = np.loadtxt(train_ids_path, delimiter=",", dtype=str)
    # temp_val_dirs = np.loadtxt(val_ids_path, delimiter=",", dtype=str)
    # temp_test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)
    # temp_dirs = next(os.walk(wandb.config.path))[1]
    
    print("loading Features", flush=True)
    if(wandb.config.label == 'sex' or wandb.config.label == 'VAT' or wandb.config.label == 'ASAT'):
        body_fields = ["eid", "22407-2.0", "22408-2.0", "31-0.0"]
        full_ukbb_data = pd.read_csv("../../../../../../vol/aimspace/projects/ukbb/data/tabular/ukb668815_imaging.csv", usecols=body_fields)
        full_ukbb_data_new_names = {'22407-2.0':'VAT', '22408-2.0':'ASAT', '31-0.0':'sex'}
        full_ukbb_data = full_ukbb_data.rename(index=str, columns=full_ukbb_data_new_names)
        features = full_ukbb_data
    else:
        basic_features = pd.read_csv("../data/basic_features.csv")
        basic_features_new_names = {'21003-2.0':'age', '31-0.0':'sex', '21001-2.0':'bmi', '21002-2.0':'weight','50-2.0':'height'}
        basic_features = basic_features.rename(index=str, columns=basic_features_new_names)
        features = basic_features

    new_train_dirs = []
    new_val_dirs = []
    new_test_dirs = []

    print("Filtering IDs", flush=True)
    for dir in train_dirs:
        cur_patient_feature = features[features['eid'] == int(dir)]
        if(len(cur_patient_feature[wandb.config.label]) == 1):
                if(not pd.isnull(cur_patient_feature[wandb.config.label].item())):
                    new_train_dirs.append(dir)
    for dir in val_dirs:
        cur_patient_feature = features[features['eid'] == int(dir)]
        if(len(cur_patient_feature[wandb.config.label]) == 1):
                if(not pd.isnull(cur_patient_feature[wandb.config.label].item())):
                    new_val_dirs.append(dir)
    for dir in test_dirs:
        cur_patient_feature = features[features['eid'] == int(dir)]
        if(len(cur_patient_feature[wandb.config.label]) == 1):
                if(not pd.isnull(cur_patient_feature[wandb.config.label].item())):
                    new_test_dirs.append(dir)

    np.savetxt(label_based_train_ids_path, np.asarray(new_train_dirs), fmt='%s')
    np.savetxt(label_based_val_ids_path, np.asarray(new_val_dirs), fmt='%s')
    np.savetxt(label_based_test_ids_path, np.asarray(new_test_dirs), fmt='%s')

    print("Loading Data", flush=True)
    train_dataset = MeshDataset(dirs_path=label_based_train_ids_path, root=wandb.config.path, organ=wandb.config.organ, label=wandb.config.label, features=features)
    val_dataset = MeshDataset(dirs_path=label_based_val_ids_path, root=wandb.config.path, organ=wandb.config.organ, label=wandb.config.label, features=features)
    test_dataset = MeshDataset(dirs_path=label_based_test_ids_path, root=wandb.config.path, organ=wandb.config.organ, label=wandb.config.label, features=features)
    train_loader = DataLoader(dataset = train_dataset, batch_size=wandb.config.batchs, shuffle=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size=wandb.config.batchs, shuffle=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size=wandb.config.batchs, shuffle=True)
   
    # train_dataset = MeshInMemoryDataset(dirs_path=label_based_train_ids_path, root=wandb.config.path, organ=wandb.config.organ, label=wandb.config.label)
    # val_dataset = MeshInMemoryDataset(dirs_path=label_based_val_ids_path, root=wandb.config.path, organ=wandb.config.organ, label=wandb.config.label)
    # test_dataset = MeshInMemoryDataset(dirs_path=label_based_test_ids_path, root=wandb.config.path, organ=wandb.config.organ, label=wandb.config.label)
    # train_loader = DataLoader(dataset = train_dataset, batch_size=wandb.config.batchs, shuffle=True)
    # val_loader = DataLoader(dataset = val_dataset, batch_size=wandb.config.batchs, shuffle=True)
    # test_loader = DataLoader(dataset = test_dataset, batch_size=wandb.config.batchs, shuffle=True)

    # train_dataset, val_dataset, test_dataset = get_data(wandb.config.path, wandb.config.organ, wandb.config.label)
    # healthy_train_dataset = []
    # healthy_val_dataset = []
    # healthy_test_dataset = []
    # unhealthy_train_dataset = []
    # unhealthy_val_dataset = []
    # unhealthy_test_dataset = []

    # if(wandb.config.label == 'disease'):
    #     count = [0,0,0,0,0,0]
    #     for i in train_dataset:
    #         count[i.y] = count[i.y] + 1
    #         if(i.y==0):
    #             healthy_train_dataset.append(i)
    #         else:
    #             unhealthy_train_dataset.append(i)
    #     for i in val_dataset:
    #         count[i.y+2] = count[i.y+2] + 1
    #         if(i.y==0):
    #             healthy_val_dataset.append(i)
    #         else:
    #             unhealthy_val_dataset.append(i)
    #     for i in test_dataset:
    #         count[i.y+4] = count[i.y+4] + 1
    #         if(i.y==0):
    #             healthy_test_dataset.append(i)
    #         else:
    #             unhealthy_test_dataset.append(i)

    #     print(f"Original Distrubution: {count}", flush=True)

    # if(wandb.config.label == 'disease'):
    #     healthy_train_dataset = random.sample(healthy_train_dataset, (count[1]))
    #     healthy_val_dataset = random.sample(healthy_val_dataset, (count[3]))
    #     healthy_test_dataset = random.sample(healthy_test_dataset, (count[5]))
    
    #     train_dataset = healthy_train_dataset + unhealthy_train_dataset
    #     val_dataset = healthy_val_dataset + unhealthy_val_dataset
    #     test_dataset = healthy_test_dataset + unhealthy_test_dataset
    
    #     count = [0,0,0,0,0,0]
    #     for i in train_dataset:
    #         count[i.y] = count[i.y] + 1
    #     for i in val_dataset:
    #         count[i.y+2] = count[i.y+2] + 1
    #     for i in test_dataset:
    #         count[i.y+4] = count[i.y+4] + 1
    
    #     print(f"Used Distrubution: {count}", flush=True)


    # train_loader = DataLoader(dataset = train_dataset, batch_size=wandb.config.batchs, shuffle=True)
    # valid_loader = DataLoader(dataset = val_dataset, batch_size=wandb.config.batchs, shuffle=True)
    # test_loader = DataLoader(dataset = test_dataset, batch_size=wandb.config.batchs, shuffle=True, drop_last=True)

    wandb.config.update( {'device': device }, allow_val_change=True)

    print("Building Model", flush=True)
    # inizialize the optimizer
    optimizer = build_optimizer(model, wandb.config.optimizer, wandb.config.lr, wandb.config.weight_decay)

    if wandb.config.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.scheduler_gamma)
    elif wandb.config.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    elif wandb.config.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=wandb.config.lr) #Causes NAN and infinity losses sometimes!

    # loss_fn = nn.BCELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))
    if(wandb.config.label == 'disease' or wandb.config.label == 'sex'):
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    print("Starting Training", flush=True)
    for epoch in range(1, wandb.config.epochs + 1):
        loss = train(model, optimizer, train_loader, wandb.config.alpha, wandb.config.gamma, wandb.config.threshold, loss_fn)
        val_loss = calculate_val_loss(model, val_loader, wandb.config.alpha, wandb.config.gamma, wandb.config.threshold, loss_fn)
        wandb.log({'train_loss': loss, 'validation_loss': val_loss, 'epoch': epoch})

        # loss, accuracy, percision, recall, f1_score, ap, auc = test(model, test_loader, wandb.config.alpha, wandb.config.gamma, wandb.config.threshold, loss_fn)
        accuracy, f1_score = test(model, test_loader, wandb.config.alpha, wandb.config.gamma, wandb.config.threshold, loss_fn, wandb.config.label)
        # wandb.log({'test_loss': loss, 'accuracy': accuracy, 'percision': percision, 'recall': recall, 'f1_score': f1_score, 'average_percision': ap, 'auc': auc, 'epoch': epoch})
        wandb.log({'accuracy': accuracy, 'f1_score': f1_score, 'epoch': epoch})
        
        if wandb.config.scheduler == "StepLR":
            scheduler.step()
        elif wandb.config.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        elif wandb.config.scheduler == "CosineAnnealingLR":
            scheduler.step()

        wandb.watch(model)

    if(wandb.config.save):
        torch.save(model, f'../models/{wandb.config.organ}_graph_clasifier.pt')
