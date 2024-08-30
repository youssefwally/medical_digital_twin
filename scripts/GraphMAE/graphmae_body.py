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
    parser.add_argument("--path", type=str, default="../../../../../../vol/space/projects/ukbb/projects/silhouette/gender_registered_1k/")
    parser.add_argument("--output", type=ast.literal_eval, default=False)
    parser.add_argument("--save", type=ast.literal_eval, default=False)
    
    
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_hidden", type=int, default=256)
    parser.add_argument("--residual", type=ast.literal_eval, default=True)
    parser.add_argument("--in_drop", type=float, default=0.0001)
    parser.add_argument("--attn_drop", type=float, default=0.04)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.00004)
    parser.add_argument("--negative_slope", type=float, default=0.6)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--mask_rate", type=float, default=0.05)
    parser.add_argument("--drop_edge_rate", type=float, default=0.01)
    parser.add_argument("--replace_rate", type=float, default=0.47)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="mse") 
    parser.add_argument("--alpha_l", type=int, default=4) #pow coefficient for sce loss
    parser.add_argument("--optimizer", type=str, default="radam")

    parser.add_argument("--max_epoch_f", type=int, default=20)
    parser.add_argument("--lr_f", type=float, default=0.00001)
    parser.add_argument("--weight_decay_f", type=float, default=0.000001)
    parser.add_argument("--linear_prob", type=ast.literal_eval, default=True)
    parser.add_argument("--concat_hidden", type=ast.literal_eval, default=True)

    parser.add_argument("--num_features", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=3)

    
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
    open3d_geometry.compute_vertex_normals()

    # Extract points, normals and adjacency information
    adjacency_matrix = triangle_mesh_to_adjacency_matrix(open3d_geometry)
    # Create a DGL graph from the adjacency matrix
    dgl_graph = dgl.from_scipy(adjacency_matrix)

    # Add node features (e.g., point coordinates) to the DGL graph
    points_np = np.array(open3d_geometry.vertices)
    normals_np = np.array(open3d_geometry.vertex_normals)
    # features = np.concatenate((points_np, normals_np), axis=1)
    features = points_np
    
    dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)

    return dgl_graph

#############################################################################################################

def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
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
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        # if epoch == (max_epoch//2):
        #     # print(model)
        #     val_acc, test_acc, best_val_test_acc = evaluate(model, (eval_train_loader, val_loader, test_loader), num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
        #     wandb.log({'val_mse': val_acc,'test_mse': test_acc, 'epoch': epoch})
        model.eval()
        val_loss_list = []
        for subgraph in val_loader:
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])
            val_loss_list.append(loss.item())
            val_loss = np.mean(val_loss_list)

        model.eval()
        test_loss_list = []
        for subgraph in test_loader:
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])
            test_loss_list.append(loss.item())
            test_loss = np.mean(test_loss_list)

        wandb.log({'train_loss': train_loss,'val_loss': val_loss,'test_loss': test_loss, 'epoch': epoch})
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | test_loss: {test_loss:.4f}")

        wandb.watch(model)
    return model

#############################################################################################################

def evaluate(model, loaders, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        if len(loaders[0]) > 1:
            x_all = {"train": [], "val": [], "test": []}
            y_all = {"train": [], "val": [], "test": []}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        # print(f'latent space: {x}')
                        # print(f'latent space shape: {x.shape}')
                        x_all[key].append(x)
                        y_all[key].append(subgraph.ndata["feat"])  
            in_dim = x_all["train"][0].shape[1]
            encoder = LogisticRegression(in_dim, num_classes)
            num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
            if not mute:
                print(f"num parameters for finetuning: {sum(num_finetune_params)}")
                # torch.save(x.cpu(), "feat.pt")
            
            encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
            val_acc, final_acc, estp_acc = mutli_graph_linear_evaluation(encoder, x_all, y_all, optimizer_f, max_epoch_f, device, mute)
            return val_acc, final_acc, estp_acc
        else:
            x_all = {"train": None, "val": None, "test": None}
            y_all = {"train": None, "val": None, "test": None}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        mask = subgraph.ndata[f"{key}_mask"]
                        x_all[key] = x[mask]
                        y_all[key] = subgraph.ndata["label"][mask]  
            in_dim = x_all["train"].shape[1]
            
            encoder = LogisticRegression(in_dim, num_classes)
            encoder = encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

            x = torch.cat(list(x_all.values()))
            y = torch.cat(list(y_all.values()))
            num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
            num_nodes = num_train + num_val + num_test
            train_mask = torch.arange(num_train, device=device)
            val_mask = torch.arange(num_train, num_train + num_val, device=device)
            test_mask = torch.arange(num_train + num_val, num_nodes, device=device)
            
            val_acc, final_acc, estp_acc = linear_probing_for_inductive_node_classiifcation(encoder, x, y, (train_mask, val_mask, test_mask), optimizer_f, max_epoch_f, device, mute)
            return val_acc, final_acc, estp_acc
    else:
        raise NotImplementedError
    
#############################################################################################################

def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            # val_out = np.where(val_out >= 0.0, 1.0, 0.0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)# 
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            # test_out = np.where(test_out >= 0.0, 1.0, 0.0)

            # val_acc = f1_score(val_label, val_out, average="micro")
            # test_acc = f1_score(test_label, test_out, average="micro")

            # mse = mean_absolute_error(val_label, val_out)
            val_acc = mean_squared_error(val_label, val_out)
            test_acc = mean_squared_error(test_label, test_out)
            # r2 = r2_score(y_true, y_pred)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    if mute:
        print(f"# IGNORE: --- Best val mse: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-Test-msa: {best_val_test_acc:.4f},  Final-Test-mse: {test_acc:.4f}--- ")
    else:
        print(f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- ")

    return val_acc, test_acc, best_val_test_acc

############################################################################################

def get_data(path):
    train_ids_path = "../data/body_split_train.txt"
    val_ids_path = "../data/body_split_val.txt"
    test_ids_path = "../data/body_split_test.txt"
    data_errors = []

    train_dirs = np.loadtxt(train_ids_path, delimiter=",", dtype=str)
    val_dirs = np.loadtxt(val_ids_path, delimiter=",", dtype=str)
    test_dirs = np.loadtxt(test_ids_path, delimiter=",", dtype=str)

    train_graphs = []
    val_graphs = []
    test_graphs = []

    for dir in train_dirs:
        try:
            mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}')
            dgl_graph = open3d_to_dgl_graph(mesh)
            dgl_graph = dgl_graph.remove_self_loop()
            dgl_graph = dgl_graph.add_self_loop()
            train_graphs.append(dgl_graph)
        except:
            data_errors.append(dir)

    for dir in val_dirs:
        try:
            mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}')
            dgl_graph = open3d_to_dgl_graph(mesh)
            dgl_graph = dgl_graph.remove_self_loop()
            dgl_graph = dgl_graph.add_self_loop()
            val_graphs.append(dgl_graph)
        except:
            data_errors.append(dir)

    for dir in test_dirs:
        try:
            mesh = o3d.io.read_triangle_mesh(f'{path}{str(dir)}')
            dgl_graph = open3d_to_dgl_graph(mesh)
            dgl_graph = dgl_graph.remove_self_loop()
            dgl_graph = dgl_graph.add_self_loop()
            test_graphs.append(dgl_graph)
        except:
            data_errors.append(dir)

    print(f'Dirs with Data Error: {data_errors}')

    return train_graphs, val_graphs, test_graphs

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
    
    train_graphs, val_graphs, test_graphs = get_data(wandb.config.path)
 
    scheduler = None
    logger = None
    wandb.config.update( {'device': device }, allow_val_change=True)

    #Model
    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(wandb.config.optimizer, model, wandb.config.lr, wandb.config.weight_decay)

    model = pretrain(model, (train_graphs, val_graphs, test_graphs, train_graphs), optimizer, wandb.config.epochs, device, scheduler, wandb.config.num_classes, wandb.config.lr_f, wandb.config.weight_decay_f, wandb.config.max_epoch_f, wandb.config.linear_prob, logger)
    print(model)

    if(wandb.config.save):
        torch.save(model, f'../models/body_graphmae_vertices.pt')

    print("---------------------------------------------", flush=True)
