# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from arguments import arg_parse
import random
import wandb
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from datas import get_data
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import json
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

class MLP(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, classification=True, use_mlp=False,alpha=0.5, beta=1., gamma=.1):
    super(InfoGraph, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = wandb.config.prior
    self.classification = classification
    self.use_mlp = use_mlp
    

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    if(self.use_mlp):
        self.mlp_model = MLP(self.embedding_dim, 1)
        if self.classification:
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.mlp_model.parameters(), lr=0.001)
        else:
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.mlp_model.parameters(), lr=0.001)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, label, edge_index, batch, num_graphs):
    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        # term_b = torch.log(self.prior_d(y)).mean()
        term_b_proc = 1.0 - self.prior_d(y)
        term_b_proc_clipped = term_b_proc.clamp(min=1e-9)
        term_b = torch.log(term_b_proc_clipped).mean()
        PRIOR = - (term_a + term_b) * self.gamma
        # print(f'prior: {prior}, term a: {term_a}, y: {y}, min: {torch.min(self.prior_d(y))}, max: n/a, prior_d_norm: n/a, 1-prior_d_norm: n/a, log_prior: n/a, term b: {term_b}, gamma: {self.gamma}, PRIOR: {PRIOR}', flush=True)
    else:
        PRIOR = 0

    if(self.use_mlp):
        mlp_outputs = self.mlp_model(g_enc)
        mlp_loss = self.criterion(mlp_outputs, y)
    else:
        mlp_loss = 0
    
    return local_global_loss + PRIOR + mlp_loss

if __name__ == '__main__':
    args = arg_parse()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.determinstic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = wandb.init(
        project="digital_twin_infograph_w_discriminator",
        entity="yussufwaly",
        notes="GAE",
        tags=[],
        config=args,
        )
    
    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
    epochs = wandb.config.epochs
    log_interval = wandb.config.log_interval
    batch_size = wandb.config.batchs
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    if(wandb.config.label == 'sex'):
        classification = True
    else:
        classification = False

    train_dataset, test_dataset = get_data(wandb.config.path, wandb.config.organ, wandb.config.label)
    dataset_num_features = 3
    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    print("Got Data", flush=True)

    print('================')
    print('lr: {}'.format(wandb.config.lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(wandb.config.hidden_dim))
    print('num_gc_layers: {}'.format(wandb.config.num_gc_layers))
    print('================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update( {'device': device }, allow_val_change=True)

    model = InfoGraph(wandb.config.hidden_dim, wandb.config.num_gc_layers, classification, wandb.config.mlp).to(device)
    print(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=wandb.config.lr, momentum=wandb.config.momentum, weight_decay=wandb.config.weight_decay)
    elif wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

    if wandb.config.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)
    elif wandb.config.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    elif wandb.config.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

    print("Starting embeds eval before training", flush=True)
    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader)
    print('===== Before training =====', flush=True)
    res = evaluate_embedding(emb, y, classification)
    accuracies['logreg'].append(res[0])
    # accuracies['svc'].append(res[1])
    # accuracies['linearsvc'].append(res[2])
    accuracies['randomforest'].append(res[3])
    wandb.log({'randomforest_score': (accuracies['randomforest'])})

    print("Starting Training", flush=True)
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data.x, data.y, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        
        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)), flush=True)

        wandb.log({'Loss': (loss_all / len(dataloader)), 'epoch': epoch})
        wandb.watch(model)

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)
            res = evaluate_embedding(emb, y, classification)
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            accuracies['linearsvc'].append(res[2])
            accuracies['randomforest'].append(res[3])
            print(accuracies, flush=True)
            wandb.log({'randomforest_score': (accuracies['randomforest'][-1]), 'epoch': epoch})

        scheduler.step((loss_all / len(dataloader)))

    print(f"Last Random Forest Result: {(accuracies['randomforest'][-1])}", flush=True)
    if(wandb.config.save):
            torch.save(model, f'../models/{wandb.config.organ}_infograph_{wandb.config.label}.pt')
    # with open('unsupervised.log', 'a+') as f:
    #     s = json.dumps(accuracies)
    #     f.write('{},{},{},{},{}\n'.format(args.num_gc_layers, epochs, log_interval, wandb.config.lr, s))
