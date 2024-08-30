import argparse
import ast
import random
import numpy as np
import wandb

import dgl

import torch as th
from dataset import get_data
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from evaluate_embeddings import evaluate_embedding
from model import InfoGraph


def argument():
    parser = argparse.ArgumentParser(description="InfoGraph")
    # data source params
    parser.add_argument("--path", type=str, default="../../../../../../vol/aimspace/users/wyo/registered_meshes/2000/")
    parser.add_argument("--organ", type=str, default="liver_mesh.ply")
    parser.add_argument("--save", type=ast.literal_eval, default=False)

    # training params
    parser.add_argument(
        "--epochs", type=int, default=30, help="Training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    # parser.add_argument(
    #     "--log_interval",
    #     type=int,
    #     default=1,
    #     help="Interval between two evaluations.",
    # )

    # model params
    parser.add_argument(
        "--n_layers",
        type=int,
        default=5,
        help="Number of graph convolution layers before each pooling.",
    )
    parser.add_argument(
        "--hid_dim", type=int, default=256, help="Hidden layer dimensionalities."
    )

    args = parser.parse_args()

    # check cuda
    # if args.gpu != -1 and th.cuda.is_available():
    #     args.device = "cuda:{}".format(args.gpu)
    # else:
    #     args.device = "cpu"

    return args


# def collate(samples):
#     """collate function for building graph dataloader"""

#     graphs, labels = map(list, zip(*samples))

#     # generate batched graphs and labels
#     batched_graph = dgl.batch(graphs)
#     batched_labels = th.tensor(labels)

#     n_graphs = len(graphs)
#     graph_id = th.arange(n_graphs)
#     graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

#     batched_graph.ndata["graph_id"] = graph_id

#     return batched_graph, batched_labels

def collate(samples):
    """collate function for building graph dataloader"""

    # graphs, labels = map(list, zip(*samples))

    # generate batched graphs and labels
    batched_graph = dgl.batch(samples)
    # batched_labels = th.tensor(labels)

    n_graphs = len(samples)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata["graph_id"] = graph_id

    return batched_graph


if __name__ == "__main__":
    # Step 1: Prepare graph data   ===================================== #
    args = argument()
    random.seed(42)
    np.random.seed(42)
    th.manual_seed(42)
    th.cuda.manual_seed(42)
    th.cuda.manual_seed_all(42)
    th.backends.cudnn.determinstic = True
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(args)

    run = wandb.init(
        project="digital_twin_infograph",
        entity="yussufwaly",
        notes="GAE",
        tags=[],
        config=args,
        )

    # load dataset from dgl.data.GINDataset
    # dataset = GINDataset(args.dataname, self_loop=False)
    train_graphs, val_graphs, test_graphs = get_data(wandb.config.path, wandb.config.organ)
    dataset = train_graphs

    wandb.config.update( {'device': device }, allow_val_change=True)

    # get graphs and labels
    # graphs, labels = map(list, zip(*dataset))

    # generate a full-graph with all examples for evaluation
    wholegraph = dgl.batch(dataset)
    wholegraph.ndata["feat"] = wholegraph.ndata["feat"].to(th.float32)

    # create dataloader for batch training
    dataloader = GraphDataLoader(
        dataset,
        batch_size=wandb.config.batch_size,
        collate_fn=collate,
        drop_last=True,
        shuffle=True,
    )

    in_dim = dataset[0].ndata["feat"].shape[1]

    # Step 2: Create model =================================================================== #
    model = InfoGraph(in_dim, wandb.config.hid_dim, wandb.config.n_layers)
    model = model.to(wandb.config.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=wandb.config.lr)

    # print("===== Before training ======")

    # wholegraph = wholegraph.to(args.device)
    # wholefeat = wholegraph.ndata["attr"]

    # emb = model.get_embedding(wholegraph, wholefeat).cpu()
    # res = evaluate_embedding(emb, labels, args.device)

    # # """ Evaluate the initialized embeddings """
    # # """ using logistic regression and SVM(non-linear) """
    # print("logreg {:4f}, svc {:4f}".format(res[0], res[1]))

    best_logreg = 0
    best_logreg_epoch = 0
    best_svc = 0
    best_svc_epoch = 0

    # Step 4: training epochs =============================================================== #
    for epoch in range(wandb.config.epochs):
        loss_all = 0
        model.train()

        for graph in dataloader:
            graph = graph.to(wandb.config.device)
            feat = graph.ndata["feat"]
            graph_id = graph.ndata["graph_id"]

            # n_graph = label.shape[0]

            optimizer.zero_grad()
            loss = model(graph, feat, graph_id)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        print("Epoch {}, Loss {:.4f}".format(epoch, loss_all))

        # if epoch % args.log_interval == 0:
        #     # evaluate embeddings
        #     model.eval()
        #     emb = model.get_embedding(wholegraph, wholefeat).cpu()
        #     res = evaluate_embedding(emb, labels, args.device)

        #     if res[0] > best_logreg:
        #         best_logreg = res[0]
        #         best_logreg_epoch = epoch

        #     if res[1] > best_svc:
        #         best_svc = res[1]
        #         best_svc_epoch = epoch

        #     print(
        #         "best logreg {:4f}, epoch {} | best svc: {:4f}, epoch {}".format(
        #             best_logreg, best_logreg_epoch, best_svc, best_svc_epoch
        #         )
        #     )
        wandb.log({'train_loss': loss_all, 'epoch': epoch})
        wandb.watch(model)

    print("Training End")

    if(wandb.config.save):
        th.save(model, f'../models/{wandb.config.organ}_infograph.pt')
    # print("best logreg {:4f} ,best svc {:4f}".format(best_logreg, best_svc))