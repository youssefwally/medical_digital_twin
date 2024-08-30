from . import random_permute

import numpy as np
import torch
from torch import Tensor
import networkx as nx


def minimize_and_pad(adj_matrix: np.ndarray, target_num_nodes: int) -> Tensor:
    adj_matrix = np.tril(adj_matrix)
    padding_size = target_num_nodes - adj_matrix.shape[0]
    padded_matrix = np.pad(
        adj_matrix,
        [(padding_size, 0), (0, padding_size)],
        "constant",
        constant_values=0.0,
    )
    torch_matrix = torch.Tensor(padded_matrix)
    if torch_matrix.ndim == 2:
        torch_matrix = torch_matrix[:, :, None]
    return torch_matrix


# Assumes the adjacency matrix is of normal size, that is it's width is equal
# to the number of nodes
def minimize_adj_matrix(adj_matrix: np.ndarray) -> Tensor:
    return minimize_and_pad(adj_matrix, adj_matrix.shape[0])
