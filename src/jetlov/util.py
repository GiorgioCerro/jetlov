import operator as op

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl import backend as B
from scipy import sparse


def wandb_cluster_mode():
    """
    Get wandb key and turn wandb offline. Requires os imported?
    """
    import os

    key = os.environ.get("WANDB_KEY")
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_MODE"] = "offline"
    # os.environ['WANDB_MODE'] = 'online'


def collate_fn(batch):
    graphs, targets = zip(*batch)
    return dgl.batch(graphs), torch.hstack(targets)


def count_params(model: torch.nn.Module) -> int:
    param_flats = map(op.methodcaller("view", -1), model.parameters())
    param_shapes = map(op.attrgetter("shape"), param_flats)
    param_lens = map(op.itemgetter(0), param_shapes)
    return sum(param_lens)


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._open_file()


def segmented_knn_graph(x, k, segs):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix.  The predecessors of each point are its k-nearest
    neighbors.

    The matrices are concatenated along the first axis, and are segmented by
    ``segs``.  Each block would be transformed into a separate graph.  The
    graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    k : int
        The number of neighbors
    segs : iterable of int
        Number of points of each point set.
        Must sum up to the number of rows in ``x``.

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    n_total_points, _ = F.shape(x)
    offset = np.insert(np.cumsum(segs), 0, 0)

    h_list = F.split(x, segs, 0)
    # dst = [
    #    F.argtopk(pairwise_squared_distance(h_g), k, 1, descending=False) +
    #    offset[i]
    #    for i, h_g in enumerate(h_list)]
    dst = [
        F.argtopk(torch.cdist(h_g, h_g), k, 1, descending=False) + offset[i]
        for i, h_g in enumerate(h_list)
    ]
    dst = F.cat(dst, 0)
    src = F.arange(0, n_total_points).unsqueeze(1).expand(n_total_points, k)

    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    # !!! fix shape
    adj = sparse.csr_matrix(
        (F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))),
        shape=(n_total_points, n_total_points),
    )

    g = dgl.from_scipy(adj)
    return g


def knn_graph(x, k):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix. The predecessors of each point are its k-nearest
    neighbors.

    If a 3D tensor is given instead, then each row would be transformed into
    a separate graph.  The graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.

        If 2D, each row of ``x`` corresponds to a node.

        If 3D, a k-NN graph would be constructed for each row.  Then
        the graphs are unioned.
    k : int
        The number of neighbors

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    if B.ndim(x) == 2:
        x = B.unsqueeze(x, 0)
    n_samples, n_points, _ = B.shape(x)
    n_total_points = n_samples * n_points

    dist = torch.cdist(x, x)
    k_indices = B.argtopk(dist, k, 2, descending=False)
    dst = B.copy_to(k_indices, B.cpu())

    src = B.zeros_like(dst) + B.reshape(B.arange(0, n_points), (1, -1, 1))

    per_sample_offset = B.reshape(B.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = B.reshape(dst, (-1,))
    src = B.reshape(src, (-1,))
    adj = sparse.csr_matrix(
        (B.asnumpy(B.zeros_like(dst) + 1), (B.asnumpy(dst), B.asnumpy(src))),
        shape=(n_total_points, n_total_points),
    )

    g = dgl.from_scipy(adj)
    return g


def VicRegLoss(x, y, inv_coeff=25, std_coeff=25, cov_coeff=1):
    batch_size, num_features = x.shape
    # invariance loss
    inv_loss = F.mse_loss(x, y)

    # variance loss
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    # covariance loss
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(
        cov_y
    ).pow_(2).sum().div(num_features)

    loss = inv_coeff * inv_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
