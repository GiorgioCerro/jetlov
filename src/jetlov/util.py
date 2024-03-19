import operator as op

import dgl
import torch


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
