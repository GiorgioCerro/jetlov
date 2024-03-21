# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

# try:
#    from uproot_methods import TLorentzVector, TLorentzVectorArray
# except ImportError:
#    from uproot3_methods import TLorentzVector, TLorentzVectorArray
import time

import dgl
import networkx as nx
import numpy as np
import pandas as pd

# from .read_data import Jets
import torch
import torch.nn.functional as F
from dgl.transforms import remove_self_loop
from torch.utils.data import Dataset
from tqdm import tqdm

from .JetTree import JetTree, LundCoordinates

groomer = None
dump_number_of_nodes = False


class DGLGraphDatasetLund(Dataset):

    fill_secondary = True
    node_coordinates = "eta-phi"  # 'lund'

    def __init__(
        self, filepath_bkg, filepath_sig, nev=-1, n_samples=1000, algorithm="cambridge"
    ):
        super(DGLGraphDatasetLund, self).__init__()
        print(
            "Start loading dataset %s (bkg) and %s (sig)" % (filepath_bkg, filepath_sig)
        )
        tic = time.process_time()
        if filepath_bkg.suffix == ".gz":
            from .read_data_old import Jets
        else:
            from .read_data import Jets
        reader_bkg = Jets(filepath_bkg, nev, groomer=groomer, algorithm=algorithm)
        reader_sig = Jets(filepath_sig, nev, groomer=groomer, algorithm=algorithm)

        # attempt at using less memory
        self.data = []
        self.label = []
        for jet_id, jet in tqdm(enumerate(reader_bkg)):
            if len(self.data) >= (n_samples / 2):
                break
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [0]
        for jet_id, jet in tqdm(enumerate(reader_sig)):
            if len(self.data) >= n_samples:
                break
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [1]
        print(
            """ ... Total time to read input files + construct the graphs
        for {num} jets: {ts} seconds""".format(
                num=len(self.label), ts=time.process_time() - tic
            )
        )
        if dump_number_of_nodes:
            df = pd.DataFrame(
                {
                    "num_nodes": np.array([g.number_of_nodes() for g in self.data]),
                    "label": np.array(self.label),
                }
            )
            df.to_csv(
                "num_nodes_lund_net_ktmin_%s_deltamin_%s.csv"
                % (JetTree.ktmin, JetTree.deltamin)
            )
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def _build_tree(self, root):
        g = nx.Graph()
        # jet_p4 = TLorentzVector(*root.node)

        def _rec_build(nid, node):
            branches = (
                [node.harder, node.softer]
                if DGLGraphDatasetLund.fill_secondary
                else [node.harder]
            )
            for branch in branches:
                if branch is None or branch.lundCoord is None:
                    # stop when reaching the leaf nodes
                    # we do not add the leaf nodes to the graph/tree as
                    # they do not have Lund coordinates
                    ### ADDING LEAVES
                    # cid = g.number_of_nodes()
                    # node_p4 = TLorentzVector(*branch.node)
                    # spatialCoord = np.array(
                    #    [node_p4.x, node_p4.y, node_p4.z, node_p4.t],
                    #    dtype="float32")
                    # g.add_node(cid, coordinates=spatialCoord,
                    #    features=np.zeros(5, dtype="float32"))
                    # g.add_edge(cid, nid)
                    continue
                cid = g.number_of_nodes()
                if DGLGraphDatasetLund.node_coordinates == "lund":
                    spatialCoord = branch.lundCoord.state()[:2]
                else:
                    # node_p4 = TLorentzVector(*branch.node)
                    # spatialCoord = np.array(
                    #    [delta_eta_reflect(node_p4, jet_p4),
                    #     node_p4.delta_phi(jet_p4)],
                    #    dtype='float32')
                    # spatialCoord = np.array(
                    #    [node_p4.x, node_p4.y, node_p4.z, node_p4.t],
                    #    dtype="float32")
                    spatialCoord = branch.lundCoord.children_pmu(pmuform=False)
                g.add_node(
                    cid, coordinates=spatialCoord, features=branch.lundCoord.state()
                )
                g.add_edge(cid, nid)
                _rec_build(cid, branch)

        # add root
        if root.lundCoord is not None:
            # if DGLGraphDatasetLund.node_coordinates == 'lund':
            #    spatialCoord = root.lundCoord.state()[:2]
            # else:
            #    spatialCoord = np.zeros(2, dtype='float32')
            spatialCoord = root.lundCoord.children_pmu(
                pmuform=False
            )  # np.zeros(4, dtype="float32")
            g.add_node(0, coordinates=spatialCoord, features=root.lundCoord.state())
            _rec_build(0, root)
        else:
            # when a jet has only one particle (?)
            g.add_node(
                0,
                coordinates=np.zeros(8, dtype="float32"),
                features=np.zeros(LundCoordinates.dimension, dtype="float32"),
            )
        ret = dgl.from_networkx(g, node_attrs=["coordinates", "features"])
        # print(ret.number_of_nodes())
        return ret

    @property
    def num_features(self):
        return self.data[0].ndata["features"].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        return x, y


def delta_eta_reflect(constits_p4, jet_p4):
    deta = constits_p4.eta - jet_p4.eta
    return deta if jet_p4.eta > 0 else -deta


def pad_array(a, min_len=20, pad_value=0):
    if a.shape[0] < min_len:
        return F.pad(
            a, (0, 0, 0, min_len - a.shape[0]), mode="constant", value=pad_value
        )
    else:
        return a


class ReshowerDataset(Dataset):

    fill_secondary = True
    node_coordinates = "eta-phi"  # 'lund'

    def __init__(
        self, filepath_bkg, filepath_sig, nev=-1, n_samples=1000, algorithm="cambridge"
    ):
        super(ReshowerDataset, self).__init__()
        print(
            "Start loading dataset %s (bkg) and %s (sig)" % (filepath_bkg, filepath_sig)
        )
        tic = time.process_time()

        from .read_data import Jets, heparchy_adapter

        reader_bkg_shower = heparchy_adapter(filepath_bkg, shower_info=True)
        reader_sig_shower = heparchy_adapter(filepath_sig, shower_info=True)
        reader_bkg = Jets(filepath_bkg, nev, groomer=groomer, algorithm=algorithm)
        reader_sig = Jets(filepath_sig, nev, groomer=groomer, algorithm=algorithm)

        # attempt at using less memory
        self.data = []
        self.label = []
        self.shower_id = []
        for jet_id, jet in tqdm(enumerate(reader_bkg)):
            if len(self.data) >= (n_samples / 2):
                break
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [0]
            self.shower_id += [next(reader_bkg_shower)]

        for jet_id, jet in tqdm(enumerate(reader_sig)):
            if len(self.data) >= (n_samples):
                break
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [1]
            self.shower_id += [-next(reader_sig_shower) - 1]

        print(
            """ ... Total time to read input files + construct the graphs
            for {num} jets: {ts} seconds""".format(
                num=len(self.label), ts=time.process_time() - tic
            )
        )
        # if dump_number_of_nodes:
        #    df = pd.DataFrame({'num_nodes': np.array(
        #        [g.number_of_nodes() for g in self.data]),
        #         'label': np.array(self.label)})
        #    df.to_csv('num_nodes_lund_net_ktmin_%s_deltamin_%s.csv' % (
        #    JetTree.ktmin, JetTree.deltamin))
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.shower_id = torch.tensor(self.shower_id, dtype=torch.float32)

    def _build_tree(self, root):
        g = nx.Graph()
        # jet_p4 = TLorentzVector(*root.node)

        def _rec_build(nid, node):
            branches = (
                [node.harder, node.softer]
                if DGLGraphDatasetLund.fill_secondary
                else [node.harder]
            )
            for branch in branches:
                if branch is None or branch.lundCoord is None:
                    # stop when reaching the leaf nodes
                    # we do not add the leaf nodes to the graph/tree as
                    # they do not have Lund coordinates
                    ### ADDING LEAVES
                    # cid = g.number_of_nodes()
                    # node_p4 = TLorentzVector(*branch.node)
                    # spatialCoord = np.array(
                    #    [node_p4.x, node_p4.y, node_p4.z, node_p4.t],
                    #    dtype="float32")
                    # g.add_node(cid, coordinates=spatialCoord,
                    # features=np.zeros(5, dtype="float32"))
                    # g.add_edge(cid, nid)
                    continue
                cid = g.number_of_nodes()
                if DGLGraphDatasetLund.node_coordinates == "lund":
                    spatialCoord = branch.lundCoord.state()[:2]
                else:
                    # node_p4 = TLorentzVector(*branch.node)
                    # spatialCoord = np.array(
                    #    [delta_eta_reflect(node_p4, jet_p4),
                    #     node_p4.delta_phi(jet_p4)],
                    #    dtype='float32')
                    # spatialCoord = np.array(
                    #    [node_p4.x, node_p4.y, node_p4.z, node_p4.t],
                    #    dtype="float32")
                    spatialCoord = branch.lundCoord.children_pmu(pmuform=False)
                g.add_node(
                    cid, coordinates=spatialCoord, features=branch.lundCoord.state()
                )
                g.add_edge(cid, nid)
                _rec_build(cid, branch)

        # add root
        if root.lundCoord is not None:
            # if DGLGraphDatasetLund.node_coordinates == 'lund':
            #    spatialCoord = root.lundCoord.state()[:2]
            # else:
            #    spatialCoord = np.zeros(2, dtype='float32')
            spatialCoord = root.lundCoord.children_pmu(
                pmuform=False
            )  # np.zeros(4, dtype="float32")
            g.add_node(0, coordinates=spatialCoord, features=root.lundCoord.state())
            _rec_build(0, root)
        else:
            # when a jet has only one particle (?)
            g.add_node(
                0,
                coordinates=np.zeros(8, dtype="float32"),
                features=np.zeros(LundCoordinates.dimension, dtype="float32"),
            )
        ret = dgl.from_networkx(g, node_attrs=["coordinates", "features"])
        # print(ret.number_of_nodes())
        return ret

    @property
    def num_features(self):
        return self.data[0].ndata["features"].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        z = self.shower_id[i]
        return x, y, z
