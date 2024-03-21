from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from heparchy.read.hdf import HdfReader
from jetlov.jet_dataset import DGLGraphDatasetLund as Dataset
from jetlov.LundNet import LundNet
from tqdm import tqdm

conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]]
fc_params = [(256, 0.1)]
use_fusion = False
input_dims = 5
modelL = LundNet(
    input_dims=input_dims,
    num_classes=2,
    conv_params=conv_params,
    fc_params=fc_params,
    use_fusion=use_fusion,
)
state_dict = torch.load("../experiments/logs/lund-default-0.pt", map_location="cpu")
modelL.load_state_dict(state_dict)
modelL.eval()


use_fusion = False
model = LundNet(
    input_dims=input_dims,
    num_classes=2,
    conv_params=conv_params,
    fc_params=fc_params,
    use_fusion=use_fusion,
)
state_dict = torch.load("../experiments/logs/lund-shower-0.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


# model0 = RegNet()
# model1 = LundNet(input_dims=input_dims, num_classes=2,
#                conv_params=conv_params, fc_params=fc_params,
#                use_fusion=use_fusion)
# model = Composite(model0, model1)
# state_dict = torch.load(
#    "../experiments/logs/jetlov-0.pt", map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval()
#
# model00 = RegNet()
# model11 = LundNet(input_dims=input_dims, num_classes=2,
#                conv_params=conv_params, fc_params=fc_params,
#                use_fusion=use_fusion)
# modelL = Composite(model00, model11)
# state_dict = torch.load(
#    "../experiments/logs/jetlov-reshower-0.pt", map_location="cpu")
# modelL.load_state_dict(state_dict)
# modelL.eval()

print("Models uploaded")

tag = []
nn = 20000
path = "/scratch/gc2c20/data/test/"
dataset = Dataset(
    Path(path + "/test_bkg-0.hdf5"),
    Path(path + "/test_top-0.hdf5"),
    nev=-1,
    n_samples=nn,
)
with HdfReader(path + "/test_bkg-0.hdf5") as file:
    process = file["background"]
    for i in tqdm(range(int(nn / 2))):
        event = process[i]
        tag.append(event.custom["shower_id"][0])
with HdfReader(path + "/test_top-0.hdf5") as file:
    process = file["signal"]
    for i in tqdm(range(int(nn / 2))):
        event = process[i]
        tag.append(event.custom["shower_id"][0])

print("Data loaded and ready")

_tag = 0
soft = torch.nn.Softmax(dim=1)
pred, target, predL = [[]], [[]], [[]]

for idx, (graph, label) in tqdm(enumerate(dataset)):
    logits = soft(model(graph.clone()))[0][1].item()
    logitsL = soft(modelL(graph.clone()))[0][1].item()
    if tag[idx] == _tag:
        pred[-1].append(logits)
        predL[-1].append(logitsL)
        target[-1].append(True) if label.item() == 1 else target[-1].append(False)
    else:
        _tag = tag[idx]
        pred.append([logits])
        predL.append([logitsL])
        target.append([True]) if label.item() == 1 else target.append([False])


mean = np.array([np.mean(p) for p in pred])
std = np.array([np.std(p) for p in pred])
meanL = np.array([np.mean(p) for p in predL])
stdL = np.array([np.std(p) for p in predL])
mask = np.array([tar[0] for tar in target])


name = "LundNet-shower"
nameL = "LundNet"
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax = ax.flatten()

ax[0].scatter(2 * np.arange(len(std[mask])), std[mask], c="#FFA650", label=f"{name}")
ax[0].scatter(2 * np.arange(len(stdL[mask])), stdL[mask], c="#50AAFF", label=f"{nameL}")
ax[0].axhline(y=np.mean(std[mask]), color="#FFA650", linestyle="--")
ax[0].axhline(y=np.mean(stdL[mask]), color="#50AAFF", linestyle="--")
ax[0].set_title("Signal")
ax[0].legend()
ax[0].set_xlabel("Events")
ax[0].set_ylabel("Average score over reshowered events")

# ax[1].axhline(y=0.5, color='r', linestyle='--')
# ax[1].errorbar(2* np.arange(len(mean[~mask])), mean[~mask],
#        yerr=std[~mask], c='#FFA650', fmt='o', label=f"{name}")
# ax[1].errorbar(2 * np.arange(len(meanL[~mask])) + 0.5, meanL[~mask],
#        yerr=stdL[~mask], c='#50AAFF', fmt='o', label=f"{name}2.0")
ax[1].scatter(2 * np.arange(len(std[~mask])), std[~mask], c="#FFA650", label=f"{name}")
ax[1].scatter(
    2 * np.arange(len(stdL[~mask])), stdL[~mask], c="#50AAFF", label=f"{nameL}"
)
ax[1].axhline(y=np.mean(std[~mask]), color="#FFA650", linestyle="--")
ax[1].axhline(y=np.mean(stdL[~mask]), color="#50AAFF", linestyle="--")
ax[1].set_title("Background")
ax[1].legend()


ax[2].scatter(2 * np.arange(len(mean[mask])), mean[mask], c="#FFA650", label=f"{name}")
ax[2].scatter(
    2 * np.arange(len(meanL[mask])), meanL[mask], c="#50AAFF", label=f"{nameL}"
)
ax[2].axhline(y=np.mean(mean[mask]), color="#FFA650", linestyle="--")
ax[2].axhline(y=np.mean(meanL[mask]), color="#50AAFF", linestyle="--")


ax[3].scatter(
    2 * np.arange(len(mean[~mask])), mean[~mask], c="#FFA650", label=f"{name}"
)
ax[3].scatter(
    2 * np.arange(len(meanL[~mask])), meanL[~mask], c="#50AAFF", label=f"{nameL}"
)
ax[3].axhline(y=np.mean(mean[~mask]), color="#FFA650", linestyle="--")
ax[3].axhline(y=np.mean(meanL[~mask]), color="#50AAFF", linestyle="--")


plt.savefig(f"reshower_top_{name}")
plt.close()
