import sys
import torch 
import torch.nn as nn
from torchmetrics.functional.regression import mean_absolute_percentage_error as MAPE
import numpy as np

from jetron.regnet import RegNet
from jetron.util import count_params

from lundnet.jetron_dataset import DGLGraphDatasetLund as Dataset
from dgl.dataloading import GraphDataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(device, model, dataloader, val_dataloader, num_epochs=100):
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.1, 
                                                milestones=[50, 80])
    best_val_loss = 1e10
    for epoch in tqdm(range(num_epochs)):
        total_loss, total_mape = 0., 0.
        model.train()
        for graph, label in dataloader:
            pmu = graph.ndata["coordinates"].to(device)
            target = graph.ndata["features"].to(device)

            optim.zero_grad()
            pred = model(pmu)
            loss = loss_func(pred, target)
            mape = MAPE(pred, target)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_mape += mape.item()

        total_loss /= len(dataloader)
        total_mape /= len(dataloader)

        print(f"epoch: {epoch} - mse loss: {total_loss:.5f} - mape: {total_mape:.5f}")
        val_loss = validate(device, model, val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(20*"!")
            print(f"Saving best model with loss: {best_val_loss:.5f}")
            print(20*"!")
            torch.save(model.state_dict(), "logs/best_regression_xs.pt")

        scheduler.step()

    print("Training complete")
    print(50*"=")
    

def validate(device, model, dataloader, plot=False):
    model.eval()
    pred, target = [], []
    for graph, label in dataloader:
        _pmu = graph.ndata["coordinates"].to(device)
        _pred = model(_pmu).detach().cpu().numpy()
        _target = graph.ndata["features"].detach().numpy()
        for i in range(len(_pred)):
            pred.append(_pred[i])
            target.append(_target[i])

    loss_func = nn.MSELoss()
    loss = loss_func(torch.tensor(pred), torch.tensor(target))
    mape = MAPE(torch.tensor(pred), torch.tensor(target))
    print(f"Validation -- mse loss: {loss:.5f} -- mape: {mape:.5f}")
    target = np.array(target)
    pred = np.array(pred)
    if plot:
        for var in range(5):
            _loss = loss_func(torch.tensor(pred[:, var]), torch.tensor(target[:, var]))
            _mape = MAPE(torch.tensor(pred[:, var]), torch.tensor(target[:, var]))
            plt.scatter(target[:, var], pred[:, var])
            title = f"Lund_variable_#{var}"
            _min, _max = int(min(target[:, var])) - 1, int(max(target[:, var])) + 2
            plt.plot(range(_min, _max), range(_min, _max), c="r", linewidth=2)
            plt.title(title + f" - mse: {_loss:.5f}" + f" - mape: {_mape:.5f}")
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            plt.savefig(f"outputs/{title}")
            plt.close()

    return loss


def main():
    path = "/scratch/gc2c20/data/train/"
    dataset = Dataset(Path(path + "QCD_500GeV.json.gz"),
            Path(path + "WW_500GeV.json.gz"), nev=-1, n_samples=50_000)
    dataloader = GraphDataLoader(dataset, batch_size=4, shuffle=True)

    path = "/scratch/gc2c20/data/valid/"
    val_dataset = Dataset(Path(path + "valid_QCD_500GeV.json.gz"),
            Path(path + "valid_WW_500GeV.json.gz"), nev=-1, n_samples=5_000)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    model = RegNet().to(device)
    print(model)
    print(f"number of parameters: {count_params(model)}")

    model = train(device, model, dataloader, val_dataloader, num_epochs=100)

    #state_dict = torch.load("logs/best_reg_2_0.pt", map_location="cpu")
    #model.load_state_dict(state_dict)
    #val_loss = validate(device, model, val_dataloader, plot=True)
    

if __name__=="__main__":
    sys.exit(main())
