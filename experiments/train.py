import sys, pickle
import torch
import torch.nn as nn
import wandb
import time
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import click

from jetron.jetnet import JetronNet
from lundnet.dgl_dataset import DGLGraphDatasetLund as Dataset

from dgl.dataloading import GraphDataLoader
from jetron.util import collate_fn, count_params, wandb_cluster_mode

NUM_GPUS = torch.cuda.device_count()
NUM_THREADS = 4
torch.set_num_threads = NUM_THREADS
#torch.manual_seed(123)

import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")


def training_loop(args, device, model, optim, scheduler, dataloader, val_loader):
    loss_function = nn.MSELoss()
    total_loss = 0

    model.train()
    for graph, label in tqdm(dataloader):
        optim.zero_grad()

        prediction = model(graph.to(device), graph.ndata['coordinates'].to(device))

        loss = loss_function(prediction, graph.ndata['features'].to(device))
        loss.backward()
        optim.step()

        total_loss += loss.item()


    print(
        f"loss: {(total_loss/len(dataloader)):.5f}, "
    )

    val_loss = evaluate(device, model, val_loader)
    wandb.log({
       "loss": total_loss/len(dataloader),
        "val_loss": val_loss,
    })

    scheduler.step()
    return model, val_loss


def evaluate(device, model, dataloader):
    loss_function = nn.MSELoss()
    loss_temp = 0

    model.eval()
    with torch.no_grad():
        for graph, label in dataloader:
            prediction = model(graph.to(device), 
                            graph.ndata["coordinates"].to(device))
            loss_temp += loss_function(prediction, 
                                    graph.ndata["features"]).item()
    
    return loss_temp/len(dataloader)


def train(args, dataset, valid_dataset):
    with wandb.init(project="jetron-learningLund", entity="office4005", 
            config=dict(args), group=args.best_model_name):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = JetronNet()
       
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        #optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.1,
                                                        milestones=[25, 27])
        print(f"Model with {count_params(model)} trainable parameters")
        dataloader = GraphDataLoader(
                dataset=dataset, 
                batch_size=args.batch_size, 
                num_workers=0,
                drop_last=False,
                shuffle=True, 
                pin_memory=True,
                collate_fn=collate_fn,
            )
        val_loader = GraphDataLoader(
                dataset=valid_dataset, 
                batch_size=args.batch_size,
                num_workers=0,
                drop_last=False,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        best_valid_loss = 1e10
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch:n}")
            init = time.time()
            model, valid_loss = training_loop(args, device, model, optim,
                                            scheduler, dataloader, val_loader)
            if epoch == 0:
                print(f"epoch time: {(time.time() - init):.2f}")
            print(10*"~")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                p = Path(args.logdir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p.joinpath(f"{args.best_model_name}.pt"))

        
        print(30*"=")
        print(f"Training complete")
        del dataset, dataloader, valid_dataset, val_loader
        PATH = args.data_path
        test_dataset = Dataset(Path(PATH+"/test/test_QCD_500GeV.json.gz"), 
                            Path(PATH+"/test/test_Top_500GeV.json.gz"), 
                            nev=-1, n_samples=args.test_samples)
        
        test_loader = GraphDataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            drop_last=False, 
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        state_dict = torch.load("logs/" + args.best_model_name + ".pt", 
                                map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)

        test_loss = evaluate(device, model, test_loader)
        print(f"Test completed -- test loss: {test_loss:.5f}")
       


@click.command()
@click.option("--lr", type=click.FLOAT, default=0.001)
@click.option("--batch_size", type=click.INT, default=32)
@click.option("--epochs", type=click.INT, default=20)
@click.option("--data_path", type=click.Path(exists=True), 
            default="/scratch/gc2c20/data/")
@click.option("--train_samples", type=click.INT, default=1_000)
@click.option("--valid_samples", type=click.INT, default=1_000)
@click.option("--test_samples", type=click.INT, default=1_000)
def main(**kwargs):
    args = OmegaConf.create(kwargs)
    print(f"Working with the following configs:")
    for key, val in args.items():
        print(f"{key}: {val}")

    PATH = args.data_path
    train_dataset = Dataset(Path(PATH+"/train/QCD_500GeV.json.gz"), 
                            Path(PATH+"/train/WW_500GeV.json.gz"), 
                            nev=-1, n_samples=args.train_samples)
    valid_dataset = Dataset(Path(PATH+"/valid/valid_QCD_500GeV.json.gz"), 
                            Path(PATH+"/valid/valid_WW_500GeV.json.gz"), 
                            nev=-1, n_samples=args.valid_samples)
    
    args.logdir = "logs/"
    args.best_model_name = "best_model"

    wandb_cluster_mode()
    train(args, train_dataset, valid_dataset)


if __name__=="__main__":
    sys.exit(main())
