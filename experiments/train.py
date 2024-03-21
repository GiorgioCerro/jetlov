import pickle
import sys
import time
import warnings
from pathlib import Path

import click
import torch
import torch.nn as nn
import wandb
from dgl.dataloading import GraphDataLoader
from jetlov.composite import Composite
from jetlov.regnet import RegNet
from jetlov.util import collate_fn, count_params, wandb_cluster_mode

# from lundnet.dgl_dataset import DGLGraphDatasetLund as Dataset
from lundnet.jetron_dataset import DGLGraphDatasetLund as Dataset
from lundnet.LundNet import LundNet
from omegaconf import OmegaConf
from torchmetrics import AUROC, ROC, MetricCollection
from torchmetrics import classification as metrics
from tqdm import tqdm

NUM_GPUS = torch.cuda.device_count()
NUM_THREADS = 4
# torch.set_num_threads = NUM_THREADS
# torch.manual_seed(123)


warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
)


def load_model(args):
    ### loading the regression network
    model_reg = RegNet()

    ### loading the tagger
    conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]]
    fc_params = [(256, 0.1)]
    use_fusion = True
    input_dims = 5
    model_lund = LundNet(
        input_dims=input_dims,
        num_classes=2,
        conv_params=conv_params,
        fc_params=fc_params,
        use_fusion=use_fusion,
    )

    ### initialise the ensemble model
    # if args.task == "w-tag":
    #    state_dict_reg = torch.load("logs/best_regression.pt",
    #            map_location="cpu")
    #    state_dict_lund = torch.load("logs/best_tagger.pt",
    #            map_location="cpu")
    # else:
    #    state_dict_reg = torch.load("logs/regression_xs_top.pt",
    #            map_location="cpu")
    #    state_dict_lund = torch.load("logs/best_tagger_top.pt",
    #            map_location="cpu")
    # model_reg.load_state_dict(state_dict_reg)
    # state_dict_lund = torch.load("logs/lund-0.pt", map_location="cpu")
    # model_lund.load_state_dict(state_dict_lund)

    if args.architecture == "composite":
        state_dict_composite = torch.load("logs/jetlov-0.pt", map_location="cpu")
        composite_model = Composite(model_reg, model_lund)
        composite_model.load_state_dict(state_dict_composite)
        return composite_model
    else:
        return model_lund


def bkg_rejection_at_threshold(signal_eff, background_eff, sig_eff=0.5):
    """Background rejection at a given signal efficiency."""
    return 1 / (1 - background_eff[torch.argmin(torch.abs(signal_eff - sig_eff)) + 1])


def training_loop(args, device, model, optim, scheduler, dataloader, val_loader):
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    soft = nn.Softmax(dim=1)
    metric_scores = MetricCollection(
        dict(
            accuracy=metrics.BinaryAccuracy(),
            precision=metrics.BinaryPrecision(),
            recall=metrics.BinaryRecall(),
            f1=metrics.BinaryF1Score(),
        )
    ).to(device)
    total_loss = 0

    model.train()
    for graph, label in tqdm(dataloader):
        graph = graph.to(device)
        label = label.to(device).squeeze().long()

        optim.zero_grad()
        logits = model(graph)

        loss = loss_function(logits, label)
        loss.backward()
        optim.step()

        pred = soft(logits)[:, 1]
        metric_scores.update(pred, label)
        total_loss += loss.item()

    scores = metric_scores.compute()
    # evaluation
    scores_eval, val_loss = eval(args, device, model, val_loader)
    fpr, tpr, threshs = scores_eval["ROC"]
    eff_s = tpr
    eff_b = 1 - fpr
    bkg_rej_03 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.3)
    bkg_rej_05 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.5)
    bkg_rej_07 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.7)

    print(
        f"loss: {(total_loss/len(dataloader)):.5f}, "
        f"accuracy: {scores['accuracy'].item():.1%}, "
        f"precision: {scores['precision'].item():.1%}, "
        f"recall: {scores['recall'].item():.1%}, "
        f"f1: {scores['f1'].item():.1%}, "
        f"\n validation: "
        f"val_acc: {scores_eval['accuracy'].item():.1%}, "
        f"val_auc: {scores_eval['auroc']:.1%}, "
        f"val_loss: {val_loss:.5f}, "
    )

    wandb.log(
        {
            "accuracy": scores["accuracy"].item(),
            "precision": scores["precision"].item(),
            "recall": scores["recall"].item(),
            "f1": scores["f1"].item(),
            "loss": total_loss / len(dataloader),
            "val/accuracy": scores_eval["accuracy"].item(),
            "val/auc": scores_eval["auroc"],
            "val/loss": val_loss,
            "val/bkg_rej_03": bkg_rej_03,
            "val/bkg_rej_05": bkg_rej_05,
            "val/bkg_rej_07": bkg_rej_07,
        }
    )

    scheduler.step()
    metric_scores.reset()
    return model, scores_eval["accuracy"]


def eval(args, device, model, dataloader):
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    soft = nn.Softmax(dim=1)
    metric_scores_eval = MetricCollection(
        dict(
            accuracy=metrics.BinaryAccuracy(),
            precision=metrics.BinaryPrecision(),
            recall=metrics.BinaryRecall(),
            f1=metrics.BinaryF1Score(),
            ROC=ROC(task="binary"),
            auroc=AUROC(task="binary"),
        )
    ).to(device)
    loss_temp = 0

    model.eval()
    with torch.no_grad():
        for graph, label in dataloader:
            graph = graph.to(device)
            label = label.to(device).squeeze().long()

            logits = model(graph)
            pred = soft(logits)[:, 1]
            metric_scores_eval.update(pred, label)

            loss_temp += loss_function(logits, label).item()

    scores_eval = metric_scores_eval.compute()
    return scores_eval, loss_temp / len(dataloader)


def train(args, model, dataset, valid_dataset):
    with wandb.init(
        project="prova",
        entity="office4005",
        config=dict(args),
        group=args.best_model_name[:-2] + "-" + args.task,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)

        ### optimizer and scheduler
        if args.optim == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        else:
            optim = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5
            )
        lr_steps = [int(x) for x in args.lr_steps.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, gamma=0.5, milestones=lr_steps
        )

        ### preparing dataloaders
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

        ### start the training
        best_valid_acc = 0
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch:n}")
            init = time.time()
            model, valid_acc = training_loop(
                args, device, model, optim, scheduler, dataloader, val_loader
            )
            if epoch == 0:
                print(f"epoch time: {(time.time() - init):.2f}")
            print(10 * "~")

            if valid_acc > best_valid_acc:
                print("!!! Saving the model !!!")
                best_valid_acc = valid_acc
                p = Path(args.logdir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p.joinpath(f"{args.best_model_name}.pt"))

        print(30 * "=")
        print("Training complete")
        del dataloader, val_loader
        return model


def test(args, model, test_dataset):
    with wandb.init(
        project="prova",
        entity="office4005",
        config=dict(args),
        group=args.best_model_name[:-2] + "-" + args.task,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)
        test_loader = GraphDataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        state_dict = torch.load(
            "logs/" + args.best_model_name + ".pt", map_location="cpu"
        )
        model.load_state_dict(state_dict)
        model.to(device)

        scores_test, test_loss = eval(args, device, model, test_loader)
        fpr, tpr, threshs = scores_test["ROC"]
        eff_s = tpr
        eff_b = 1 - fpr
        bkg_rej_03 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.3)
        bkg_rej_05 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.5)
        bkg_rej_07 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.7)

        wandb.log(
            {
                "test/accuracy": scores_test["accuracy"].item(),
                "test/auc": scores_test["auroc"],
                "test/loss": test_loss,
                "test/bkg_rej_03": bkg_rej_03,
                "test/bkg_rej_05": bkg_rej_05,
                "test/bkg_rej_07": bkg_rej_07,
            }
        )

        print(f"Accuracy: {scores_test['accuracy'].item():.5f}")
        print(f"AUC: {scores_test['auroc']:.5f}")
        print(f"Loss: {test_loss:.5f}")
        print(f"Inv_bkg_at_sig_03: {bkg_rej_03:.5f}")
        print(f"Inv_bkg_at_sig_05: {bkg_rej_05:.5f}")
        print(f"Inv_bkg_at_sig_07: {bkg_rej_07:.5f}")

        with open(
            "outputs/" + args.best_model_name + "-" + args.task + ".pickle", "wb"
        ) as f:
            pickle.dump(
                {
                    "signal_eff": eff_s.to(torch.device("cpu")),
                    "background_eff": eff_b.to(torch.device("cpu")),
                    "threshs": threshs.to(torch.device("cpu")),
                    "description": str(args),
                },
                f,
            )


@click.command()
@click.option("--lr", type=click.FLOAT, default=0.001)
@click.option("--lr_steps", type=click.STRING, default="10,20")
@click.option("--batch_size", type=click.INT, default=256)
@click.option("--epochs", type=click.INT, default=30)
@click.option(
    "--data_path", type=click.Path(exists=True), default="/scratch/gc2c20/data/"
)
@click.option("--train_samples", type=click.INT, default=1_000_000)
@click.option("--valid_samples", type=click.INT, default=100_000)
@click.option("--test_samples", type=click.INT, default=100_000)
@click.option("--best_model_name", type=click.STRING, default="best")
@click.option("--task", type=click.STRING, default="w-tag")
@click.option("--optim", type=click.STRING, default="adam")
@click.option("--architecture", type=click.STRING, default="composite")
@click.option("--runs", type=click.INT, default=1)
def main(**kwargs):
    args = OmegaConf.create(kwargs)
    print("Working with the following configs:")
    for key, val in args.items():
        print(f"{key}: {val}")

    background = "bkg-0.hdf5"  # "QCD_500GeV.json.gz"
    if args.task == "w-tag":
        signal = "WW_500GeV.json.gz"
    elif args.task == "top-tag":
        signal = "top-0.hdf5"  # "Top_500GeV.json.gz"
    else:
        signal = "Quark_500GeV.json.gz"
        background = "Gluon_500GeV.json.gz"
    PATH = args.data_path
    train_dataset = Dataset(
        Path(PATH + "/train/" + background),
        Path(PATH + "/train/" + signal),
        nev=-1,
        n_samples=args.train_samples,
    )
    # valid_dataset = Dataset(Path(PATH+"/valid/valid_"+background),
    #                        Path(PATH+"/valid/valid_"+signal),
    #                        nev=-1, n_samples=args.valid_samples)
    valid_dataset = Dataset(
        Path(PATH + "/valid/valid_" + background),
        Path(PATH + "/valid/valid_" + signal),
        nev=-1,
        n_samples=args.valid_samples,
    )

    args.logdir = "logs/"
    model = load_model(args)
    print(f"Model with {count_params(model)} trainable parameters")

    wandb_cluster_mode()
    for i in range(args.runs):
        model = load_model(args)
        args.best_model_name += f"-{str(i)}"
        model = train(args, model, train_dataset, valid_dataset)
        args.best_model_name = args.best_model_name[:-2]
    del train_dataset, valid_dataset

    # test_dataset = Dataset(Path(PATH+"/test/test_"+background),
    #                        Path(PATH+"/test/test_"+signal),
    #                        nev=-1, n_samples=args.test_samples)
    test_dataset = Dataset(
        Path(PATH + "/test/test_bkg-0.hdf5"),
        Path(PATH + "/test/test_top-0.hdf5"),
        nev=-1,
        n_samples=args.test_samples,
    )
    for i in range(args.runs):
        args.best_model_name += f"-{str(i)}"
        model = load_model(args)
        test(args, model, test_dataset)
        args.best_model_name = args.best_model_name[:-2]
    del test_dataset


if __name__ == "__main__":
    sys.exit(main())
