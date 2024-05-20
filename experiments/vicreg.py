import pickle
import random
import sys
import time
import warnings
from pathlib import Path

import click
import dgl
import torch
import torch.nn as nn
import wandb
from jetlov.LundVicReg import LundVicReg
from jetlov.util import VicRegLoss, count_params, wandb_cluster_mode
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


def create_batches(args, dataset):
    counts = torch.unique(dataset.shower_id)
    track = [torch.where(dataset.shower_id == c)[0].tolist() for c in counts]
    indices = []
    step = 2
    for _ in range(int(len(dataset) / args.batch_size / 2)):
        if len(track) < args.batch_size:
            break
        indices.append([[], []])
        tags = random.sample(range(len(track)), args.batch_size)
        for tag in tags:
            elements = random.sample(track[tag], step)
            for s in range(step):
                track[tag].remove(elements[s])
                indices[-1][s].append(elements[s])
        track = [lst for lst in track if lst]
    return indices


def bkg_rejection_at_threshold(signal_eff, background_eff, sig_eff=0.5):
    """Background rejection at a given signal efficiency."""
    return 1 / (1 - background_eff[torch.argmin(torch.abs(signal_eff - sig_eff)) + 1])


def training_loop(args, device, model, optim, scheduler, dataset):
    total_loss = 0
    model.train()
    batches_idx = create_batches(args, dataset)

    num_batches = len(batches_idx)
    for idx in tqdm(range(num_batches)):
        optim.zero_grad()

        Z = []
        for _ in range(2):
            graph = []
            for event in batches_idx[idx][_]:
                _graph, _label, _id = dataset[event]
                graph.append(_graph)

            batch = dgl.batch(graph).to(device)
            Z.append(model(batch))

        loss = VicRegLoss(Z[0], Z[1])
        loss.backward()
        optim.step()

        total_loss += loss.item()
        wandb.log({"loss": loss.item()})

    print(f"loss: {(total_loss/num_batches):.5f}, ")
    # wandb.log({"loss": total_loss / num_batches,})

    scheduler.step()
    return model, total_loss / num_batches


def tune(args, model, dataset):
    with wandb.init(
        project=args.wandb_project,
        entity="office4005",
        config=dict(args),
        group=args.best_model_name[:-2] + "-" + args.task,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        loss_function = nn.CrossEntropyLoss()
        soft = nn.Softmax(dim=1)
        metric_scores = MetricCollection(
            dict(
                accuracy=metrics.BinaryAccuracy(),
                precision=metrics.BinaryPrecision(),
                recall=metrics.BinaryRecall(),
                f1=metrics.BinaryF1Score(),
                ROC=ROC(task="binary"),
                auroc=AUROC(task="binary"),
            )
        ).to(device)
        model = model.to(device)
        model.eval()

        fc_linear = torch.nn.Linear(1024, 2).to(device)
        # optim = torch.optim.Adam(fc_linear.parameters(), lr=args.lr)
        optim = torch.optim.SGD(
            fc_linear.parameters(), momentum=0.9, weight_decay=1e-6, lr=0.2
        )
        best_valid_accuracy = 0
        for epoch in range(args.tune_epochs):
            loss_temp = 0

            # creating batches
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            batch_size = 256
            batches_idx = [
                indices[i * batch_size : (i + 1) * batch_size]
                for i in range(len(indices) // batch_size)
            ]
            num_batches = len(batches_idx)
            for idx in tqdm(range(num_batches)):
                optim.zero_grad()
                graph, label = [], []
                for event in batches_idx[idx]:
                    _graph, _label, _id = dataset[event]
                    graph.append(_graph)
                    label.append(_label)

                label = torch.tensor(label).squeeze().long().to(device)
                batch = dgl.batch(graph).to(device)
                with torch.no_grad():
                    Z = model(batch)
                logit = fc_linear(Z)
                pred = soft(logit)[:, 1]

                loss = loss_function(logit, label)
                loss.backward()
                optim.step()

                loss_temp += loss.item()
                metric_scores.update(pred, label)

            loss_temp /= 2 * num_batches
            scores = metric_scores.compute()
            print(f"Completed epoch: {epoch}")

            wandb.log(
                {
                    "tune/accuracy": scores["accuracy"].item(),
                    "tune/precision": scores["precision"].item(),
                    "tune/recall": scores["recall"].item(),
                    "tune/auc": scores["auroc"],
                    "tune/loss": loss_temp,
                }
            )
            if scores["accuracy"] > best_valid_accuracy:
                print("!!! Saving the classifier !!!")
                best_valid_accuracy = scores["accuracy"]
                p = Path(args.logdir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save(
                    fc_linear.state_dict(),
                    p.joinpath(f"{args.best_model_name}-linear.pt"),
                )

            metric_scores.reset()

            if epoch > 0 and epoch % 8 == 0:
                for opt_params in optim.param_groups:
                    opt_params["lr"] *= 0.5

        return fc_linear


def train(args, model, dataset):  # , valid_dataset):
    with wandb.init(
        project=args.wandb_project,
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
            optim = torch.optim.Adam(
                model.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-5
            )
        else:
            optim = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5
            )
        lr_steps = [int(x) for x in args.lr_steps.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=lr_steps, gamma=0.5
        )

        ### start the training
        best_loss = 1e5
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch:n}")
            init = time.time()
            model, epoch_loss = training_loop(
                args,
                device,
                model,
                optim,
                scheduler,
                dataset,
            )
            if epoch == 0:
                print(f"epoch time: {(time.time() - init):.2f}")
            print(10 * "~")

            if epoch_loss < best_loss:
                print("!!! Saving the model !!!")
                best_loss = epoch_loss
                p = Path(args.logdir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p.joinpath(f"{args.best_model_name}.pt"))

        print(30 * "=")
        print("Training complete")
        return model


def eval(args, model, fc_linear, valid_dataset):
    with wandb.init(
        project=args.wandb_project,
        entity="office4005",
        config=dict(args),
        group=args.best_model_name[:-2] + "-" + args.task,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        fc_linear = fc_linear.to(device)

        state_dict = torch.load(
            "logs/" + args.best_model_name + ".pt", map_location="cpu"
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        state_dict = torch.load(
            "logs/" + args.best_model_name + "-linear.pt", map_location="cpu"
        )
        fc_linear.load_state_dict(state_dict)
        fc_linear.eval()

        loss_function = nn.CrossEntropyLoss()
        soft = nn.Softmax(dim=1)
        metric_scores = MetricCollection(
            dict(
                accuracy=metrics.BinaryAccuracy(),
                precision=metrics.BinaryPrecision(),
                recall=metrics.BinaryRecall(),
                f1=metrics.BinaryF1Score(),
                ROC=ROC(task="binary"),
                auroc=AUROC(task="binary"),
            )
        ).to(device)

        total_loss = 0
        indices = list(range(len(valid_dataset)))
        random.shuffle(indices)
        batches_idx = [
            indices[i * args.batch_size : (i + 1) * args.batch_size]
            for i in range(len(indices) // args.batch_size)
        ]
        num_batches = len(batches_idx)
        with torch.no_grad():
            for idx in tqdm(range(num_batches)):
                graph, label = [], []
                for event in batches_idx[idx]:
                    _graph, _label, _id = valid_dataset[event]
                    graph.append(_graph)
                    label.append(_label)

                label = torch.tensor(label).squeeze().long().to(device)
                batch = dgl.batch(graph).to(device)
                Z = model(batch)
                logit = fc_linear(Z)
                pred = soft(logit)[:, 1]

                loss = loss_function(logit, label)

                total_loss += loss.item()
                metric_scores.update(pred, label)

        total_loss /= num_batches
        scores = metric_scores.compute()

        fpr, tpr, threshs = scores["ROC"]
        eff_s = tpr
        eff_b = 1 - fpr
        bkg_rej_03 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.3)
        bkg_rej_05 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.5)
        bkg_rej_07 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.7)

        wandb.log(
            {
                "test/accuracy": scores["accuracy"].item(),
                "test/auc": scores["auroc"],
                "test/loss": total_loss,
                "test/bkg_rej_03": bkg_rej_03,
                "test/bkg_rej_05": bkg_rej_05,
                "test/bkg_rej_07": bkg_rej_07,
            }
        )

        print(f"Accuracy: {scores['accuracy'].item():.5f}")
        print(f"AUC: {scores['auroc']:.5f}")
        print(f"Loss: {total_loss:.5f}")
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
@click.option("--epochs", type=click.INT, default=100)
@click.option("--tune_epochs", type=click.INT, default=30)
@click.option(
    "--data_path", type=click.Path(exists=True), default="/scratch/gc2c20/data/"
)
@click.option("--train_samples", type=click.INT, default=1_000_000)
@click.option("--valid_samples", type=click.INT, default=100_000)
@click.option("--test_samples", type=click.INT, default=100_000)
@click.option("--best_model_name", type=click.STRING, default="best")
@click.option("--task", type=click.STRING, default="w-tag")
@click.option("--optim", type=click.STRING, default="adam")
@click.option("--architecture", type=click.STRING, default="lund")
def main(**kwargs):
    args = OmegaConf.create(kwargs)
    print("Working with the following configs:")
    for key, val in args.items():
        print(f"{key}: {val}")

    if args.architecture == "lund":
        from jetlov.jet_dataset import VicRegLundDataset as Dataset

        input_dims = 5
    else:
        from jetlov.jet_dataset import VicRegPartDataset as Dataset

        input_dims = 4
    args.logdir = "logs/"
    conv_params = [[32, 32], [64, 64], [128, 128]]
    # conv_params = [[64, 64], [128, 128]]
    model = LundVicReg(input_dims=input_dims, conv_params=conv_params)
    print(f"Model with {count_params(model)} trainable parameters")

    args.wandb_project = "vicreg"
    background = "bkg-vicreg.hdf5"
    signal = "ww-vicreg.hdf5"
    PATH = args.data_path
    train_dataset = Dataset(
        Path(PATH + "/train/" + background),
        Path(PATH + "/train/" + signal),
        nev=-1,
        n_samples=args.train_samples,
    )
    # valid_dataset = Dataset(
    #    Path(PATH + "/valid/valid_" + signal),
    #    nev=-1,
    #    n_samples=args.valid_samples,
    # )

    wandb_cluster_mode()
    # model = train(args, model, train_dataset)

    print("Starting the tuning for the classifier")
    state_dict = torch.load("logs/" + args.best_model_name + ".pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    fc_linear = tune(args, model, train_dataset)
    del train_dataset  # , valid_dataset

    test_dataset = Dataset(
        Path(PATH + "/test/test_" + background),
        Path(PATH + "/test/test_" + signal),
        nev=-1,
        n_samples=args.test_samples,
    )
    print("Evaluating the classifier")
    eval(args, model, fc_linear, test_dataset)
    del test_dataset


if __name__ == "__main__":
    sys.exit(main())
