import argparse
import json
import matplotlib.pyplot as plt
import math
import numpy as np
import torch

from data.constants import SOURCE
from data.dataset import DependencyParsingDataset
from data.embedding import DataEmbedding
from models.dependency_parser_v1 import DependencyParserV1
from models.nllloss import DependencyParserNLLLoss
from pathlib import Path
from predict import predict
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.parser import Parser
from utils.torch import to_device


OPTIMIZERS = torch.optim.__dict__


def plot_stats(opts: argparse.Namespace, path: Path):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    start = math.ceil(len(opts.batch_losses) // opts.num_epochs)
    axs[0].plot(np.arange(len(opts.batch_losses), step=1), opts.batch_losses, label="batch loss")
    axs[0].plot(np.arange(start, len(opts.batch_losses), step=opts.n_batches - 1), opts.epoch_losses, label="epoch loss")
    axs[0].plot(np.arange(start, len(opts.batch_losses), step=opts.n_batches - 1), opts.test_losses, label="test loss")
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel("batch")
    axs[0].set_ylabel("loss")
    axs[0].set_title("train and test losses")

    axs[1].plot(opts.test_accuracies, label="test accuracy")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("accuracy[%]")
    axs[1].set_title("test accuracy vs. epoch")

    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    opts = Parser.train()
    print(f"Opts: {opts}")

    exp_dir = Path("checkpoints").joinpath(opts.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_embedding = DataEmbedding(corpora=[SOURCE["train"], SOURCE["test"]])  # SOURCE["comp"]
    # data_embedding.save(Path("assets/data_embedding.pth"))

    train_ds = DependencyParsingDataset(data_embedding, mode="train")
    train_dl = DataLoader(dataset=train_ds, batch_size=1, num_workers=opts.num_workers, drop_last=False, shuffle=True)
    test_ds = DependencyParsingDataset(data_embedding, mode="test", dropout=0.0)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, num_workers=opts.num_workers, drop_last=False, shuffle=False)

    model = DependencyParserV1(
        words_vocab_size=data_embedding.words_vocab_size(), poses_vocab_size=data_embedding.poses_vocab_size()
    )
    model = to_device(model, dtype=torch.float64)

    optimizer = OPTIMIZERS[opts.optimizer](model.parameters(), lr=opts.lr, **opts.optimizer_params)

    loss_fn = DependencyParserNLLLoss(dim=1, ignore_index=-1)
    loss_fn = to_device(loss_fn, dtype=torch.float64)

    opts.n_batches = len(train_dl) / opts.batch_size
    opts.batch_losses = []
    opts.epoch_losses = []
    opts.test_losses = []
    opts.test_accuracies = []

    for epoch in range(opts.num_epochs):

        optimizer.zero_grad()
        batch_loss = .0
        epoch_loss = .0
        for i, sentence in tqdm(enumerate(train_dl)):

            words_embedding_indices, poses_embedding_indices, heads_indices = sentence

            scores = model(words_embedding_indices, poses_embedding_indices)

            batch_loss += loss_fn(scores, heads_indices)

            if (i + 1) % opts.batch_size == 0:
                batch_loss /= opts.batch_size
                batch_loss.backward()

                optimizer.step()
                opts.batch_losses.append(batch_loss.item())
                epoch_loss += batch_loss.item()

                optimizer.zero_grad()
                batch_loss = .0

        opts.epoch_losses.append(epoch_loss / opts.n_batches)

        test_loss, test_accuracy = predict(test_dl, model, loss_fn)
        opts.test_losses.append(test_loss)
        opts.test_accuracies.append(test_accuracy)
        torch.save(model, exp_dir.joinpath(str(epoch).zfill(3) + ".pth"))

    plot_stats(opts, exp_dir.joinpath("statistics.pdf"))

    params_path = exp_dir.joinpath("params.json")
    with open(params_path, "w") as f:
        json.dump(vars(opts), f, indent=4)
