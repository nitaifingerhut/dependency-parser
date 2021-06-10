import torch

from data.constants import SOURCE
from data.dataset import DependencyParsingDataset
from data.embedding import DataEmbedding
from models.dependency_parser_v1 import DependencyParserV1
from models.nllloss import DependencyParserNLLLoss
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.parser import Parser
from utils.statistics import Statistics
from utils.torch import to_device


OPTIMIZERS = torch.optim.__dict__


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

    stats = Statistics()
    for epoch in range(opts.num_epochs):

        optimizer.zero_grad()
        loss = .0
        for i, sentence in tqdm(enumerate(train_dl)):

            words_embedding_indices, poses_embedding_indices, heads_indices = sentence

            scores = model(words_embedding_indices, poses_embedding_indices)

            loss += loss_fn(scores, heads_indices)

            if (i + 1) % opts.batch_size == 0:
                loss /= opts.batch_size
                loss.backward()

                optimizer.step()

                stats.append("train_batch_loss", loss.item())

                optimizer.zero_grad()
                loss = .0

        stats.step()