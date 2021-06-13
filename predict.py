import numpy as np
import torch
import torch.nn as nn

from assets.chu_liu_edmonds import decode_mst
from data.constants import SOURCE
from data.dataset import DependencyParsingDataset
from data.embedding import DataEmbedding
from models.nllloss import DependencyParserNLLLoss
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from utils.parser import Parser
from utils.torch import to_numpy
from utils.torch import to_device


def predict_dependencies_by_arc_scores(scores: np.ndarray) -> np.ndarray:
    scores[:, 0] = float("-inf")
    mst, _ = decode_mst(scores, len(scores), has_labels=False)
    return mst[1:]


def predict_with_gt(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> Tuple[float, float]:
    with torch.no_grad():
        loss = .0
        accuracy = .0
        n_preds = 0
        for i, sentence in tqdm(enumerate(dataloader)):

            words_embedding_indices, poses_embedding_indices, heads_indices = sentence

            scores = model(words_embedding_indices, poses_embedding_indices)

            loss += loss_fn(scores, heads_indices).item()

            np_scores = to_numpy(scores.squeeze(0))
            pred_indices = predict_dependencies_by_arc_scores(np_scores)

            heads_indices = to_numpy(heads_indices.squeeze(0))
            heads_indices = heads_indices[1:]
            accuracy += np.sum(heads_indices == pred_indices)
            n_preds += pred_indices.size

    loss = loss / len(dataloader)
    accuracy = 100 * accuracy / n_preds
    return loss, accuracy


if __name__ == "__main__":
    opts = Parser.predict()
    print(f"Opts: {opts}")

    exp_dir = Path("checkpoints").joinpath(opts.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_embedding = DataEmbedding(corpora=[SOURCE["train"], SOURCE["test"], SOURCE["comp"]])
    # data_embedding.save(Path("assets/data_embedding.pth"))

    test_ds = DependencyParsingDataset(data_embedding, mode=opts.test_type, dropout=0.0)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, num_workers=opts.num_workers, drop_last=False, shuffle=False)

    model = torch.load(opts.checkpoint)
    model = to_device(model)

    loss_fn = DependencyParserNLLLoss(dim=1, ignore_index=-1)
    loss_fn = to_device(loss_fn, dtype=torch.float64)

    pass  # TODO: complete prediction and save to new file
