import numpy as np
import torch
import torch.nn as nn

from assets.chu_liu_edmonds import decode_mst
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from utils.torch import to_numpy


def predict_dependencies_by_arc_scores(scores: np.ndarray) -> np.ndarray:
    scores[:, 0] = float("-inf")
    mst, _ = decode_mst(scores, len(scores), has_labels=False)
    return mst[1:]


def predict(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> Tuple[float, float]:
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
