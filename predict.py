import numpy as np
import torch
import torch.nn as nn

from assets.chu_liu_edmonds import decode_mst
from data.constants import INDICES, SOURCE
from data.dataset import DependencyParsingDataset
from data.embedding import DataEmbedding
from data.sentence import Sentence
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
        loss = 0.0
        accuracy = 0.0
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

    if torch.cuda.is_available():
        model = torch.load(opts.checkpoint)
    else:
        model = torch.load(opts.checkpoint, map_location=torch.device('cpu'))
    model = to_device(model)
    model.eval()

    sentences, _ = DependencyParsingDataset.init_sentences(SOURCE["comp"])
    for i, sentence in tqdm(enumerate(sentences)):
        (words_embedding_indices, poses_embedding_indices, _,) = DependencyParsingDataset.init_dataset_sentence(
            data_embedding=data_embedding, sentence=sentence, dropout=0.0
        )

        scores = model(words_embedding_indices.unsqueeze(0), poses_embedding_indices.unsqueeze(0))
        np_scores = to_numpy(scores.squeeze(0))
        pred_indices = predict_dependencies_by_arc_scores(np_scores)

        sentence.data = sentence.data[1:]
        sentence.update(pred_indices, "HEAD")

    with open(SOURCE["comp"], "r") as f:
        raw_sentences = f.read().split("\n\n")
    raw_sentences = [s.split("\n") for s in raw_sentences]

    updated_raw_sentences = []
    for raw_sentence, sentence in zip(raw_sentences, sentences):
        raw_sentence = [s.split("\t") for s in raw_sentence]
        updated_raw_sentence = []
        for entry, pred in zip(raw_sentence, sentence):
            entry[INDICES["head"]] = str(pred[Sentence.HEAD])
            updated_raw_sentence.append("\t".join(entry))
        updated_raw_sentences.append("\n".join(updated_raw_sentence))
    updated_raw_sentences += [""]

    pred_path = exp_dir.joinpath("comp.labeled")
    with open(pred_path, "w") as f:
        f.writelines("\n\n".join(updated_raw_sentences))
