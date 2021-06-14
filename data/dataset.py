import numpy as np
import torch

from data.constants import SOURCE, TOKENS
from data.embedding import DataEmbedding
from data.sentence import Sentence
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from utils.torch import to_device


class DependencyParsingDataset(Dataset):
    def __init__(
        self,
        data_embedding: DataEmbedding,
        mode: Optional[str] = "train",
        padding: Optional[bool] = False,
        dropout: Optional[float] = 0.1,
    ):
        """

        Parameters
        ----------
        data_embedding: An object with words and poses embeddings.
        mode: train/test/comp.
        padding: A boolean indicates whether to pad sentences to the same length.
        dropout: Amount (p) of words to drop (in practice; replace with `unknown` token).
        """

        super().__init__()

        if mode not in SOURCE.keys():
            raise ValueError(f"mode = {mode} not in {list(SOURCE.keys())}")

        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout = {dropout} not in [0, 1]")

        self.mode = mode
        self.file = SOURCE[mode]

        self.sentences, self.max_sentence_length = self.init_sentences(path=self.file)
        self.data = self._init_dataset(data_embedding=data_embedding, padding=padding, dropout=dropout)

    @staticmethod
    def init_sentences(path: Path) -> Tuple[List[Sentence], int]:
        """
        Initiates the dataset's sentences from file.

        Parameters
        ----------
        path: A path to file.

        Returns
        -------
        A tuple of (sentences, max sentence length).
        """

        with open(path, "r") as f:
            raw_sentences = f.read().split("\n\n")

        raw_sentences = [s.split("\n") for s in raw_sentences]
        raw_sentences = raw_sentences[:-1]  # Remove the empty line at the end of files.

        sentences = [Sentence(s) for s in raw_sentences]
        sentences = [i for i in sentences if i is not None]
        max_sentence_length = max([len(s) for s in sentences])

        return sentences, max_sentence_length

    def _init_dataset(
        self, data_embedding: DataEmbedding, padding: Optional[bool] = False, dropout: float = 0.0
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """
        Initiated the dataset from the sentences.

        Parameters
        ----------
        data_embedding: An object with words and poses embeddings.
        padding: A boolean indicates whether to pad sentences to the same length.
        dropout: Amount (p) of words to drop (in practice; replace with `unknown` token).

        Returns
        -------
        A list of samples' tuples.
        """
        data = []
        for sentence in self.sentences:
            data.append(self.init_dataset_sentence(data_embedding, sentence, padding=padding, dropout=dropout))
        return data

    @staticmethod
    def init_dataset_sentence(
        data_embedding: DataEmbedding, sentence: Sentence, padding: Optional[bool] = False, dropout: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Initiates a single sample from a sentence.

        Parameters
        ----------
        data_embedding: An object with words and poses embeddings.
        padding: A boolean indicates whether to pad sentences to the same length.
        dropout: Amount (p) of words to drop (in practice; replace with `unknown` token).

        Returns
        -------
        A tuple of (words embedding, poses embedding, heads indices, sentence length).
        """
        words_indices = []
        poses_indices = []
        heads_indices = []

        for i, w, p, h in sentence:
            if dropout > 0:
                dropout_probability = dropout / (data_embedding.words_dict[w] + dropout)
                if dropout_probability > np.random.rand():
                    w = TOKENS["unknown"]

            w_embedding = data_embedding.word_to_ind.get(w, data_embedding.word_to_ind[TOKENS["unknown"]])
            words_indices.append(w_embedding)

            p_embedding = data_embedding.pos_to_ind.get(p, data_embedding.pos_to_ind[TOKENS["unknown"]])
            poses_indices.append(p_embedding)

            heads_indices.append(h)

        return (
            to_device(torch.tensor(words_indices, requires_grad=False), dtype=torch.int64),
            to_device(torch.tensor(poses_indices, requires_grad=False), dtype=torch.int64),
            to_device(torch.tensor(heads_indices, requires_grad=False), dtype=torch.int64),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.data[idx]

    def __str__(self):
        return f"{self.__class__.__name__} :: mode = {self.mode} :: num sample = {len(self.data)}"
