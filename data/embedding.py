import functools
import operator
import torch

from collections import Counter, defaultdict
from data.constants import INDICES, SPECIAL_TOKENS
from pathlib import Path
from torchtext.vocab import Vocab
from typing import List, Optional, Tuple


class DataEmbedding(object):
    """
    Utility class for collecting corpora statistics
    and hold the associated words embeddings.
    """

    def __init__(
        self,
        corpora: List[Path],
        words_embeddings: Optional[Tuple[defaultdict, List[str], torch.Tensor]] = None,
        glove_vectors: Optional[str] = "glove.6B.300d",
    ):
        """

        Parameters
        ----------
        corpora: A list of paths to possible data files.
        words_embeddings [Optional]: Define specific words embedding.
        glove_vectors [Optional]: A string indicate GloVe to use from torchtext.vocab.
        """
        super().__init__()
        self.words_dict, self.poses_dict = self._init_vocabs(corpora)
        self.word_to_ind, self.ind_to_word, self.words_vectors = (
            words_embeddings if words_embeddings else self._init_words_embedding(glove_vectors)
        )
        self.pos_to_ind, self.ind_to_pos = self._init_pos_embedding()

    def __str__(self):
        return f"{self.__class__.__name__} :: #words = {len(self.words_dict)} :: #POSes = {len(self.poses_dict)}"

    def _init_vocabs(self, corpora: List[Path]) -> Tuple[defaultdict, defaultdict]:
        """
        Init vocabularies of all words and tags.

        Parameters
        ----------
        corpora: A list of paths to possible data files.

        Returns
        -------
        A dictionary with all possible words and their frequencies.
        A dictionary with all possible POSes and their frequencies.
        """
        words_dict, poses_dict = {}, {}
        for corpus in corpora:
            with open(corpus) as f:
                lines = [l.split("\t") for l in f.readlines() if l != "\n"]

                words = Counter([w[INDICES["word"]].lower() for w in lines])
                words_dict = dict(functools.reduce(operator.add, map(Counter, [words_dict, words])))

                POSes = Counter([p[INDICES["pos"]] for p in lines])
                poses_dict = dict(functools.reduce(operator.add, map(Counter, [poses_dict, POSes])))

        for token in SPECIAL_TOKENS:
            words_dict[token] = 0
            poses_dict[token] = 0

        return defaultdict(int, words_dict), defaultdict(int, poses_dict)

    def _init_words_embedding(
        self, glove_vectors: Optional[str] = "glove.6B.300d"
    ) -> Tuple[defaultdict, List[str], torch.Tensor]:
        """
        Init words embedding with GloVe from torchtext.vocab.

        Parameters
        ----------
        glove_vectors [Optional]: A string indicate GloVe to use from torchtext.vocab.

        Returns
        -------
        A dictionary with words to indices mapping.
        A list with indices to words mapping.
        A torch.Tensor with the embedding vectors of all possible words.
        """
        vocab = Vocab(Counter(self.words_dict), vectors=glove_vectors, specials=SPECIAL_TOKENS)
        return vocab.stoi, vocab.itos, vocab.vectors

    def _init_pos_embedding(self) -> Tuple[defaultdict, List[str]]:
        """
        Init POSes embedding with GloVe from torchtext.vocab.

        Returns
        -------
        A dictionary with POSes to indices mapping.
        A list with indices to POSes mapping.
        """
        ind_to_pos = []
        pos_to_ind = {}
        for i, pos in enumerate(sorted(self.poses_dict.keys())):
            pos_to_ind[pos] = i
            ind_to_pos.append(pos)
        return pos_to_ind, ind_to_pos

    def words_vocab_size(self):
        return len(self.words_dict)

    def poses_vocab_size(self):
        return len(self.poses_dict)

    # def save(self, path: Path):
    #     """
    #     Saves DataEmbedding to file.
    #
    #     Parameters
    #     ----------
    #     path: Path to save object to.
    #     """
    #     torch.save(self, path)
    #
    # @staticmethod
    # def load(path: Path):
    #     """
    #     Loades DataEmbedding from file.
    #
    #     Parameters
    #     ----------
    #     path: Path to load object from.
    #
    #     Returns
    #     -------
    #     DataEmbedding object.
    #     """
    #     return torch.load(path)
