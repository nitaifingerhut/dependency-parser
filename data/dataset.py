from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple


class DatasetSentence(object):

    DS_LOC = 0
    DS_WORD = 1
    DS_POS = 3
    DS_HEAD = 6

    S_LOC = 0
    S_WORD = 1
    S_POS = 2
    S_HEAD = 3

    @classmethod
    def from_file(cls, sentence: List[str], add_root: bool = True):
        entries = [e.split('\t') for e in sentence]
        locs = [int(e[cls.DS_LOC]) for e in entries]
        words = [e[cls.DS_WORD] for e in entries]
        POSs = [e[cls.DS_POS] for e in entries]
        heads = [int(e[cls.DS_HEAD]) for e in entries]
        return cls(locs, words, POSs, heads, add_root)

    def __init__(
        self,
        locs: List[int],
        words: List[str],
        POSs: List[str],
        heads: List[int],
        add_root: bool = True
    ):
        self.base_len = len(locs)
        if not all(len(i) == self.base_len for i in (words, POSs, heads)):
            raise ValueError

        self.locs = locs if not add_root else [0] + locs
        self.words = words if not add_root else ["ROOT"] + words
        self.POSs = POSs if not add_root else [""] + POSs
        self.heads = heads if not add_root else [-1] + heads

    def __len__(self):
        return self.base_len

    def __getitem__(self, idx) -> Tuple[int, str, str, int]:
        if not 0 <= idx < self.base_len:
            raise IndexError
        return self.locs[idx], self.words[idx], self.POSs[idx], self.heads[idx]

    def __str__(self):
        return f"DatasetSentence: {self.base_len}"

    def __repr__(self):
        title = "| Index |      Word      |   POS   | Head |"
        hline = "==========================================="
        zip_data = zip(self.locs, self.words, self.POSs, self.heads)
        entries = [title, hline] + [f"| {l:<5} | {w:<14} | {p:<7} | {h:<4} |" for l, w, p, h in zip_data] + [hline]
        entries_str = "\n".join(entries)
        return entries_str


SOURCE = {
    "train": Path.cwd().joinpath("assets/train.labeled"),
    "test": Path.cwd().joinpath("assets/test.labeled"),
    "comp": Path.cwd().joinpath("assets/comp.unlabeled")
}


class POSDepsDataset(Dataset):

    def __init__(self, mode: str = "train"):

        super().__init__()

        if mode not in SOURCE:
            raise ValueError(f"mode = {mode} not in {list(SOURCE.keys())}")

        self.mode = mode
        self.file = SOURCE[mode]

        with open(self.file, "r") as f:
            sentences = f.read().split("\n\n")
            sentences = [s.split("\n") for s in sentences]
            sentences = sentences[:-1]  # Remove the empty line at the end of files.
            self.data = [DatasetSentence.from_file(s) for s in sentences]
            self.data = [i for i in self.data if i is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> DatasetSentence:
        # TODO: need to assign values to tensors + words to embedding vectors
        return self.data[idx]

    def __str__(self):
        return f"POSDepsDataset :: mode = {self.mode} | num sample = {len(self.data)}"
