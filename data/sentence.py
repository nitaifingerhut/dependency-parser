from data.constants import INDICES, TOKENS
from typing import Iterable, List


class Sentence(object):

    INDEX = 0
    WORD = 1
    POS = 2
    HEAD = 3

    def __init__(self, sentence: List[str], add_root: bool = True):

        super(Sentence, self).__init__()

        sentence = [s.split("\t") for s in sentence]
        try:
            self.data = [
                [s[INDICES["index"]], s[INDICES["word"]], s[INDICES["pos"]], int(s[INDICES["head"]])] for s in sentence
            ]
        except ValueError:
            self.data = [[s[INDICES["index"]], s[INDICES["word"]], s[INDICES["pos"]], -1] for s in sentence]
        if add_root:
            self.data.insert(0, [0, TOKENS["root"], TOKENS["root"], -1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> List:
        if not 0 <= index < len(self.data):
            raise IndexError
        return self.data[index]

    def __iter__(self):
        self.iter = iter(self.data)
        return self

    def __next__(self):
        return next(self.iter)

    def __repr__(self):
        return f"{self.__class__.__name__} :: length = {len(self.data)}"

    def __str__(self):
        title = "| Index |      Word      |   POS   | Head |"
        hline = "==========================================="
        entries = [title, hline] + [f"| {a:<5} | {b:<14} | {c:<7} | {d:<4} |" for a, b, c, d in self.data] + [hline]
        sentence_str = "\n".join(entries)
        return sentence_str

    def update(self, values: Iterable, attr: str):
        if len(values) != len(self.data):
            raise ValueError

        key = getattr(self, attr)
        for i in range(len(self.data)):
            self.data[i][key] = values[i]
