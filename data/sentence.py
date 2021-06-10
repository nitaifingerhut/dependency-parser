from data.constants import INDICES, TOKENS
from typing import List, Tuple


class Sentence(object):
    def __init__(self, sentence: List[str], add_root: bool = True):

        super(Sentence, self).__init__()

        sentence = [s.split("\t") for s in sentence]
        self.data = [
            (s[INDICES["index"]], s[INDICES["word"]], s[INDICES["pos"]], int(s[INDICES["head"]])) for s in sentence
        ]
        if add_root:
            self.data.insert(0, (0, TOKENS["root"], TOKENS["root"], -1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[int, str, str, int]:
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
