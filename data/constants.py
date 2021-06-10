from pathlib import Path


INDICES = {
    "index": 0,
    "word": 1,
    "pos": 3,
    "head": 6,
}

SOURCE = {
    "train": Path("assets/train.labeled"),
    "test": Path("assets/test.labeled"),
    "comp": Path("assets/comp.unlabeled"),
}

TOKENS = {
    "pad": "<pad>",
    "root": "<root>",
    "unknown": "<???>",
}
SPECIAL_TOKENS = (TOKENS["pad"], TOKENS["root"], TOKENS["unknown"])
