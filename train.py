from data.dataset import POSDepsDataset
from pathlib import Path
from torch.utils.data import DataLoader
from utils.parser import Parser


if __name__ == "__main__":
    opts = Parser.train()
    print(f"Opts:")
    print(opts)
    print()

    exp_dir = Path("checkpoints").joinpath(opts.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = POSDepsDataset(mode="train")
    print(f"Loaded `{train_ds}`")
    print(f"Sentence Example:")
    print(train_ds[0].__repr__())
    print()

    train_dl = DataLoader(dataset=train_ds, batch_size=opts.batch_size, num_workers=opts.num_workers,
                          drop_last=False, shuffle=False)
    # print(f"Batch Example:")
    # samp = next(iter(train_dl))
    # print(samp)