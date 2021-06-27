import json
from pathlib import Path


if __name__ == "__main__":

    path = Path("checkpoints")

    results = {}
    for dir in path.iterdir():
        if not dir.is_dir():
            continue

        with open(dir.joinpath("params.json"), "r") as f:
            data = json.load(f)
        results[data["test_max_accuracy"]["value"]] = str(dir)

    results = dict(sorted(results.items(), key=lambda x: x[0], reverse=True))
    s = [str(k) + "@" + v for k, v in results.items()]
    print("\n".join(s))