import abc
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.dependency_parser import MODELS
from pathlib import Path


class ParseKWArgs(argparse.Action):
    @abc.abstractmethod
    def parse_value(self, value: str):
        raise NotImplementedError

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = self.parse_value(value)


class ParseOptimParams(ParseKWArgs):
    def parse_value(self, value: str):
        try:
            value = float(value)
        except ValueError:
            value = tuple([float(x) for x in value.split(",")])
        return value


class ParseModelParams(ParseKWArgs):
    def parse_value(self, value: str):
        return value


def set_seed(seed: int):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True


class Parser(object):
    @staticmethod
    def train():
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--name", type=str, default="dev", help="experiment name",
        )

        parser.add_argument("--random-word-embedding", type=int, default=0)
        parser.add_argument("--glove-word-embedding", type=str, default="glove.6B.300d")

        parser.add_argument("--model", type=str, default="DependencyParserV1", choices=list(MODELS.keys()))
        parser.add_argument("--model-params", nargs="*", action=ParseModelParams, default=dict())

        parser.add_argument("--num-epochs", type=int, default=200)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--num-workers", type=int, default=0)

        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            choices=[k for k in torch.optim.__dict__.keys() if not k.startswith("__")],
        )
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--optimizer-params", nargs="*", action=ParseOptimParams, default=dict())

        opt = parser.parse_args()
        set_seed(opt.seed)
        return opt

    @staticmethod
    def predict():
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--name", type=str, default="dev", help="experiment name",
        )

        parser.add_argument("--checkpoint", type=Path, required=True)

        opt = parser.parse_args()
        set_seed(opt.seed)
        return opt

    @staticmethod
    def generate():
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--config", type=str, default="base", choices=("base", "advanced"),
        )
        opt = parser.parse_args()
        set_seed(opt.seed)
        return opt
