import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime


class ParseKwargs(argparse.Action):
    def _get_value_from_str(self, value: str):
        try:
            value = int(value)
        except ValueError:
            value = tuple([float(x) for x in value.split(",")])
        return value

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = self._get_value_from_str(value)


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
        # parser.add_argument(
        #     "--name", type=str, default=datetime.now().strftime("%m-%d_%H-%M-%S"), help="name",
        # )
        parser.add_argument(
            "--name", type=str, default="dev", help="experiment name",
        )
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
        parser.add_argument("--optimizer-params", nargs="*", action=ParseKwargs, default=dict())

        opt = parser.parse_args()
        set_seed(opt.seed)
        return opt
