import torch.nn as nn


ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
