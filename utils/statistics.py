import matplotlib.pyplot as plt
import numpy as np


class Statistics(object):

    def __init__(self):

        super(Statistics, self).__init__()

        self._start_ptr = 0
        self._stop_ptr = 0
        self._train_batch_loss = []
        self._train_epochs_loss = []

        self._test_epochs_loss = []
        self._test_epochs_acc = []

    def append(self, attr: str, value: float):
        x = getattr(self, "_" + attr)
        x.append(value)
        if attr == "train_batch_loss":
            self._stop_ptr += 1

    def step(self):
        batch_losses = self._train_batch_loss[self._start_ptr:self._stop_ptr]
        self._train_epochs_loss.append(np.asarray(batch_losses).mean())
        self._start_ptr = self._stop_ptr

    def plot(self, ax: plt.Axes):
        raise NotImplementedError


if __name__ == "__main__":
    s = Statistics()
    s.append("train_batch_loss", 0.5)
    s.append("train_batch_loss", 1.0)
    s.append("train_batch_loss", 1.5)
    s.step()
    s.append("train_batch_loss", 0.1)
    s.append("train_batch_loss", 0.2)
    s.append("train_batch_loss", 0.3)
    s.step()
    print(s)
