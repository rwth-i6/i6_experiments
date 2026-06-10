__all__ = [
    "RunningAverageUpdater",
    "RelativeFrequencyUpdater"
]

import numpy as np

class RunningAverageUpdater:
    def __init__(self, shape):
        self.value = np.zeros(shape)
        self.counts = np.zeros(shape[0] if len(shape) > 0 else ())
    
    def update(self, update, update_counts):
        assert update.shape == self.value.shape
        assert update_counts.shape == self.counts.shape
        total = self.value * self.counts[:,np.newaxis] + update
        counts = self.counts + update_counts
        self.value = np.where(
            total != 0.0,
            total / counts[:,np.newaxis],
            np.zeros_like(total)
        )
        self.counts = counts

    def update_single(self, value: float):
        new_count = self.counts + 1
        self.value = (self.value * self.counts + value) / new_count
        self.counts = new_count
    
    def update_sequence(self, seq):
        self.value = (
            self.value * self.counts
            + np.sum(seq)
        ) / (self.counts + len(seq))
        self.counts += len(seq)


class RelativeFrequencyUpdater:
    def __init__(self, shape):
        self.shape = shape
        self._counts = np.zeros(shape, dtype=float)
        self.total = 0.0

    @property
    def value(self):
        if self.total == 0:
            return np.zeros_like(self._counts)
        return self._counts / self.total

    @property
    def counts(self):
        return self._counts

    def update(self, update: list[int] | np.ndarray, update_counts=None):
        """
        Add raw counts for each entry.

        Parameters
        ----------
        update : np.ndarray
            Count increments, same shape as self._counts.
        update_counts : ignored
            Present only to keep the same interface as your original class.
        """
        self.update_sequence(update)
        # update = np.asarray(update, dtype=float)
        # assert update.shape == self._counts.shape

        # self._counts += update
        # self.total += np.sum(update)

    def update_single(self, value: int):
        """
        Add one occurrence of a single index.
        """
        self._counts[value] += 1.0
        self.total += 1.0

    def update_sequence(self, seq):
        """
        Add a sequence of observed indices.
        """
        seq = np.asarray(seq)
        binc = np.bincount(seq, minlength=self._counts.shape[0])
        self._counts += binc
        self.total += len(seq)