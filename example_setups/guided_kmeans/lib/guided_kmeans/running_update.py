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

    def merge(self, other: "RunningAverageUpdater") -> "RunningAverageUpdater":
        """
        Combine another updater's state into this one, in place.

        Count-weighted, hence associative: merging per-chunk updaters in any
        order/grouping gives the same result as feeding every observation to a
        single updater (up to float summation order). This is what lets the
        clustering epoch be split over independent chunk tasks.

        Entries with zero total count keep value 0, matching what __init__
        leaves behind for a cluster nothing was ever assigned to.
        """
        assert self.value.shape == other.value.shape, \
            f"shape mismatch: {self.value.shape} vs {other.value.shape}"
        assert self.counts.shape == other.counts.shape, \
            f"count shape mismatch: {self.counts.shape} vs {other.counts.shape}"

        total_counts = self.counts + other.counts
        if self.value.ndim > self.counts.ndim:
            # per-row counts broadcast over the feature axis, as in update()
            w_self = np.expand_dims(self.counts, axis=-1)
            w_other = np.expand_dims(other.counts, axis=-1)
            denom = np.expand_dims(total_counts, axis=-1)
        else:
            w_self, w_other, denom = self.counts, other.counts, total_counts

        weighted_sum = self.value * w_self + other.value * w_other
        self.value = np.divide(
            weighted_sum,
            denom,
            out=np.zeros_like(weighted_sum),
            where=denom > 0,
        )
        self.counts = total_counts
        return self


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

    def merge(self, other: "RelativeFrequencyUpdater") -> "RelativeFrequencyUpdater":
        """
        Combine another updater's raw counts into this one, in place.
        Trivially associative since the state is a plain count vector.
        """
        assert self._counts.shape == other._counts.shape, \
            f"shape mismatch: {self._counts.shape} vs {other._counts.shape}"
        self._counts += other._counts
        self.total += other.total
        return self