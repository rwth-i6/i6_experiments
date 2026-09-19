"""The dataset that carries the reverse observations into the model (SAE §4a step 4).

``ModelCriterion`` calls ``model(**sample["net_input"])``, so ``net_input`` is the ONLY route from
the data to the loss.  This subclass of fairseq's ``ExtractedFeaturesDataset`` adds three keys:

  ``rev_units``      [B, S] int64   the enc50 K = 500 unit stream on the retained 50 Hz clock
  ``rev_unit_lens``  [B]    int64   S_b, asserted EQUAL to the feature length of the same utterance
  ``rev_eta``        [B, 16] float  the frozen PCA-16 speaker vector of the utterance

The units and eta come from ``W2vu2RevUnitsJob``'s directory, one row per line/row of the split's
``.lengths`` / ``.ids``.  Alignment is by ROW, and the parent's ``min_length`` / ``max_length``
filter is re-applied here to the same ``.lengths`` file so that a dataset index means the same
utterance on both sides; the per-utterance length equality is then asserted for every row.  There is
no fallback: a missing or mismatched row is an error.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from unsupervised.data import ExtractedFeaturesDataset

UNITS_SUFFIX = "rev500"
ETA_SUFFIX = "eta.npy"


def _kept_rows(lengths_path, min_length, max_length):
    """The row indices the parent dataset keeps, and their lengths -- its filter, re-applied."""
    rows, lens = [], []
    with open(lengths_path) as fh:
        for i, line in enumerate(fh):
            length = int(line.rstrip())
            if length >= min_length and (max_length is None or length <= max_length):
                rows.append(i)
                lens.append(length)
    return rows, lens


class RevExtractedFeaturesDataset(ExtractedFeaturesDataset):
    def __init__(self, *, rev_units_dir, path, split, min_length=3, max_length=None, **kwargs):
        super().__init__(path=path, split=split, min_length=min_length, max_length=max_length, **kwargs)

        rows, lens = _kept_rows(os.path.join(path, f"{split}.lengths"), min_length, max_length)
        assert len(rows) == len(self.sizes), (len(rows), len(self.sizes))
        assert np.array_equal(np.asarray(lens), np.asarray(self.sizes)), "length filter disagrees"

        units_path = os.path.join(rev_units_dir, f"{split}.{UNITS_SUFFIX}")
        eta_path = os.path.join(rev_units_dir, f"{split}.{ETA_SUFFIX}")
        with open(units_path) as fh:
            all_lines = fh.read().splitlines()
        eta = np.load(eta_path)
        n_rows = max(rows) + 1 if rows else 0
        assert len(all_lines) >= n_rows and eta.shape[0] >= n_rows, (
            f"{units_path}: {len(all_lines)} unit rows / {eta.shape[0]} eta rows for {n_rows} features"
        )

        self.rev_units = []
        for i, size in zip(rows, self.sizes):
            u = np.array(all_lines[i].split(), dtype=np.int64)
            assert len(u) == size, f"{units_path} row {i}: {len(u)} units vs {size} feature frames"
            assert u.min() >= 0 and u.max() < 500, f"{units_path} row {i}: unit id out of [0, 500)"
            self.rev_units.append(u.astype(np.int16))
        self.rev_eta = np.ascontiguousarray(eta[rows], dtype=np.float32)
        assert self.rev_eta.shape == (len(rows), 16), self.rev_eta.shape

    def __getitem__(self, index):
        res = super().__getitem__(index)
        res["rev_units"] = torch.from_numpy(self.rev_units[index].astype(np.int64))
        res["rev_eta"] = torch.from_numpy(self.rev_eta[index])
        return res

    def collater(self, samples):
        res = super().collater(samples)
        if not res:
            return res
        units = [s["rev_units"] for s in samples]
        lens = torch.LongTensor([len(u) for u in units])
        padded = torch.zeros(len(units), int(lens.max()), dtype=torch.long)
        for i, u in enumerate(units):
            padded[i, : len(u)] = u
        res["net_input"]["rev_units"] = padded
        res["net_input"]["rev_unit_lens"] = lens
        res["net_input"]["rev_eta"] = torch.stack([s["rev_eta"] for s in samples])
        return res
