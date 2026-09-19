"""SAE §4a step 4 -- the reverse observations, re-keyed onto the fairseq GAN's own data rows.

``BlankfreeVadHdfJob`` stores the enc50 K = 500 unit stream per utterance tag (RETURNN HDF, one
shard per split); ``SpeakerEtaJob`` stores the frozen PCA-16 eta per tag.  The GAN reads a feature
dump addressed by ROW: ``{split}.npy`` + ``{split}.lengths`` + ``{split}.ids``.  This job writes the
units and eta in the dump's own row order, so the dataset can align them by row and assert equality
per utterance:

  ``{split}.rev500``   one line per row of ``{split}.ids``, whitespace-separated unit ids (the
                       ``.km`` text format the reference dump already uses for its aux target)
  ``{split}.eta.npy``  ``[N, 16]`` float32, row-aligned to ``{split}.ids``

Every check is fatal: an id with no units, an id with no eta, a per-utterance length that differs
from ``{split}.lengths``, a unit outside ``[0, 500)``, or a corpus frame total that differs from the
feature dump's.  There is no fallback and no silent skip -- if the two clocks ever disagree the term
would be scoring a different utterance.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

from sisyphus import Job, Task, tk

N_UNITS = 500  # the enc50 codebook (SAE_4A.md:43); asserted, not assumed
ETA_DIM = 16   # the PCA-16 speaker vector (SAE_4A.md:57)


class W2vu2RevUnitsJob(Job):
    """Row-aligned unit + eta files for the GAN's feature dump (see the module docstring)."""

    def __init__(
        self,
        *,
        units_hdfs: Sequence[tk.Path],
        eta_npz: tk.Path,
        data_dir: tk.Path,
        splits: Sequence[str] = ("train", "valid"),
    ):
        super().__init__()
        self.units_hdfs = list(units_hdfs)
        self.eta_npz = eta_npz
        self.data_dir = data_dir
        self.splits = list(splits)

        self.out_dir = self.output_path("rev_units", directory=True)
        self.out_stats = self.output_path("stats.txt")
        self.rqmt = {"cpu": 2, "mem": 24, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _read_units(self) -> Dict[str, "object"]:
        import h5py
        import numpy as np

        units: Dict[str, np.ndarray] = {}
        for path in self.units_hdfs:
            with h5py.File(path.get_path(), "r") as fh:
                assert int(fh.attrs["numLabels"]) == N_UNITS, (path, dict(fh.attrs))
                lens = fh["seqLengths"][:, 0].astype("int64")
                tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
                flat = fh["inputs"][:].astype("int64")
            assert int(lens.sum()) == flat.shape[0], (path, lens.sum(), flat.shape)
            off = np.concatenate([[0], np.cumsum(lens)])
            for i, tag in enumerate(tags):
                assert tag not in units, f"duplicate tag {tag} in {path}"
                units[tag] = flat[off[i]:off[i + 1]]
        return units

    def run(self):
        import numpy as np

        units = self._read_units()
        eta_npz = np.load(self.eta_npz.get_path(), allow_pickle=False)
        eta_tab = np.asarray(eta_npz["eta"], dtype=np.float32)
        eta_idx = {str(t): i for i, t in enumerate(eta_npz["tags"])}
        assert eta_tab.shape[1] == ETA_DIM and eta_tab.shape[0] == len(eta_idx), eta_tab.shape

        lines = []
        for split in self.splits:
            base = os.path.join(self.data_dir.get_path(), split)
            ids = [x.strip() for x in open(base + ".ids")]
            lengths = [int(x) for x in open(base + ".lengths")]
            assert len(ids) == len(lengths), (len(ids), len(lengths))

            eta = np.zeros((len(ids), ETA_DIM), dtype=np.float32)
            total = 0
            with open(os.path.join(self.out_dir.get_path(), f"{split}.rev500"), "w") as fh:
                for row, (tag, length) in enumerate(zip(ids, lengths)):
                    assert tag in units, f"{split} row {row}: no reverse units for {tag}"
                    assert tag in eta_idx, f"{split} row {row}: no eta for {tag}"
                    u = units[tag]
                    assert len(u) == length, (
                        f"{split} row {row} ({tag}): {len(u)} unit frames vs {length} feature frames"
                    )
                    assert u.min() >= 0 and u.max() < N_UNITS, f"{split} row {row}: unit out of range"
                    eta[row] = eta_tab[eta_idx[tag]]
                    total += len(u)
                    fh.write(" ".join(str(int(x)) for x in u) + "\n")
            np.save(os.path.join(self.out_dir.get_path(), f"{split}.eta.npy"), eta)
            assert total == sum(lengths), (total, sum(lengths))
            lines.append(f"{split}: {len(ids)} utterances, {total} retained frames, eta {eta.shape}")

        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
