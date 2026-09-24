"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/speaker.py.

The frozen speaker vector eta of SAE 4a (SAE_4A.md:57-60).

    eta = PCA-16 of the utterance-mean L15 features, fitted once on train-clean-100, frozen.

Label-free and content-poor by construction (one 1024-d average per utterance -> 16 numbers), it
exists so the reverse model does not have to encode speaker in the symbol stream: 3g found duration,
then speaker, were what its scorer learned instead of content (SAE_3G.md:472-478).

The utterance means are NOT recomputed here.  The L15 feature dump (:mod:`..w2v2.features`, which
replaces the source's ``AvStatesJob``) writes, beside each feature HDF, a pickle
``{seq_tag: float32 [2, D]}`` of per-utterance per-dim mean/std computed from the float32 states
before the fp16 cast; row 0 IS the utterance-mean 1024-d L15 vector.

Two stores, both clean (w2v2-lv60 layer 15, 1024-d @ 50 Hz, unperturbed audio):

    FIT  the four train-clean-100 shards, 28,539 utts -- the spec's fit corpus
         (banked: ``AvStatesJob.Dsynh5MqmgjY``).
    DEV  the dev dump, all 5,567 LibriSpeech dev utts (banked: the seed+dev ``AvStatesJob.c4Ak1rACchRC``).

In the port both stores are the ``perutt_stats.pkl`` outputs of the ogg-zip L15 dumps (FIT: the
four train-clean-100 shards; DEV: the dev-clean and dev-other dumps), keyed by bare utterance ids;
:func:`speaker_of` also accepts an ogg-zip segment name.

On the utterances two dumps share, the per-utterance means were identical in the source (max
|difference| 0.0 over 500 sampled utts x 1024 dims), so a PCA fitted on one store is a legal
projection for the other.  The per-utterance statistics are raw state moments; the global
standardization vector is a separate output and is not applied here.  train-clean-100 carries no dev
utterance, so the fit corpus and the projected dev set are disjoint by construction.

Scaling: each retained component is divided by its standard deviation on the fit corpus, so the
emission head sees O(1) inputs.  That is a per-axis affine rescaling of a frozen embedding, fixed
before any eta is consumed; the rotation, the corpus and the dimension are the experimental content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sisyphus import Job, Task, tk

__all__ = ["ETA_DIM", "SpeakerPca", "fit_speaker_pca", "speaker_of", "SpeakerEtaJob"]

ETA_DIM = 16  # PCA-16 (SAE_4A.md:57)
# train-clean-100 (AvStatesJob.Dsynh5MqmgjY/output/av_states.stats.txt: "train 28539 utts,
# 18088388 frames"; setup report §1 "100 h bed"; SAE_1d.md:73).
_EXPECT_FIT_UTTS = 28_539
_FIT_CORPUS = "train-clean-100, clean"
_EXPECT_DEV_UTTS = 5_567  # all LibriSpeech dev utts (c4Ak1rACchRC/output/av_states.stats.txt)


@dataclass
class SpeakerPca:
    """The frozen projection: centre, rotate onto ``dim`` axes, scale each to unit fit variance."""

    mean: "object"  # [D]
    components: "object"  # [dim, D], rows orthonormal
    scales: "object"  # [dim], per-component std on the fit corpus
    explained_variance_ratio: "object"  # [dim]
    fit_corpus: str = ""
    n_fit_utts: int = 0

    def apply(self, x):
        """``[N, D]`` utterance means -> ``[N, dim]`` eta."""
        import numpy as np

        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2 and x.shape[1] == self.mean.shape[0], (x.shape, self.mean.shape)
        return (((x - self.mean) @ self.components.T) / self.scales).astype(np.float32)

    def save(self, path: str) -> None:
        import numpy as np

        np.savez_compressed(
            path, mean=self.mean, components=self.components, scales=self.scales,
            explained_variance_ratio=self.explained_variance_ratio,
            fit_corpus=np.array([self.fit_corpus]), n_fit_utts=np.array([self.n_fit_utts]),
        )

    @classmethod
    def load(cls, path: str) -> "SpeakerPca":
        import numpy as np

        d = np.load(path, allow_pickle=False)
        return cls(
            mean=d["mean"], components=d["components"], scales=d["scales"],
            explained_variance_ratio=d["explained_variance_ratio"],
            fit_corpus=str(d["fit_corpus"][0]), n_fit_utts=int(d["n_fit_utts"][0]),
        )


def fit_speaker_pca(x, *, dim: int = ETA_DIM, fit_corpus: str = "", eps: float = 1e-8) -> SpeakerPca:
    """PCA of the utterance means ``x [N, D]`` by eigendecomposition of the covariance.

    Deterministic end to end: a symmetric eigensolver (no randomized SVD) plus a fixed sign
    convention (the largest-magnitude loading of each component is positive), so the same store
    always yields the same eta.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] > dim, x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:dim]
    comp = vecs[:, order].T  # [dim, D]
    sign = np.sign(comp[np.arange(dim), np.abs(comp).argmax(axis=1)])
    sign[sign == 0] = 1.0
    comp = comp * sign[:, None]
    proj = xc @ comp.T
    scales = np.maximum(proj.std(axis=0), eps)
    total = max(float(np.trace(cov)), eps)
    return SpeakerPca(
        mean=mean, components=comp, scales=scales,
        explained_variance_ratio=np.maximum(vals[order], 0.0) / total,
        fit_corpus=fit_corpus, n_fit_utts=int(x.shape[0]),
    )


def speaker_of(seq_tag: str) -> str:
    """LibriSpeech seq tags are ``<speaker>-<chapter>-<utt>``; the speaker is the first field.

    An ogg-zip segment name ``<corpus>/<utt>/<utt>`` is mapped to its bare utterance id first.
    """
    tag = str(seq_tag)
    if "/" in tag:
        from .ogg_zip import utt_id_of_segment

        tag = utt_id_of_segment(tag)
    return tag.split("-")[0]


class SpeakerEtaJob(Job):
    """Fit eta once on train-clean-100 and project it onto the dev utterances S1a scores.

    CPU, in-process. Both per-utterance mean/std stores are consumed as frozen paths (the
    ``out_perutt_stats`` pickles of the L15 dumps). ``fit_perutt`` supplies the fit corpus and ``dev_perutt``
    the dev utterances listed in ``dev_tags_json`` (GoldPhonesJob's gold.json). Both counts are
    asserted against ``expect_fit_utts`` / ``expect_dev_utts``, so a store that silently covers
    something else fails here rather than producing an eta of unknown provenance, and the fit corpus
    is asserted to contain no dev utterance.
    """

    def __init__(
        self,
        *,
        fit_perutt: Sequence[tk.Path],
        dev_perutt: Sequence[tk.Path],
        dev_tags_json: tk.Path,
        dim: int = ETA_DIM,
        fit_corpus: str = _FIT_CORPUS,
        expect_fit_utts: int = _EXPECT_FIT_UTTS,
        expect_dev_utts: int = _EXPECT_DEV_UTTS,
    ):
        self.fit_perutt = list(fit_perutt)
        self.dev_perutt = list(dev_perutt)
        self.dev_tags_json = dev_tags_json
        self.dim = int(dim)
        self.fit_corpus = fit_corpus
        self.expect_fit_utts = int(expect_fit_utts)
        self.expect_dev_utts = int(expect_dev_utts)
        self.out_eta = self.output_path("eta.npz")
        self.out_pca = self.output_path("speaker_pca.npz")
        self.out_stats = self.output_path("eta.stats.txt")
        self.out_json = self.output_path("eta.json")
        # CPU, one pass over 5 per-utterance stat pickles (measured: 4 tc100 shards load in 0.6 s)
        # plus one 1024 x 1024 eigendecomposition. Peak arrays: 34,106 x 1024 float64 = 0.27 GB.
        self.rqmt = {"cpu": 2, "mem": 24, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json
        import pickle

        import numpy as np

        from ..analysis.json_io import dump_json

        def _read_means(paths, keep=None):
            """``(tags, [N, D] means)`` over the given per-utterance stat pickles, in tag order.

            One streaming pass: only row 0 (the per-utterance MEAN) of each ``[2, D]`` entry is
            kept, so the 1024-d moments never accumulate beyond N x D float64.
            """
            out: Dict[str, "np.ndarray"] = {}
            for p in paths:
                with open(p.get_path(), "rb") as fh:
                    shard = pickle.load(fh)
                for t, v in shard.items():
                    t = str(t)
                    if keep is not None and t not in keep:
                        continue
                    v = np.asarray(v)
                    assert v.ndim == 2 and v.shape[0] == 2, f"{t}: expected [2, D], got {v.shape}"
                    assert t not in out, f"duplicate seq tag {t!r} across stores"
                    out[t] = v[0].astype(np.float64)
                del shard
            names = sorted(out)
            return names, (np.stack([out[t] for t in names]) if names else np.zeros((0, 0)))

        with open(self.dev_tags_json.get_path()) as fh:
            gold = json.load(fh)
        dev_tags = {t for split in gold for t in gold[split]}

        fit_names, x_fit = _read_means(self.fit_perutt)
        assert len(fit_names) == self.expect_fit_utts, (
            f"fit corpus is {len(fit_names)} utts, expected {self.expect_fit_utts} "
            f"({self.fit_corpus}); the eta store does not cover what the spec names"
        )
        assert not (set(fit_names) & dev_tags), "the fit corpus contains dev utterances"

        dev_names, x_dev = _read_means(self.dev_perutt, keep=dev_tags)
        assert len(dev_names) == self.expect_dev_utts, (
            f"dev source covers {len(dev_names)} of the {len(dev_tags)} gold dev utts, "
            f"expected {self.expect_dev_utts}"
        )
        assert set(dev_names) == dev_tags, "a gold dev utterance is missing from the dev store"
        assert x_fit.shape[1] == x_dev.shape[1], (x_fit.shape, x_dev.shape)

        pca = fit_speaker_pca(x_fit, dim=self.dim, fit_corpus=self.fit_corpus)
        tags = fit_names + dev_names
        eta = np.concatenate([pca.apply(x_fit), pca.apply(x_dev)], axis=0)
        order = np.argsort(np.array(tags))
        tags = [tags[i] for i in order]
        eta = eta[order]
        is_dev = np.array([t in dev_tags for t in tags])
        n_fit = len(fit_names)

        np.savez_compressed(self.out_eta.get_path(), tags=np.array(tags), eta=eta)
        pca.save(self.out_pca.get_path())
        speakers = sorted({speaker_of(t) for t in tags})
        record = {
            "fit_corpus": self.fit_corpus,
            "n_fit_utts": n_fit,
            "n_projected_utts": int(len(tags)),
            "n_dev_utts": int(is_dev.sum()),
            "n_speakers_total": len(speakers),
            "dim": self.dim,
            "feature": "utterance-mean wav2vec2-lv60 L15, 1024-d @ 50 Hz (AvStatesJob tap=encoder)",
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio],
            "explained_variance_total": float(sum(pca.explained_variance_ratio)),
            "fit_sources": [p.get_path() for p in self.fit_perutt],
            "dev_sources": [p.get_path() for p in self.dev_perutt],
        }
        lines = [
            f"speaker eta = PCA-{self.dim} of the utterance-mean L15 features",
            f"fit corpus = {self.fit_corpus} ({n_fit} utts)",
            f"projected  = {len(tags)} utts ({int(is_dev.sum())} dev), "
            f"{len(speakers)} speakers total",
            "explained variance (top %d) = %.4f" % (self.dim, record["explained_variance_total"]),
            "eta std over all projected utts = "
            + " ".join("%.3f" % v for v in eta.std(axis=0)),
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        dump_json(record, self.out_json.get_path(), indent=2)
        print("\n".join(lines), flush=True)
