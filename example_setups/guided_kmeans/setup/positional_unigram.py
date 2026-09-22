"""
Sisyphus wiring for the positional unigram initialization.

The one-line summary of what this buys: under a uniform table the first E-step
uses no acoustic information, so it can be computed from label sequences alone
and the first M-step becomes a single GEMM. That replaces one full RASR
forward-backward epoch - the dominant cost of a run - with a job that reads the
features once and never searches.

The chain, per corpus::

    LengthBandsJob(features)                  -> band_edges          §2.3
    PhonemeSequencesFromCorpusJob(text)       -> labels.pkl          §1
    PositionalUnigramJob(labels, band_edges)  -> gamma.npz           §2-3, §5
    BinnedCodewordHistogramJob(features)      -> histogram.npz       §4
    PositionalUnigramTableJob(gamma, H)       -> table.npy           §4

and ``table.npy`` goes straight into ``vq_flavor(table=...)`` where
:class:`...chunked_clustering.NormalTableJob`'s output used to go. Nothing
downstream changes: ``MaterializeModelJob`` writes the epoch-0 model directory
from it, the decode path reads that directory, and the epoch jobs re-estimate
from it as they would from any other initial table.

Section numbers refer to ``docs/positional_unigram_init/segmented_base.md``; the
arithmetic all lives in ``...lib.guided_kmeans.positional_unigram`` and is
tested against the closed form there.

**Where the labels come from is a modelling decision, not a detail.** They are
the corpus's own transcriptions phonemized through the lexicon - text only, no
alignment, the same information class as the phoneme LM the search already uses.
:class:`PhonemeSequencesFromCorpusJob` writes them in exactly the format
``...vq_baseline.SegmentedFeaturesFromAlignmentJob.out_labels`` uses, so
swapping in the oracle segment labels is a one-line A/B and needs no change
anywhere downstream.
"""

from __future__ import annotations

__all__ = [
    "BinnedCodewordHistogramJob",
    "LengthAgreementJob",
    "LengthBandsJob",
    "PhonemeSequencesFromCorpusJob",
    "PositionalUnigramResult",
    "PositionalUnigramTableJob",
    "PositionalUnigramJob",
    "TableComparisonJob",
    "phoneme_label_sequences",
    "positional_unigram_table",
]

import gzip
import hashlib
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
from xml.etree import ElementTree as ET

import numpy as np

from sisyphus import Job, Task, tk

# Imported here rather than inside the task that uses it, and that is not a
# style preference. Sisyphus resolves recipe modules through RecipeFinder over
# gs.IMPORT_PATHS = ["config", "recipe", "recipe/"] - all *relative* - while
# Task.run executes the task inside execute_in_dir(JOB_WORK_DIR). So a
# **top-level** recipe package imported for the first time inside run() resolves
# "recipe" against the job's work directory and dies with ModuleNotFoundError,
# while the same import at module level succeeds because the module is loaded
# when the job is unpickled, before that chdir.
#
# Relative imports (``from ..lib...``) are unaffected wherever they appear: they
# resolve through the parent package's already-absolute __path__. Only top-level
# ones are a trap, which is what ``chunked_clustering.prepare_worker_sys_path``
# exists to work around for the RASR worker pool.
from i6_core.lib.corpus import Corpus

from ..lib.guided_kmeans import positional_unigram as pu
from ..lib.guided_kmeans.util import ProgressLogger

#: Chunk file for :class:`BinnedCodewordHistogramJob`. ``num_chunks`` is in the
#: name because it is unhashed: re-running with a different value must not
#: silently mix results from two different partitions of the corpus, the same
#: reason ``GuidedClusteringEpochJob`` names its chunk files that way.
_CHUNK_FILE = "histogram.{num_chunks}.{index}.npz"


def _read_phoneme_inventory(lexicon_path: str) -> List[str]:
    """
    The lexicon's ``phoneme-inventory`` in file order, which *is* the label
    index space.

    Same convention ``...score.FrameErrorRateJob`` reads, and the same one
    ``PhoneticLexiconFromPhonemeListJob`` writes: ``[SILENCE]`` first, then the
    phonemes sorted. Reading it rather than reconstructing it is what keeps this
    job's indices identical to the ones the model and the GMM alignment use.
    """
    open_fn = gzip.open if lexicon_path.endswith(".gz") else open
    with open_fn(lexicon_path, "rb") as handle:
        root = ET.parse(handle).getroot()
    return [e.findtext("symbol") for e in root.findall(".//phoneme-inventory/phoneme")]


def _hdf_lengths(
    files: Sequence[str], segments: Optional[str] = None
) -> Dict[str, int]:
    """``{seq_tag: length}`` over one or more RETURNN HDFs, optionally filtered."""
    import h5py

    wanted = None
    if segments is not None:
        with open(segments) as handle:
            wanted = {line.strip() for line in handle if line.strip()}

    lengths: Dict[str, int] = {}
    for filename in files:
        with h5py.File(filename, "r") as hdf:
            seq_lengths = hdf["seqLengths"][:]
            if seq_lengths.ndim == 2:
                seq_lengths = seq_lengths[:, 0]
            for tag, length in zip(hdf["seqTags"][:], seq_lengths):
                tag = tag.decode("utf-8") if isinstance(tag, bytes) else str(tag)
                if wanted is None or tag in wanted:
                    lengths[tag] = int(length)
    if not lengths:
        raise RuntimeError(
            "no sequence survived; check that the segment list matches the HDF tags"
        )
    return lengths


def _as_list(paths: Union[tk.Path, Sequence[tk.Path]]) -> List[tk.Path]:
    return list(paths) if isinstance(paths, (list, tuple)) else [paths]


def _corpus_hash(sequences: Dict[str, np.ndarray]) -> str:
    """
    A stable fingerprint of the label corpus, written into the artifact's meta.

    §1 asks for this for a reason: a gamma built on one corpus and consumed
    against another is wrong in a way that produces no runtime signal at all.
    The job hash already covers the *inputs*, but not what was actually read out
    of them, and this is cheap.
    """
    digest = hashlib.sha1()
    for tag in sorted(sequences):
        digest.update(tag.encode("utf-8"))
        digest.update(np.asarray(sequences[tag], dtype=np.int32).tobytes())
    return digest.hexdigest()


class PhonemeSequencesFromCorpusJob(Job):
    """
    Frame-level label sequences from a *phonemized* bliss corpus - the §1 input,
    built from text alone.

    Takes the output of ``ApplyLexiconToCorpusJob``, whose segment orths are
    space-separated phonemes, and writes ``{seq_tag: int32[T]}`` in the label
    index space of ``lexicon``. That is the same pickle format as
    ``SegmentedFeaturesFromAlignmentJob.out_labels``, deliberately: the oracle
    labels and these are interchangeable inputs to
    :class:`PositionalUnigramJob`, which makes "how much does the text-only
    estimate lose" a single job swap rather than a second pipeline.

    **Two different lexicons meet here.** ``ApplyLexiconToCorpusJob`` used the
    *corpus* lexicon (the g2p-augmented LibriSpeech one) to turn words into
    phonemes; ``lexicon`` here is the *recognition* lexicon, and is used only for
    the inventory order that fixes the indices. They agree on the symbol set -
    both are stress-free ARPAbet - but that is a fact about this setup, not a
    guarantee, so an unknown symbol is an error rather than a silent drop: a
    dropped token would shorten ``T`` and shift every position after it.

    :param exclude_symbols: tokens removed before indexing. ``[SILENCE]`` by
        default, because the features this gamma is consumed against are
        silence-free - the segmentation drops silence segments, so a transcript
        that still carried silence would not describe the same sequence. The
        silence *label* still exists in the inventory and still gets a table row;
        see :class:`PositionalUnigramTableJob` on what it is given.
    """

    __sis_hash_exclude__ = {"rqmt": None}

    def __init__(
        self,
        phoneme_corpus: tk.Path,
        lexicon: tk.Path,
        exclude_symbols: Sequence[str] = ("[SILENCE]",),
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        self.phoneme_corpus = phoneme_corpus
        self.lexicon = lexicon
        self.exclude_symbols = tuple(exclude_symbols)

        self.out_labels = self.output_path("labels.pkl")
        self.out_statistics = self.output_path("statistics.json")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 2}
        if rqmt:
            self.rqmt.update(rqmt)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        phonemes = _read_phoneme_inventory(self.lexicon.get_path())
        index_of = {symbol: index for index, symbol in enumerate(phonemes)}
        excluded = set(self.exclude_symbols)

        corpus = Corpus()
        corpus.load(self.phoneme_corpus.get_path())

        sequences: Dict[str, np.ndarray] = {}
        unknown: Counter = Counter()
        dropped = 0
        empty = 0
        label_counts = np.zeros(len(phonemes), dtype=np.int64)

        for segment in corpus.segments():
            tokens = (segment.orth or "").split()
            indices = []
            for token in tokens:
                if token in excluded:
                    dropped += 1
                    continue
                index = index_of.get(token)
                if index is None:
                    unknown[token] += 1
                    continue
                indices.append(index)
            if not indices:
                empty += 1
                continue
            sequence = np.asarray(indices, dtype=np.int32)
            label_counts += np.bincount(sequence, minlength=len(phonemes))
            sequences[segment.fullname()] = sequence

        if unknown:
            raise ValueError(
                f"{sum(unknown.values())} token(s) over {len(unknown)} distinct symbols "
                f"are not in the recognition lexicon's phoneme inventory, e.g. "
                f"{unknown.most_common(10)}. Dropping them would shorten T and shift "
                f"every position after them, so this is an inventory mismatch to fix "
                f"rather than to smooth over."
            )
        if not sequences:
            raise RuntimeError("no segment produced a label sequence")

        with open(self.out_labels.get_path(), "wb") as handle:
            pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

        lengths = np.array([len(s) for s in sequences.values()])
        statistics = {
            "sequences": len(sequences),
            "tokens": int(lengths.sum()),
            "dropped_excluded_tokens": dropped,
            "empty_segments": empty,
            "excluded_symbols": list(self.exclude_symbols),
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "length_mean": float(lengths.mean()),
            "length_quantiles": {
                str(q): float(np.percentile(lengths, q)) for q in (1, 5, 25, 50, 75, 95, 99)
            },
            "labels_seen": int((label_counts > 0).sum()),
            "num_labels": len(phonemes),
            "label_counts": label_counts.tolist(),
            "corpus_hash": _corpus_hash(sequences),
        }
        with open(self.out_statistics.get_path(), "w") as handle:
            json.dump(statistics, handle, indent=4)
        print(
            f"{len(sequences)} sequences, {lengths.sum()} tokens, "
            f"T {lengths.min()}..{lengths.max()} (mean {lengths.mean():.1f}), "
            f"{int((label_counts > 0).sum())}/{len(phonemes)} labels seen",
            flush=True,
        )


class LengthBandsJob(Job):
    """
    Length bands from the corpus that will *consume* gamma (§2.3).

    Deliberately built on the **features**, not on the text: the bands decide
    which cells the M-step reads, so they have to be well populated in the
    feature corpus. The text side only has to be able to fill them, and
    :class:`PositionalUnigramJob` reports ``R_l`` there separately - that is what
    the backoff of §3.4 weights with.

    Measured here: cv-nosil (2786 sequences, T 11..221) gives 3 bands at
    ``r_min = 200``, ls100-nosil (28,234 sequences, T 7..285) gives 4.

    :param labels: switches the banding variable to the **token count**, for the
        ``L << T`` case (addendum §9.2). The profile width there is
        ``sd_tau ~ 1/(2 sqrt(mL))``, governed by the chain length rather than by
        the frames; §2.3 bands by ``T`` only because in the frame-level case
        ``S`` was proportional to ``T`` and the two coincided. Banding an
        unsegmented corpus by frames would put segments of very different
        resolution in one cell. The features are still what decides *which*
        sequences are counted, so the band population matches the corpus that
        consumes it either way.
    """

    __sis_hash_exclude__ = {"labels": None}

    def __init__(
        self,
        features_hdf: Union[tk.Path, Sequence[tk.Path]],
        segments: Optional[tk.Path] = None,
        r_min: int = 200,
        ratio: float = 2.0,
        max_bands: int = 8,
        labels: Optional[tk.Path] = None,
    ):
        self.features_hdf = _as_list(features_hdf)
        self.segments = segments
        self.r_min = r_min
        self.ratio = ratio
        self.max_bands = max_bands
        self.labels = labels

        self.out_band_edges = self.output_path("band_edges.npy")
        self.out_statistics = self.output_path("statistics.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lengths_by_tag = _hdf_lengths(
            [f.get_path() for f in self.features_hdf],
            self.segments.get_path() if self.segments else None,
        )
        banded_by = "frames"
        if self.labels is not None:
            with open(self.labels.get_path(), "rb") as handle:
                tokens = pickle.load(handle)
            missing = set(lengths_by_tag) - set(tokens)
            if missing:
                raise ValueError(
                    f"{len(missing)} feature sequence(s) have no token sequence, e.g. "
                    f"{sorted(missing)[:3]}; the bands would then be built on a "
                    f"different corpus than the one that fills them"
                )
            lengths_by_tag = {tag: len(tokens[tag]) for tag in lengths_by_tag}
            banded_by = "tokens"
        lengths = np.array(sorted(lengths_by_tag.values()))
        edges = pu.make_bands(
            lengths, r_min=self.r_min, ratio=self.ratio, max_bands=self.max_bands
        )
        np.save(self.out_band_edges.get_path(), edges)

        counts = [
            int(((lengths >= edges[i]) & (lengths < edges[i + 1])).sum())
            for i in range(len(edges) - 1)
        ]
        statistics = {
            "sequences": len(lengths),
            "frames": int(lengths.sum()),
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "length_mean": float(lengths.mean()),
            "band_edges": edges.tolist(),
            "sequences_per_band": counts,
            "frames_per_band": [
                int(lengths[(lengths >= edges[i]) & (lengths < edges[i + 1])].sum())
                for i in range(len(edges) - 1)
            ],
            "r_min": self.r_min,
            "ratio": self.ratio,
            "max_bands": self.max_bands,
            "banded_by": banded_by,
        }
        with open(self.out_statistics.get_path(), "w") as handle:
            json.dump(statistics, handle, indent=4)
        print(
            f"{len(lengths)} sequences, {banded_by} {lengths.min()}..{lengths.max()}: "
            f"{len(counts)} band(s) {edges.tolist()} holding {counts}",
            flush=True,
        )


class LengthAgreementJob(Job):
    """
    Does ``|c_r| == T_r``? The assumption everything rests on, measured.

    §0's cancellation needs one label per frame. In the segmented arm that is
    supposed to hold by construction - one mean-pooled vector per phone segment,
    ``LOOP_PROB = 0.0`` so the search can neither loop nor skip - but the FB
    search also carries ``blank_index=0`` and ``collapse_repeated_labels=True``,
    and the transcript's phoneme count is produced by a different lexicon than
    the alignment that cut the segments.

    So this compares, per sequence tag, the number of phonemes in the transcript
    against the number of feature vectors. A large disagreement does not break
    the *estimate* - gamma conditions on length and both sides are binned the
    same way - but it does mean the text corpus and the feature corpus describe
    different sequences, and the length bands then pair up cells that do not
    correspond.

    Diagnostic only: nothing consumes its output, so it can be read after the
    fact without re-running anything.
    """

    def __init__(
        self,
        labels: tk.Path,
        features_hdf: Union[tk.Path, Sequence[tk.Path]],
        segments: Optional[tk.Path] = None,
    ):
        self.labels = labels
        self.features_hdf = _as_list(features_hdf)
        self.segments = segments
        self.out_report = self.output_path("length_agreement.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.labels.get_path(), "rb") as handle:
            sequences = pickle.load(handle)
        feature_lengths = _hdf_lengths(
            [f.get_path() for f in self.features_hdf],
            self.segments.get_path() if self.segments else None,
        )

        shared = sorted(set(sequences) & set(feature_lengths))
        text = np.array([len(sequences[tag]) for tag in shared])
        feat = np.array([feature_lengths[tag] for tag in shared])
        difference = text - feat

        examples = [
            {"tag": tag, "text": int(len(sequences[tag])), "features": int(feature_lengths[tag])}
            for tag in sorted(
                shared, key=lambda t: -abs(len(sequences[t]) - feature_lengths[t])
            )[:10]
        ]
        report = {
            "tags_in_labels": len(sequences),
            "tags_in_features": len(feature_lengths),
            "tags_shared": len(shared),
            "tags_only_in_labels": len(set(sequences) - set(feature_lengths)),
            "tags_only_in_features": len(set(feature_lengths) - set(sequences)),
            "exact_match_fraction": float((difference == 0).mean()) if len(shared) else 0.0,
            "mean_absolute_difference": float(np.abs(difference).mean()) if len(shared) else 0.0,
            "mean_signed_difference": float(difference.mean()) if len(shared) else 0.0,
            "max_absolute_difference": int(np.abs(difference).max()) if len(shared) else 0,
            "length_correlation": (
                float(np.corrcoef(text, feat)[0, 1]) if len(shared) > 1 else float("nan")
            ),
            "worst_examples": examples,
        }
        with open(self.out_report.get_path(), "w") as handle:
            json.dump(report, handle, indent=4)
        print(
            f"{len(shared)} shared tags: {100 * report['exact_match_fraction']:.1f}% exact, "
            f"mean |delta T| {report['mean_absolute_difference']:.2f}, "
            f"correlation {report['length_correlation']:.4f}",
            flush=True,
        )


class PositionalUnigramJob(Job):
    """
    ``gamma[L, B, C] = p(c_tau = c | length band)`` and the §5 diagnostics.

    No audio, no search, no features - this is the whole first E-step under a
    uniform table, and it can be computed, inspected and thrown away before a
    single feature vector is read. The diagnostics are the point as much as the
    matrix is: they say whether uniform initialization carries any positional
    information at all, and a flat profile means the first M-step will hand every
    label the same row and EM will sit there (§5).

    :param labels: ``{seq_tag: int[T]}`` pickle, from
        :class:`PhonemeSequencesFromCorpusJob` or from
        ``SegmentedFeaturesFromAlignmentJob.out_labels``
    :param band_edges: ``.npy`` from :class:`LengthBandsJob`. ``None`` derives
        the bands from this corpus's own lengths, which is right only when
        nothing else will consume the result.
    :param num_bins: ``B``. 64 by default: bin width has to sit well below the
        profile width in ``tau``, which is of order ``1/sqrt(T)``, and at the
        measured mean ``T ~ 125`` that is 1/64 against 1/11. Oversizing ``B`` is
        cheap because §3 handles the variance; undersizing destroys structure
        irrecoverably.
    :param sigma_bins: kernel width in bins for §3.3
    :param kappa0, kappa1: pseudo-count budgets for the §3.4 backoff, in
        segments' worth of prior mass
    :param features_hdf: switches to **token mode** (addendum §8-§9): ``labels``
        is then a token sequence per segment with ``L << T``, and the frame
        counts come from here. Each token is spread over the frames by the
        alignment kernel ``A[t, i] = p(frame t belongs to token i | length = T)``
        instead of owning exactly one frame. With geometric durations that
        kernel is the same hypergeometric as the frame-level case with
        ``S -> L``, and the self-loop probability cancels under the length
        conditioning - so nothing has to be fitted or measured.
    :param sub_states: §10.1. Splits each token into ``m`` sub-states with tied
        self-loops, making the duration negative binomial instead of geometric -
        unimodal, mode away from 1, which is what a phone duration actually
        looks like. Costs nothing: the kernel is built at chain length ``m*L``
        and the sub-state columns summed back. 3 is the standard phone-model
        value and the recommended default; 1 is the plain geometric model.
    """

    __sis_hash_exclude__ = {
        "rqmt": None,
        "features_hdf": None,
        "segments": None,
        "sub_states": 1,
    }

    def __init__(
        self,
        labels: tk.Path,
        num_labels: int,
        band_edges: Optional[tk.Path] = None,
        num_bins: int = 64,
        sigma_bins: float = 1.0,
        kappa0: float = 10.0,
        kappa1: float = 10.0,
        rqmt: Optional[Dict[str, Any]] = None,
        features_hdf: Optional[Union[tk.Path, Sequence[tk.Path]]] = None,
        segments: Optional[tk.Path] = None,
        sub_states: int = 1,
    ):
        self.labels = labels
        self.num_labels = num_labels
        self.band_edges = band_edges
        self.num_bins = num_bins
        self.sigma_bins = sigma_bins
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        self.features_hdf = _as_list(features_hdf) if features_hdf is not None else None
        self.segments = segments
        self.sub_states = sub_states

        self.out_gamma = self.output_path("gamma.npz")
        self.out_diagnostics = self.output_path("diagnostics.json")
        self.out_plots = self.output_path("plots", directory=True)

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}
        if rqmt:
            self.rqmt.update(rqmt)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with open(self.labels.get_path(), "rb") as handle:
            sequences = pickle.load(handle)

        frame_counts = None
        if self.features_hdf is not None:
            frame_counts = _hdf_lengths(
                [f.get_path() for f in self.features_hdf],
                self.segments.get_path() if self.segments else None,
            )
            # Only sequences the features actually carry: the kernel needs a
            # frame count, and a token sequence without one describes nothing
            # this gamma will ever be consumed against.
            sequences = {t: s for t, s in sequences.items() if t in frame_counts}
            if not sequences:
                raise RuntimeError(
                    "no sequence tag occurs in both the token labels and the features"
                )
        lengths = np.array([len(s) for s in sequences.values()])

        if self.band_edges is not None:
            band_edges = np.load(self.band_edges.get_path())
        else:
            band_edges = pu.make_bands(lengths)

        tags = sorted(sequences)
        progress = ProgressLogger(max(len(sequences), 1), bar_length=40, logging_step=1024)
        progress.start()
        shared_meta = {
            "corpus_hash": _corpus_hash(sequences),
            "source": os.path.basename(os.path.dirname(self.labels.get_path())),
            "sequences": len(sequences),
        }
        if frame_counts is None:
            unigram = pu.build_gamma(
                (sequences[tag] for tag in tags),
                num_labels=self.num_labels,
                band_edges=band_edges,
                num_bins=self.num_bins,
                sigma_bins=self.sigma_bins,
                kappa0=self.kappa0,
                kappa1=self.kappa1,
                meta=shared_meta,
                progress=progress,
            )
        else:
            frames = np.array([frame_counts[tag] for tag in tags])
            resolution = pu.resolution_report(lengths, frames, self.sub_states)
            print(
                f"resolution: {resolution['verdict']}, sd_tau median "
                f"{resolution['sd_tau_median']:.4f} (worst "
                f"{resolution['sd_tau_worst']:.4f}), {resolution['frames_per_token_mean']:.2f} "
                f"frames per token, chain fills "
                f"{100 * resolution['chain_fill_mean']:.0f}% of the frames, "
                f"{resolution['sharpening_over_asymptotic']:.2f}x sharper than the "
                f"asymptotic 1/(2 sqrt(mL))",
                flush=True,
            )
            suggested = pu.suggested_sigma_bins(
                self.num_bins, lengths, frames, self.sub_states
            )
            if suggested != self.sigma_bins:
                print(
                    f"note: sigma_bins is {self.sigma_bins}, §9.5's rule on this corpus "
                    f"suggests {suggested} (the kernel already covers "
                    f"{self.num_bins * resolution['sd_tau_median']:.2f} bins, and "
                    f"smoothing on top of it is smoothing twice)",
                    flush=True,
                )
            if resolution["verdict"] == "flat":
                print(
                    "WARNING: the duration kernel is wider than the profiles it has to "
                    "resolve. Middle tokens are indistinguishable and gamma will be near "
                    "rank 1 outside the boundary tokens (§9.3). Raise sub_states, or "
                    "chunk the utterances so boundary conditioning dominates.",
                    flush=True,
                )
            shared_meta["resolution"] = resolution
            unigram = pu.build_gamma_tokens(
                ((sequences[tag], frame_counts[tag]) for tag in tags),
                num_labels=self.num_labels,
                band_edges=band_edges,
                num_bins=self.num_bins,
                sub_states=self.sub_states,
                sigma_bins=self.sigma_bins,
                kappa0=self.kappa0,
                kappa1=self.kappa1,
                meta=shared_meta,
                progress=progress,
            )
            reduced = unigram.meta.get("sub_states_reduced", 0)
            if reduced:
                print(
                    f"WARNING: {reduced} of {len(tags)} sequences have fewer than "
                    f"{self.sub_states} frames per token, so their sub-state chain did "
                    f"not fit and m was reduced for them.",
                    flush=True,
                )
        unigram.save(self.out_gamma.get_path())

        diagnostics = pu.gamma_diagnostics(unigram)
        with open(self.out_diagnostics.get_path(), "w") as handle:
            json.dump(diagnostics, handle, indent=4)
        self._plot(unigram)

        clamped = unigram.meta["clamped_sequences"]
        if clamped:
            print(
                f"WARNING: {clamped} of {len(sequences)} label sequences fall outside the "
                f"band edges {band_edges.tolist()} and were clamped into the first or last "
                f"band. The bands come from the feature corpus; this many means the two "
                f"corpora do not describe the same sequences.",
                flush=True,
            )
        variable = "L" if self.features_hdf is not None else "T"
        for band in diagnostics["bands"]:
            print(
                f"band {band['band']} {variable} in [{band['length_range'][0]}, "
                f"{band['length_range'][1]}): {band['sequences']} sequences, "
                f"effective rank {band['effective_rank']:.2f} of {self.num_labels}, "
                f"non-stationarity max {band['non_stationarity_max']:.4f} "
                f"(interior {band['non_stationarity_interior_max']:.4f})",
                flush=True,
            )
        if diagnostics["max_effective_rank"] < 1.5:
            print(
                f"WARNING: the highest effective rank over all bands is "
                f"{diagnostics['max_effective_rank']:.3f}. gamma is close to flat in tau, "
                f"so every row of the initial table will be close to the global codeword "
                f"histogram and EM starts at (or near) the fixed point of §5. This is a "
                f"property of the label prior, not a bug here.",
                flush=True,
            )

    def _plot(self, unigram: "pu.PositionalUnigram") -> None:
        """Profile heatmaps, the first row of §5's table. Best effort."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"no plots: {exc}", flush=True)
            return
        for band in range(unigram.gamma.shape[0]):
            figure, axis = plt.subplots(figsize=(8, 4))
            image = axis.imshow(
                unigram.gamma[band].T, aspect="auto", origin="lower", cmap="magma"
            )
            axis.set_xlabel("relative position bin")
            axis.set_ylabel("label")
            axis.set_title(
                f"band {band}: T in [{unigram.band_edges[band]}, "
                f"{unigram.band_edges[band + 1]}), {int(unigram.n_eff[band])} sequences"
            )
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(os.path.join(self.out_plots.get_path(), f"band_{band}.png"), dpi=120)
            plt.close(figure)


class BinnedCodewordHistogramJob(Job):
    """
    ``H[l, beta, k]``: how much mass each codeword carries at each relative
    position (§4). The only place this method reads the audio.

        H[l, beta, k] = sum_{r in l} sum_t w(t, beta; T_r) * delta(q(x_t) = k)

    Quantization is plain L2 against the frozen codebook, exactly as
    ``VectorQuantizedModel.quantize`` and ``SupervisedVQTableJob`` do it, and the
    features are read through ``HDFFeatureSource`` with the same
    ``segments``/``subsampling``/``pooling_function`` the run uses - otherwise
    this describes different frames than the model will see.

    Split into ``num_chunks`` tasks with a merge, like
    ``GuidedClusteringEpochJob``. **``num_chunks`` is unhashed**, for the same
    reason it is there: the accumulation is a plain sum over sequences, so the
    partition cannot change the result.

    :param gamma: the ``gamma.npz`` whose ``band_edges`` and ``num_bins`` this
        histogram has to share. Read rather than re-derived, because a histogram
        binned differently from gamma pairs up cells that mean different things
        and nothing downstream would notice - :func:`...first_m_step` checks it,
        and this is what makes that check pass for the right reason.
    :param token_counts: token sequences, for the ``L << T`` case. The band of a
        sequence is then its **token** count rather than its frame count, which
        is what gamma was banded by (addendum §9.2) - getting this wrong pairs
        cells that mean different things, and :func:`...first_m_step`'s
        consistency check would not catch it, because both sides would still sum
        to the same total mass.
    """

    __sis_hash_exclude__ = {"rqmt": None, "token_counts": None}

    def __init__(
        self,
        features_hdf: Union[tk.Path, Sequence[tk.Path]],
        centroids: tk.Path,
        gamma: tk.Path,
        segments: Optional[tk.Path] = None,
        subsampling: Optional[int] = None,
        pooling_function: str = "maxpool_time_np",
        num_chunks: int = 1,
        rqmt: Optional[Dict[str, Any]] = None,
        token_counts: Optional[tk.Path] = None,
    ):
        self.features_hdf = _as_list(features_hdf)
        self.centroids = centroids
        self.gamma = gamma
        self.segments = segments
        self.subsampling = subsampling
        self.pooling_function = pooling_function
        self.num_chunks = num_chunks
        self.token_counts = token_counts

        self.out_histogram = self.output_path("histogram.npz")
        self.out_statistics = self.output_path("statistics.json")

        # Dominated by reading the features and by one [T, D] x [D, K] product
        # per sequence; memory is one sequence plus the [L, B, K] accumulator,
        # which is 4 x 64 x 512 float64 = 1 MB.
        self.rqmt = {"cpu": 4, "mem": 8, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

    @classmethod
    def hash(cls, kwargs):
        return super().hash(
            {k: v for k, v in kwargs.items() if k not in ("num_chunks", "rqmt")}
        )

    def tasks(self):
        yield Task(
            "accumulate",
            resume="accumulate",
            args=range(self.num_chunks),
            rqmt=self.rqmt,
            parallel=self.num_chunks,
        )
        yield Task("reduce", mini_task=True)

    def _chunk_path(self, index: int) -> str:
        return _CHUNK_FILE.format(num_chunks=self.num_chunks, index=index)

    def accumulate(self, index: int):
        from ..lib.guided_kmeans.chunked.features import HDFFeatureSource

        unigram = pu.PositionalUnigram.load(self.gamma.get_path())
        centroids = np.load(self.centroids.get_path()).astype(np.float64)
        if centroids.ndim != 2:
            raise ValueError(f"expected centroids [K, D], got {centroids.shape}")
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, and ||x||^2 is the same for
        # every codeword, so the argmin only needs the last two terms. One GEMM
        # per sequence, and no [T, K] distance matrix materialized by scipy.
        norms = (centroids ** 2).sum(axis=1)

        source = HDFFeatureSource(
            files=[f.get_path() for f in self.features_hdf],
            segments=self.segments.get_path() if self.segments else None,
            subsampling=self.subsampling,
            pooling_function=self.pooling_function,
            chunk=index,
            num_chunks=self.num_chunks,
        )

        token_counts = None
        if self.token_counts is not None:
            with open(self.token_counts.get_path(), "rb") as handle:
                token_counts = {t: len(s) for t, s in pickle.load(handle).items()}

        progress = ProgressLogger(max(len(source), 1), bar_length=40, logging_step=256)
        progress.start()

        def codeword_sequences():
            for seq_index, (tag, features) in enumerate(source):
                features = np.asarray(features, dtype=np.float64)
                if features.shape[1] != centroids.shape[1]:
                    raise ValueError(
                        f"feature dim {features.shape[1]} does not match the codebook's "
                        f"{centroids.shape[1]}"
                    )
                codewords = (
                    norms[None, :] - 2.0 * (features @ centroids.T)
                ).argmin(axis=1)
                if token_counts is None:
                    band_key = len(codewords)
                else:
                    band_key = token_counts.get(tag)
                    if band_key is None:
                        raise KeyError(
                            f"no token sequence for {tag!r}; the histogram cannot be "
                            f"banded the way gamma was"
                        )
                yield codewords, band_key
                progress.progress(seq_index)

        counts, mass, per_band, clamped = pu.accumulate_binned_keyed(
            codeword_sequences(),
            unigram.band_edges,
            unigram.num_bins,
            centroids.shape[0],
        )
        np.savez(
            self._chunk_path(index),
            counts=counts,
            mass=mass,
            per_band=per_band,
            clamped=np.asarray(clamped),
            frames=np.asarray(source.total_frames),
        )

    def reduce(self):
        unigram = pu.PositionalUnigram.load(self.gamma.get_path())
        counts = mass = per_band = None
        clamped = frames = 0
        for index in range(self.num_chunks):
            with np.load(self._chunk_path(index)) as chunk:
                if counts is None:
                    counts = chunk["counts"].copy()
                    mass = chunk["mass"].copy()
                    per_band = chunk["per_band"].copy()
                else:
                    counts += chunk["counts"]
                    mass += chunk["mass"]
                    per_band += chunk["per_band"]
                clamped += int(chunk["clamped"])
                frames += int(chunk["frames"])

        residual = np.abs(counts.sum(axis=2) - mass).max()
        if residual > 1e-6 * max(mass.max(), 1.0):
            raise AssertionError(
                f"merged histogram and mass disagree by up to {residual:.3e}"
            )
        np.savez(
            self.out_histogram.get_path(),
            histogram=counts.astype(np.float64),
            mass=mass.astype(np.float64),
            band_edges=np.asarray(unigram.band_edges, dtype=np.int32),
            meta=np.asarray(
                json.dumps(
                    {
                        "position_convention": pu.POSITION_CONVENTION,
                        "num_bins": unigram.num_bins,
                        "num_codewords": int(counts.shape[2]),
                        "gamma_corpus_hash": unigram.meta.get("corpus_hash"),
                        "sequences_per_band": per_band.tolist(),
                        "clamped_sequences": clamped,
                        "frames": frames,
                    }
                )
            ),
        )
        used = int((counts.sum(axis=(0, 1)) > 0).sum())
        statistics = {
            "sequences": int(per_band.sum()),
            "frames": frames,
            "sequences_per_band": per_band.tolist(),
            "clamped_sequences": clamped,
            "codewords_used": used,
            "num_codewords": int(counts.shape[2]),
            "num_chunks": self.num_chunks,
        }
        with open(self.out_statistics.get_path(), "w") as handle:
            json.dump(statistics, handle, indent=4)
        print(
            f"{int(per_band.sum())} sequences, {frames} frames, "
            f"{used}/{counts.shape[2]} codewords used, per band {per_band.tolist()}",
            flush=True,
        )


class PositionalUnigramTableJob(Job):
    """
    The first M-step: ``pi1 = (gamma^T H) / (gamma^T N)``, §4, one GEMM.

    Output is an ``[L, C]`` ``table.npy`` - the same artifact
    ``...chunked_clustering.NormalTableJob`` produces, so it substitutes for it
    in ``vq_flavor(table=...)`` with no other change.

    Note which ``N`` this uses: the **histogram's**, i.e. the frame mass of the
    feature corpus, not the label corpus's. The two differ whenever gamma is
    estimated from text, and ``first_m_step`` catches the mix-up because
    ``sum_k H = N`` only holds for the right one.

    :param unseen: what a label gamma never puts mass on gets. Silence is
        exactly that case here - the features are silence-free so the transcripts
        carry no silence, but the FB lexicon still has silence as blank 0 and the
        model needs 40 rows. ``"global"`` gives it the global codeword histogram,
        the maximum-entropy choice consistent with having no evidence. §3.5 rules
        out a *silent* uniform row, not a reported one: the substitution is in
        ``diagnostics.json`` and in the log.
    """

    def __init__(
        self,
        gamma: tk.Path,
        histogram: tk.Path,
        table_floor: float = 1e-2,
        unseen: str = "global",
    ):
        self.gamma = gamma
        self.histogram = histogram
        self.table_floor = table_floor
        self.unseen = unseen

        self.out_table = self.output_path("table.npy")
        self.out_diagnostics = self.output_path("diagnostics.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        unigram = pu.PositionalUnigram.load(self.gamma.get_path())
        with np.load(self.histogram.get_path(), allow_pickle=False) as data:
            histogram = data["histogram"]
            mass = data["mass"]
            band_edges = data["band_edges"]
            meta = json.loads(str(data["meta"]))

        if meta.get("position_convention") != pu.POSITION_CONVENTION:
            raise ValueError(
                f"histogram was binned under {meta.get('position_convention')!r}, "
                f"gamma speaks {pu.POSITION_CONVENTION!r}"
            )
        if not np.array_equal(np.asarray(band_edges), np.asarray(unigram.band_edges)):
            raise ValueError(
                f"band edges disagree: gamma {unigram.band_edges.tolist()}, "
                f"histogram {np.asarray(band_edges).tolist()}"
            )

        table, info = pu.first_m_step(
            unigram.gamma,
            histogram,
            mass,
            table_floor=self.table_floor,
            unseen=self.unseen,
        )
        np.save(self.out_table.get_path(), table)

        diagnostics = {
            "table_shape": list(table.shape),
            "m_step": info,
            "rows": pu.table_diagnostics(table),
            "gamma": pu.gamma_diagnostics(unigram),
            "histogram_meta": meta,
        }
        with open(self.out_diagnostics.get_path(), "w") as handle:
            json.dump(diagnostics, handle, indent=4)

        rows = diagnostics["rows"]
        if info["unseen_labels"]:
            print(
                f"WARNING: labels {info['unseen_labels']} have no mass in gamma and were "
                f"given the {self.unseen} distribution. Expected for silence, which the "
                f"silence-free transcripts cannot contain; anything else is an inventory "
                f"mismatch.",
                flush=True,
            )
        print(
            f"table {table.shape}: max row cosine {rows['max_row_cosine']:.4f}, "
            f"{rows['tied_pairs']} tied pair(s) "
            f"({100 * rows['tied_pair_fraction']:.1f}%), effective rank "
            f"{rows['effective_rank']:.2f}, mean row entropy "
            f"{rows['row_entropy_mean']:.3f} of {np.log(table.shape[1]):.3f}",
            flush=True,
        )
        if rows["tied_pair_fraction"] > 0.5:
            print(
                "WARNING: more than half of all label pairs have cosine above "
                f"{rows['tied_threshold']}. With a frozen codebook the table is the whole "
                "model, so labels that score alike now score alike after every subsequent "
                "E-step - this initialization is at or near the degenerate fixed point.",
                flush=True,
            )


class TableComparisonJob(Job):
    """
    §7.6: does the analytic first M-step reproduce a real forward-backward pass?

    With uniform tables one FB epoch must produce exactly ``pi1``, because the
    emission product cancels and the search's posteriors *are* the positional
    unigram of the label prior. That makes this the strongest end-to-end check
    available, and the one that catches integration errors between this stage and
    the trainer - a mismatch localizes to one of three places: the position
    convention, the label prior gamma was estimated from, or the search.

    **Read it as a diagnostic, not as a pass/fail.** The spec's ``1e-6`` holds
    only if gamma is the *exact* prior marginal. Estimated by counting
    transcripts it is a sampling estimate of a slightly different distribution
    (the empirical label prior rather than the n-gram the search uses), and the
    search is beam-limited on top. So this reports how far apart the two are and
    leaves the judgement to whoever reads it; a correlation near 1 with a small
    row-wise cosine distance is the expected outcome, and a correlation near 0
    means something is wired wrong.
    """

    def __init__(self, reference: tk.Path, hypothesis: tk.Path):
        self.reference = reference
        self.hypothesis = hypothesis
        self.out_report = self.output_path("comparison.json")
        self.out_max_relative_deviation = self.output_var("max_relative_deviation")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        reference = np.load(self.reference.get_path()).astype(np.float64)
        hypothesis = np.load(self.hypothesis.get_path()).astype(np.float64)
        if reference.shape != hypothesis.shape:
            raise ValueError(
                f"tables differ in shape: {reference.shape} against {hypothesis.shape}"
            )

        difference = np.abs(reference - hypothesis)
        relative = difference / np.maximum(np.abs(reference), 1e-12)
        normalized_ref = reference / np.maximum(
            np.linalg.norm(reference, axis=1, keepdims=True), 1e-300
        )
        normalized_hyp = hypothesis / np.maximum(
            np.linalg.norm(hypothesis, axis=1, keepdims=True), 1e-300
        )
        cosine = (normalized_ref * normalized_hyp).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(hypothesis > 0, reference / np.maximum(hypothesis, 1e-300), 1.0)
            kl = np.where(reference > 0, reference * np.log(ratio), 0.0).sum(axis=1)

        report = {
            "shape": list(reference.shape),
            "max_absolute_deviation": float(difference.max()),
            "mean_absolute_deviation": float(difference.mean()),
            "max_relative_deviation": float(relative.max()),
            "correlation": float(
                np.corrcoef(reference.ravel(), hypothesis.ravel())[0, 1]
            ),
            "row_cosine_min": float(cosine.min()),
            "row_cosine_mean": float(cosine.mean()),
            "row_kl_max": float(np.nanmax(kl)),
            "row_kl_mean": float(np.nanmean(kl)),
            "worst_rows": np.argsort(-difference.max(axis=1))[:5].tolist(),
        }
        with open(self.out_report.get_path(), "w") as handle:
            json.dump(report, handle, indent=4)
        self.out_max_relative_deviation.set(report["max_relative_deviation"])
        print(
            f"correlation {report['correlation']:.6f}, row cosine "
            f"{report['row_cosine_min']:.6f}..{report['row_cosine_mean']:.6f}, "
            f"max |delta| {report['max_absolute_deviation']:.3e}, "
            f"mean KL {report['row_kl_mean']:.4f} nats",
            flush=True,
        )


@dataclass(frozen=True)
class PositionalUnigramResult:
    """Everything :func:`positional_unigram_table` built, for registering."""

    table: tk.Path
    gamma: tk.Path
    histogram: tk.Path
    band_edges: tk.Path
    labels: tk.Path
    gamma_diagnostics: tk.Path
    table_diagnostics: tk.Path
    length_agreement: tk.Path
    plots: tk.Path


def phoneme_label_sequences(
    corpus_key: str,
    lexicon: tk.Path,
    segments: Optional[tk.Path] = None,
    exclude_symbols: Sequence[str] = ("[SILENCE]",),
) -> PhonemeSequencesFromCorpusJob:
    """
    Text-only label sequences for one corpus: bliss -> filter -> phonemize ->
    index.

    ``segments`` restricts the corpus *before* phonemization, which is what keeps
    the mini-task cheap on the 960h corpus (281,241 segments in, 2,786 out for
    cv) and keeps the length distribution equal to the feature corpus's.
    """
    # Safe to defer: this runs at config time in the sis process, where the
    # working directory is the experiment root - unlike a task body, see the
    # note on RecipeFinder at the top of this module.
    from .corpus_setup import setup_corpus
    from i6_core.corpus.filter import FilterCorpusBySegmentsJob
    from i6_core.corpus.transform import ApplyLexiconToCorpusJob

    setup = setup_corpus(corpus_key)
    corpus = setup.corpus
    if segments is not None:
        corpus = FilterCorpusBySegmentsJob(
            corpus, segments, compressed=True, delete_empty_recordings=True
        ).out_corpus
    phonemized = ApplyLexiconToCorpusJob(corpus, setup.lexicon).out_corpus
    return PhonemeSequencesFromCorpusJob(
        phonemized, lexicon, exclude_symbols=exclude_symbols
    )


def positional_unigram_table(
    *,
    features_hdf: Union[tk.Path, Sequence[tk.Path]],
    centroids: tk.Path,
    lexicon: tk.Path,
    num_labels: int,
    corpus_key: str,
    segments: Optional[tk.Path] = None,
    labels: Optional[tk.Path] = None,
    num_bins: int = 64,
    r_min: int = 200,
    max_bands: int = 8,
    sigma_bins: float = 1.0,
    kappa0: float = 10.0,
    kappa1: float = 10.0,
    table_floor: float = 1e-2,
    unseen: str = "global",
    subsampling: Optional[int] = None,
    pooling_function: str = "maxpool_time_np",
    num_chunks: int = 1,
    histogram_rqmt: Optional[Dict[str, Any]] = None,
    alias_prefix: Optional[str] = None,
    token_mode: bool = False,
    sub_states: int = 3,
) -> PositionalUnigramResult:
    """
    The whole chain, as one call, returning the paths worth registering.

    ``result.table`` is what ``vq_flavor(table=...)`` wants; everything else is
    there to be looked at.

    :param labels: skip the text pipeline and use this ``{tag: int[T]}`` pickle -
        e.g. ``SegmentedFeaturesFromAlignmentJob.out_labels`` for the oracle A/B.
    :param corpus_key: bliss corpus the transcriptions come from, ignored when
        ``labels`` is given. ``"train-other-960"`` covers the cv set's tags,
        ``"train-clean-100"`` the ls-100h ones.
    :param num_chunks: tasks the histogram job is split into. Unhashed.
    :param token_mode: the ``L << T`` case of the durations addendum -
        **unsegmented** features, where a label spans several frames instead of
        owning one. Everything moves onto the token axis: the bands are over
        token counts, gamma is accumulated by scattering each token's alignment
        kernel over the frames rather than counting frames, and the codeword
        histogram is banded by token count too. The features themselves are
        still read and binned exactly as before.
    :param sub_states: §10.1's ``m``, ignored unless ``token_mode``. 3 is the
        standard phone-model value; 1 is the plain geometric duration model.
    """
    # Labels first: in token mode the bands are built on *their* lengths, not on
    # the features'.
    if labels is None:
        label_job = phoneme_label_sequences(corpus_key, lexicon, segments=segments)
        labels = label_job.out_labels
        if alias_prefix:
            label_job.add_alias(f"{alias_prefix}/labels")

    bands = LengthBandsJob(
        features_hdf,
        segments=segments,
        r_min=r_min,
        max_bands=max_bands,
        labels=labels if token_mode else None,
    )

    agreement = LengthAgreementJob(labels, features_hdf, segments=segments)
    unigram = PositionalUnigramJob(
        labels=labels,
        num_labels=num_labels,
        band_edges=bands.out_band_edges,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        kappa0=kappa0,
        kappa1=kappa1,
        features_hdf=features_hdf if token_mode else None,
        segments=segments if token_mode else None,
        sub_states=sub_states if token_mode else 1,
    )
    histogram = BinnedCodewordHistogramJob(
        features_hdf=features_hdf,
        centroids=centroids,
        gamma=unigram.out_gamma,
        segments=segments,
        subsampling=subsampling,
        pooling_function=pooling_function,
        num_chunks=num_chunks,
        rqmt=histogram_rqmt,
        token_counts=labels if token_mode else None,
    )
    table = PositionalUnigramTableJob(
        gamma=unigram.out_gamma,
        histogram=histogram.out_histogram,
        table_floor=table_floor,
        unseen=unseen,
    )

    if alias_prefix:
        bands.add_alias(f"{alias_prefix}/bands")
        unigram.add_alias(f"{alias_prefix}/gamma")
        histogram.add_alias(f"{alias_prefix}/histogram")
        table.add_alias(f"{alias_prefix}/table")

    return PositionalUnigramResult(
        table=table.out_table,
        gamma=unigram.out_gamma,
        histogram=histogram.out_histogram,
        band_edges=bands.out_band_edges,
        labels=labels,
        gamma_diagnostics=unigram.out_diagnostics,
        table_diagnostics=table.out_diagnostics,
        length_agreement=agreement.out_report,
        plots=unigram.out_plots,
    )
