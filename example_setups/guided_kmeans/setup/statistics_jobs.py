"""
Jobs that turn a guided-k-means ``epoch_statistics.json`` into Variables.

The statistics file is written by ``EpochwiseStatisticsLogger`` during the RETURNN
forward pass, so it is a plain output file of the clustering job. Reports want
individual numbers out of it, and want them without opening the file themselves -
hence these small jobs, all of which publish a ``{epoch: ...}`` Variable.
"""

__all__ = [
    "EpochStatisticsJob",
    "epoch_phoneme_frequencies",
    "MixtureDiagnosticsJob",
    "mixture_diagnostics",
    "ExtractEpochStatisticsJob",
    "ExtractPhonemeFrequenciesJob",
    "PhonemePriorDistanceJob",
    "phoneme_prior_distance",
    "load_epoch_statistics",
    "read_count_file",
]

import json
import logging
from typing import Iterable, Optional, Sequence

import numpy as np

# Module level, not inside the method that uses it. Sisyphus resolves recipe
# packages relative to the *current working directory*, and Task.run chdirs into
# the job's work/ directory before calling run() - so an import executed there
# can no longer find recipe/i6_core and dies with ModuleNotFoundError, while the
# identical import at module level has already run at unpickle time, from the
# setup root, and is fine. Every other i6_core import in this package is at
# module level for this reason; keep it that way.
from i6_core.lib.lexicon import Lexicon

from sisyphus import Job, Task, tk


def load_epoch_statistics(path: str) -> dict:
    """
    Read an ``epoch_statistics.json`` and key it by ``int`` epoch.

    :param path: path to the json file, whose top level maps epoch (as a string,
        json has no integer keys) to that epoch's statistics dict
    """
    with open(path, "rt") as f:
        raw = json.load(f)
    return {int(epoch): stats for epoch, stats in raw.items()}


def read_count_file(path: str) -> dict:
    """
    Read a ``<count>\\t<word>`` count file, as written by ``CountCorpusWordFrequenciesJob``.

    :param path: path to the counts file
    """
    counts = {}
    with open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            count, word = line.split("\t", 1)
            counts[word] = int(count)
    return counts


def normalize_counts(counts: dict) -> dict:
    total = sum(counts.values())
    assert total > 0, "cannot normalize an all-zero distribution"
    return {key: value / total for key, value in counts.items()}


def epoch_phoneme_frequencies(
    epoch_stats: dict, phonemes: Optional[Sequence[str]] = None
) -> Optional[dict]:
    """
    One epoch's phoneme distribution as ``{phoneme: mass}``, whichever counter
    set produced it - or None when this epoch's statistics carry no phoneme
    distribution at all.

    Three shapes reach this, and only the first two are named already:

    ``relative_phoneme_frequencies`` / ``absolute_phoneme_counts``
        What ``PhoenemeFrequencyCounter`` writes for a Viterbi run: already a
        dict keyed by phoneme.
    ``soft_cluster_frequencies``
        What :class:`...lib.guided_kmeans.statistics.FBStatisticsCounter` writes
        for a forward-backward run: a plain list indexed by cluster, because the
        counter is built from ``num_clusters`` alone and never sees a lexicon.
        Naming it needs ``phonemes`` - the label inventory in cluster order,
        i.e. ``i6_core.lib.lexicon.Lexicon.phonemes``, which is exactly the
        order ``...chunked.recognizers.PhonemeIdxMap`` assigns and therefore the
        order the gamma columns are in.

    Returning None rather than raising for the unknown case is deliberate: a new
    counter set should cost a caller its phoneme-distribution columns, not fail
    the job that would have reported everything else about the epoch.
    """
    counts = epoch_stats.get("relative_phoneme_frequencies")
    if counts is None:
        counts = epoch_stats.get("absolute_phoneme_counts")
    if counts is not None:
        return dict(counts)

    soft = epoch_stats.get("soft_cluster_frequencies")
    if soft is None:
        return None
    if phonemes is None:
        logging.warning(
            "statistics carry soft_cluster_frequencies (a forward-backward run) but no "
            "phoneme inventory was given to name the clusters with; pass lexicon= to "
            "EpochStatisticsJob to get the phoneme distribution and prior distance"
        )
        return None
    phonemes = list(phonemes)
    if len(phonemes) != len(soft):
        raise ValueError(
            f"soft_cluster_frequencies has {len(soft)} entries but the lexicon has "
            f"{len(phonemes)} phonemes; the cluster axis and the label inventory have "
            f"to be the same thing for this mapping to mean anything"
        )
    return dict(zip(phonemes, soft))


def phoneme_prior_distance(
    epoch_stats: dict,
    reference: dict,
    excluded: Iterable[str] = ("[SILENCE]",),
    renormalize: bool = True,
    order: int = 1,
    epoch=None,
    phonemes: Optional[Sequence[str]] = None,
) -> float:
    """
    $L_p$ distance between one epoch's phoneme distribution and a reference.

    Shared by :class:`PhonemePriorDistanceJob`, which does this for every epoch of a
    merged statistics file, and :class:`EpochStatisticsJob`, which does it for one
    epoch on its own. Keeping one implementation is the point: the two differ in
    what they read, not in what they compute.

    :param epoch_stats: one epoch's statistics dict
    :param reference: reference distribution, already normalized over its support
    :param excluded: phonemes dropped from the hypothesis before comparing
    :param epoch: only used to identify the epoch in warnings
    :param phonemes: label inventory in cluster order, needed only when the
        epoch reports a forward-backward run's unnamed ``soft_cluster_frequencies``
        - see :func:`epoch_phoneme_frequencies`
    """
    excluded = set(excluded)
    support = sorted(reference)

    counts = epoch_phoneme_frequencies(epoch_stats, phonemes)
    assert counts is not None, (
        f"epoch {epoch} carries no phoneme distribution: none of "
        f"relative_phoneme_frequencies, absolute_phoneme_counts or "
        f"soft_cluster_frequencies (the last needs phonemes= to be named)"
    )

    unknown = sorted(set(counts) - set(reference) - excluded)
    if unknown:
        logging.warning("epoch %s: phonemes not in the reference, dropped: %s", epoch, unknown)

    hypothesis = {p: counts.get(p, 0.0) for p in support}
    total = sum(hypothesis.values())
    if total <= 0:
        logging.warning("epoch %s: no mass on the reference support, distance is nan", epoch)
        return float("nan")
    if renormalize:
        hypothesis = {p: value / total for p, value in hypothesis.items()}

    distance = sum(abs(hypothesis[p] - reference[p]) ** order for p in support)
    return distance ** (1.0 / order)


def mixture_diagnostics(mixtures: np.ndarray, weight_floor: float = 1e-8) -> dict:
    """
    How differentiated a set of mixture weights is, from the weights alone.

    ``mixtures`` is ``[L, C]``, ``p(density | label)`` with rows summing to 1.
    Everything here is a function of that array and nothing else - no
    alignment, no transcription, no reference of any kind - which is what makes
    these usable as the convergence signal for a run whose whole point is that
    no such reference is available.

    The quantity to watch is ``label_codeword_mi``: the mutual information
    between label and density under a uniform label prior,

        I(L;C) = (1/L) sum_l sum_c w_lc log(w_lc / p(c)),   p(c) = mean_l w_lc

    in nats, bounded above by ``log(min(L, C))``. It is exactly zero when every
    label weights the codebook identically, and that is not a hypothetical: it
    is the state a shared codebook starts in under uniform weights, and the one
    a run collapses back to when the search stops distinguishing labels
    acoustically and follows the language model instead. A run whose
    ``label_codeword_mi`` does not climb away from its value at initialization
    within a few epochs has not learned anything about the acoustics, whatever
    its total score is doing - and that can be seen without decoding anything.

    Read it alongside the ``average_am_score``/``average_lm_score`` split in the
    epoch statistics, which says the same thing from the search's side: a total
    score improving only through the LM term is the same collapse.

    The label prior is taken uniform rather than from occupancy so that this
    depends on the model alone and can be computed for an initialization that
    no epoch has run yet. That makes it a statement about how distinct the rows
    are, which is the property in question; a run with very unbalanced label
    occupancy will read slightly high.

    :param weight_floor: below this a density counts as unused by a label. Only
        the ``used``/``dead`` counts use it; the information-theoretic
        quantities take the weights as they are.
    """
    mixtures = np.asarray(mixtures, dtype=np.float64)
    if mixtures.ndim != 2:
        raise ValueError(f"expected 2-D mixtures [L, C], got {mixtures.shape}")
    num_labels, num_densities = mixtures.shape

    row_sums = mixtures.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            "mixture rows are not normalized; row sums range over "
            f"[{row_sums.min():.6g}, {row_sums.max():.6g}]"
        )

    marginal = mixtures.mean(axis=0)  # p(c) under a uniform label prior

    # 0 log 0 = 0, and a density with zero marginal has zero weight under every
    # label, so those terms vanish rather than needing a floor.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mixtures > 0, mixtures / marginal[np.newaxis, :], 1.0)
        terms = np.where(mixtures > 0, mixtures * np.log(ratio), 0.0)
        row_entropy = -np.where(
            mixtures > 0, mixtures * np.log(np.where(mixtures > 0, mixtures, 1.0)), 0.0
        ).sum(axis=1)
    mi = float(terms.sum() / num_labels)
    ceiling = float(np.log(min(num_labels, num_densities)))

    used = int((mixtures.max(axis=0) > weight_floor).sum())
    return {
        "label_codeword_mi": mi,
        "label_codeword_mi_normalized": mi / ceiling if ceiling > 0 else 0.0,
        "label_codeword_mi_ceiling": ceiling,
        "mean_label_entropy": float(row_entropy.mean()),
        # exp(H) - the number of densities a label effectively spreads over,
        # which is the readable form of the entropy and comparable across C.
        "mean_label_perplexity": float(np.exp(row_entropy.mean())),
        "mean_max_label_weight": float(mixtures.max(axis=1).mean()),
        "num_labels": int(num_labels),
        "num_densities": int(num_densities),
        "used_densities": used,
        "dead_densities": int(num_densities - used),
        # The share of the codebook actually in play, which is the number to
        # watch when mixture_floor is 0.0 and weights can only ever be lost.
        "used_density_fraction": used / num_densities if num_densities else 0.0,
    }


class MixtureDiagnosticsJob(Job):
    """
    Reference-free convergence diagnostics for one epoch's mixture weights.

    A separate job rather than part of the epoch, for the reason the guided
    scoring is: it keeps the diagnostic out of the expensive job's hash, and
    lets a metric that has not been invented yet be computed over a run that
    already finished without re-running any search.

    :param mixtures: an epoch's ``mixtures.npy``, i.e.
        ``exp_result.out_artifacts["mixtures"][epoch]``
    """

    def __init__(self, mixtures: tk.Path, weight_floor: float = 1e-8):
        self.mixtures = mixtures
        self.weight_floor = weight_floor

        self.out_diagnostics = self.output_path("mixture_diagnostics.json")
        self.out_scalars = self.output_var("scalars")
        self.out_mi = self.output_var("label_codeword_mi")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        stats = mixture_diagnostics(
            np.load(self.mixtures.get_path()), weight_floor=self.weight_floor
        )
        with open(self.out_diagnostics.get_path(), "w") as fp:
            json.dump(stats, fp, indent=4)
        self.out_scalars.set(stats)
        self.out_mi.set(stats["label_codeword_mi"])
        logging.info(
            "I(L;C) = %.4f nats of a possible %.4f (%.1f%%); labels spread over "
            "%.1f densities on average; %d of %d densities in use",
            stats["label_codeword_mi"],
            stats["label_codeword_mi_ceiling"],
            100 * stats["label_codeword_mi_normalized"],
            stats["mean_label_perplexity"],
            stats["used_densities"],
            stats["num_densities"],
        )


class EpochStatisticsJob(Job):
    """
    Scalars, phoneme distribution and prior distance for a *single* epoch.

    The other jobs here read the merged ``epoch_statistics.json``, which
    ``MergeEpochStatisticsJob`` cannot write until every epoch of a run has
    finished. That is fine for a completed run and useless for one being watched:
    a report on a run at epoch 5 of 10 shows blank statistics columns, because the
    single job feeding all of them is still waiting on epochs 6 to 10.

    This reads one epoch's own ``statistics.json`` instead, so a report gets one
    job per epoch and each column fills in as its epoch lands.

    :param statistics: one epoch's ``statistics.json``, i.e.
        ``GuidedClusteringEpochJob.out_statistics``
    :param reference_counts: as in :class:`PhonemePriorDistanceJob`; None leaves
        ``out_distance`` unset
    :param lexicon: the run's lexicon, needed only for a forward-backward run.
        Its counter reports ``soft_cluster_frequencies`` - a list indexed by
        cluster, with no phoneme names in it - so without this the phoneme
        distribution and the prior distance cannot be produced, and both are
        left empty rather than failing the job. Viterbi runs need nothing here.

        Hash-excluded at None so adding it left every existing job's hash alone.
    """

    __sis_hash_exclude__ = {"lexicon": None}

    def __init__(
        self,
        statistics: tk.Path,
        reference_counts: tk.Path | None = None,
        exclude_phonemes: Iterable[str] = ("[SILENCE]",),
        renormalize: bool = True,
        order: int = 1,
        lexicon: tk.Path | None = None,
    ):
        self.statistics = statistics
        self.reference_counts = reference_counts
        self.exclude_phonemes = sorted(set(exclude_phonemes))
        self.renormalize = renormalize
        self.order = order
        self.lexicon = lexicon

        self.out_scalars = self.output_var("scalars")
        self.out_frequencies = self.output_var("frequencies")
        self.out_distance = self.output_var("distance")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.statistics.get_path(), "rt") as f:
            stats = json.load(f)

        self.out_scalars.set(
            {
                key: value
                for key, value in stats.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        phonemes = None
        if self.lexicon is not None:
            lex = Lexicon()
            lex.load(self.lexicon.get_path())
            # `.phonemes`, deliberately not `.lemmata`. The two lexicons this
            # setup builds share an identical phoneme inventory but *differ* in
            # lemma order: create_fb_lexicon() sets phonemes_before_special so
            # phoneme lemmas run 1..N right after silence (which is what makes
            # the gamma columns line up), while the Viterbi lexicon puts the
            # sentence markers there instead - its lemma 1 is [SENTENCE_BEGIN].
            # Reading the inventory therefore gives the right names whichever
            # lexicon a config passes; reading lemmata would give the right
            # answer only for the FB one and silently mis-name every cluster
            # for the other.
            phonemes = list(lex.phonemes)

        frequencies = epoch_phoneme_frequencies(stats, phonemes)
        self.out_frequencies.set(dict(frequencies) if frequencies else {})

        # Every output has to be set or sisyphus treats the job as unfinished,
        # so a missing distance is None rather than an omission. Note the second
        # case: an epoch whose counter set carries no phoneme distribution at
        # all costs this job its distance, not its scalars - which are the
        # columns a forward-backward run is actually read through.
        if self.reference_counts is None or frequencies is None:
            self.out_distance.set(None)
            return
        excluded = set(self.exclude_phonemes)
        raw_reference = read_count_file(self.reference_counts.get_path())
        reference = normalize_counts(
            {p: c for p, c in raw_reference.items() if p not in excluded}
        )
        self.out_distance.set(
            phoneme_prior_distance(
                stats,
                reference,
                excluded=excluded,
                renormalize=self.renormalize,
                order=self.order,
                phonemes=phonemes,
            )
        )


class ExtractEpochStatisticsJob(Job):
    """
    Reduce a guided-k-means ``epoch_statistics.json`` to its scalar entries.

    The file mixes scalars (``average_am_score``, ``relative_loop_frequency``, ...)
    with whole distributions (``relative_phoneme_frequencies``) and debug output
    (``sampled_tracebacks``). Everything that is not a number is dropped here, so
    the result stays small enough to live in a Variable.

    Note on epoch numbering: statistics epoch ``e`` is the recognition pass that ran
    with ``centroids.e.npy`` and produced ``centroids.{e+1}.npy``, so a decoding run
    on epoch ``e``'s centroids lines up with statistics epoch ``e``. A run with
    ``num_epochs`` epochs writes centroids ``0..num_epochs`` but statistics only for
    ``0..num_epochs-1``; the final centroids have no recognition pass of their own.
    """

    def __init__(self, statistics: tk.Path):
        """
        :param statistics: ``epoch_statistics.json`` of a clustering job
        """
        self.statistics = statistics

        self.out_statistics = self.output_var("statistics")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        stats = load_epoch_statistics(self.statistics.get_path())
        scalars = {
            epoch: {
                key: value
                for key, value in epoch_stats.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            for epoch, epoch_stats in stats.items()
        }
        self.out_statistics.set(scalars)


class ExtractPhonemeFrequenciesJob(Job):
    """
    Publish the per-epoch phoneme distribution as a Variable.

    ``ExtractEpochStatisticsJob`` keeps only the scalar entries, so
    ``relative_phoneme_frequencies`` - a whole distribution - is dropped there and
    cannot be reached by a report column. It is small enough to live in a Variable
    on its own (one float per phoneme per epoch), and having it means a column can
    ask for any single phoneme's share; the silence fraction is the one that gets
    asked for, being the quickest read on whether a run has started explaining
    everything as silence.

    :param statistics: ``epoch_statistics.json`` of a clustering job
    """

    def __init__(self, statistics: tk.Path):
        self.statistics = statistics

        self.out_frequencies = self.output_var("frequencies")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        stats = load_epoch_statistics(self.statistics.get_path())
        frequencies = {
            epoch: dict(epoch_stats.get("relative_phoneme_frequencies", {}))
            for epoch, epoch_stats in stats.items()
        }
        self.out_frequencies.set(frequencies)


class PhonemePriorDistanceJob(Job):
    """
    Distance per epoch between the unigram phoneme distribution the guided search
    produced and a reference (transcription) unigram distribution.

    Both sides are restricted to the phonemes the reference knows about and
    renormalized over that shared support, so the comparison is between two proper
    distributions. This matters because the hypothesis side contains ``[SILENCE]``,
    which transcription counts do not - without excluding it, the distance would
    mostly measure how much silence the search emitted.
    """

    def __init__(
        self,
        statistics: tk.Path,
        reference_counts: tk.Path,
        exclude_phonemes: Iterable[str] = ("[SILENCE]",),
        renormalize: bool = True,
        order: int = 1,
    ):
        """
        :param statistics: ``epoch_statistics.json``; each epoch needs either
            ``relative_phoneme_frequencies`` or ``absolute_phoneme_counts``
        :param reference_counts: ``<count>\\t<phoneme>`` file, e.g.
            ``CountCorpusWordFrequenciesJob.out_word_counts`` run on a phoneme corpus
            (``constants.PHONEME_UNIGRAM_PRIORS``)
        :param exclude_phonemes: dropped from both sides before normalizing
        :param renormalize: renormalize the hypothesis over the shared support. With
            ``False`` the mass on excluded/unknown phonemes is simply dropped and the
            hypothesis no longer sums to one.
        :param order: p of the L_p distance; 1 (the default) gives the L1 distance
        """
        self.statistics = statistics
        self.reference_counts = reference_counts
        # sorted list rather than a set, so the job hash does not depend on set order
        self.exclude_phonemes = sorted(set(exclude_phonemes))
        self.renormalize = renormalize
        self.order = order

        self.out_distances = self.output_var("distances")
        self.out_reference_priors = self.output_var("reference_priors")

    def tasks(self):
        yield Task("run", mini_task=True)

    _normalize = staticmethod(normalize_counts)

    def run(self):
        excluded = set(self.exclude_phonemes)

        raw_reference = read_count_file(self.reference_counts.get_path())
        reference = normalize_counts({p: c for p, c in raw_reference.items() if p not in excluded})
        self.out_reference_priors.set(reference)

        self.out_distances.set(
            {
                epoch: phoneme_prior_distance(
                    epoch_stats,
                    reference,
                    excluded=excluded,
                    renormalize=self.renormalize,
                    order=self.order,
                    epoch=epoch,
                )
                for epoch, epoch_stats in load_epoch_statistics(
                    self.statistics.get_path()
                ).items()
            }
        )
