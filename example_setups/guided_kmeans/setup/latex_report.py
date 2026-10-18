r"""
LaTeX table reports for guided-k-means experiments.

``report.create_report`` renders the plain-text ``i6_core`` summary table; this module
renders the same kind of data as a ``tabular`` that can be pasted straight into a
document. On top of the decoding scores it can pull per-epoch numbers out of a
clustering job's ``epoch_statistics.json``, including the L1 distance between the
unigram phoneme distribution the guided search produced and the transcription priors.

Minimal use in a config::

    from ...setup.latex_report import LatexTableReport, clustering_statistics

    report = LatexTableReport(
        columns=["lm_scale", "loop_probability", "epoch", "per", "del", "ins", "sub",
                 "l1", "am_score", "lm_score"],
        sort_by=["lm_scale", "loop_probability"],
        caption="Guided k-means, cheating initialization",
    )

    for lm_scale, loop_probability in parameters:
        exp_result = clustering(...)
        statistics = clustering_statistics(exp_result.out_statistics)

        for epoch in range(num_epochs + 1):
            res = decode_and_score(...)
            report.add_row(
                result=res,
                params={"lm_scale": lm_scale, "loop_probability": loop_probability},
                epoch=epoch,
                statistics=statistics,
            )

    report.register("guided_kmeans/cov/recognition/results.tex")

Rows that share the same ``params`` form a group and are merged with ``\multirow`` in
every column that is constant across the group, which is what produces the
"one block per hyperparameter setting, one line per epoch" layout.

``columns`` accepts, in any mix:

* a name from ``COLUMN_LIBRARY`` (``"epoch"``, ``"per"``, ``"del"``, ``"ins"``,
  ``"sub"``, ``"guided_per"``, ``"l1"``, ``"am_score"``, ...),
* any other string, which becomes a column reading that key out of ``params`` -
  note the silent fallback: a misspelled library name does not raise, it becomes
  an empty params column, so a column that renders blank everywhere is usually a
  typo rather than a job that has not finished,
* a ``Column`` built by hand or by ``param()`` / ``statistic()`` / ``prior_l1()``
  when the defaults do not fit.

Leaving ``columns=None`` derives them: the hyperparameters that actually vary, then
epoch, the decoding scores, and the statistics columns if a statistics source was
given.

The statistics columns come in two mutually exclusive groups, because the two
searches record different things: a Viterbi run has ``am_score`` /
``lm_score`` / ``transition_score`` from the traceback counters, and a
forward-backward run has ``log_likelihood`` / ``posterior_entropy`` /
``dead_clusters`` from :class:`...statistics.FBStatisticsCounter`. Asking for
the wrong group is the blank-column case above.

Rendering the table needs ``\usepackage{multirow}`` in the document, plus
``\usepackage[table]{xcolor}`` when ``highlight`` is used.
"""

__all__ = [
    "Row",
    "Column",
    "LatexTableReport",
    "StatisticsSource",
    "clustering_statistics",
    "clustering_statistics_per_epoch",
    "param",
    "statistic",
    "prior_l1",
    "scientific_above",
    "phoneme_fraction",
    "score",
    "COLUMN_LIBRARY",
    "PARAM_HEADERS",
    "create_latex_report",
    "latex_escape",
    "TableReport",
    "TextTableReport",
    "delatex",
]

import itertools
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase
from sisyphus.tools import try_get

from .constants import PHONEME_UNIGRAM_PRIORS
from .decode_config import DecodeRecogResult
from .statistics_jobs import (
    EpochStatisticsJob,
    ExtractEpochStatisticsJob,
    ExtractPhonemeFrequenciesJob,
    PhonemePriorDistanceJob,
)


class _Missing:
    """Marks a cell whose value cannot be read yet (or at all)."""

    def __repr__(self):
        return "<missing>"


MISSING = _Missing()

#: Paths of Variables already registered as a report dependency, see
#: ``LatexTableReport.register``. Keyed by path, so it is safe across configs.
_REGISTERED_DEPENDENCIES: set = set()

_LATEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


def latex_escape(text: str) -> str:
    """Escape the characters that LaTeX treats specially. Backslash goes first."""
    for char, replacement in _LATEX_ESCAPES:
        text = text.replace(char, replacement)
    return text


def _hashable(value: Any) -> Any:
    """Key for grouping/comparing parameter values, tolerant of unhashable ones."""
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class _EpochLookup(DelayedBase):
    """
    Lazily index a ``{epoch: value}`` or ``{epoch: {key: value}}`` Variable.

    ``epoch`` may be ``"last"``, which is resolved against the epochs the Variable
    actually contains rather than against anything known at graph-construction time.
    """

    def __init__(self, variable, epoch, key=None):
        super().__init__(variable, None)
        self.epoch = epoch
        self.key = key

    def _resolve(self):
        """
        The epoch's value, from either shape.

        A plain dict means one Variable per epoch, which is what the per-epoch
        extraction produces; a Variable means one holding every epoch, which is what
        the merged statistics file produces. Both index the same way from here on.
        """
        source = self.a
        if isinstance(source, dict):
            epoch = max(source) if self.epoch == "last" else self.epoch
            return try_get(source[epoch])
        values = try_get(source)
        epoch = max(values) if self.epoch == "last" else self.epoch
        return values[epoch]

    def is_set(self):
        if not isinstance(self.a, dict):
            return super().is_set()
        # Only this epoch's Variable has to exist; the others are separate jobs and
        # may well still be running, which is the whole point of the per-epoch form.
        epochs = self.a
        if self.epoch == "last":
            return bool(epochs) and all(self._is_set_helper(v) for v in epochs.values())
        variable = epochs.get(self.epoch)
        return variable is not None and self._is_set_helper(variable)

    def get(self):
        value = self._resolve()
        return value if self.key is None else value[self.key]


@dataclass(eq=False)
class StatisticsSource:
    """
    The Variables derived from one clustering job's ``epoch_statistics.json``.

    :param scalars: ``{epoch: {key: value}}``, from ``ExtractEpochStatisticsJob``
    :param distances: ``{epoch: distance}``, from ``PhonemePriorDistanceJob``, or None
    :param frequencies: ``{epoch: {phoneme: fraction}}``, from
        ``ExtractPhonemeFrequenciesJob``, or None
    :param name: used when registering the Variables as outputs; filled in from the
        first row's descriptor when left unset
    :param epoch_offset: added to a row's epoch to get the statistics key, see
        ``clustering_statistics``
    """

    scalars: tk.Variable
    distances: tk.Variable | None = None
    frequencies: tk.Variable | None = None
    name: str | None = None
    epoch_offset: int = 0


def clustering_statistics(
    statistics: tk.Path,
    reference_counts: tk.Path | None = PHONEME_UNIGRAM_PRIORS,
    name: str | None = None,
    epoch_offset: int = 0,
    phoneme_frequencies: bool = True,
    **distance_kwargs,
) -> StatisticsSource:
    """
    Build the extraction jobs for one clustering job's statistics file.

    :param statistics: ``ClusteringExpResult.out_statistics``
    :param reference_counts: phoneme counts the prior distance is measured against;
        None skips the distance job (and the ``"l1"`` column stays empty)
    :param name: name for the registered dependency outputs, see ``StatisticsSource``
    :param epoch_offset: added to a row's epoch to get the key to look up, so that a
        decode of ``centroids[e]`` is lined up with the recognition pass that ran with
        those centroids. The two pipelines number this differently: ``clustering()``
        keys statistics ``0..num_epochs-1`` with key ``e`` being the pass that used
        ``centroids[e]``, so the default 0 is right; ``chunked_clustering()`` keys them
        ``1..num_epochs`` with key ``e`` being the pass that used ``centroids[e-1]``,
        so those configs need ``epoch_offset=1``. Columns given an explicit ``epoch=``
        index the file directly and ignore this.
    :param phoneme_frequencies: also extract the per-epoch phoneme distribution, so
        the ``"silence"`` column (and any other ``phoneme_fraction``) has something to
        read. One extra mini-task; set False to skip it.
    :param distance_kwargs: forwarded to ``PhonemePriorDistanceJob``
        (``exclude_phonemes``, ``renormalize``, ``order``)
    """
    scalars = ExtractEpochStatisticsJob(statistics).out_statistics
    distances = None
    if reference_counts is not None:
        distances = PhonemePriorDistanceJob(statistics, reference_counts, **distance_kwargs).out_distances
    frequencies = (
        ExtractPhonemeFrequenciesJob(statistics).out_frequencies if phoneme_frequencies else None
    )
    return StatisticsSource(
        scalars=scalars,
        distances=distances,
        frequencies=frequencies,
        name=name,
        epoch_offset=epoch_offset,
    )


def clustering_statistics_per_epoch(
    epoch_statistics: dict,
    reference_counts: tk.Path | None = PHONEME_UNIGRAM_PRIORS,
    name: str | None = None,
    epoch_offset: int = 0,
    **distance_kwargs,
) -> StatisticsSource:
    """
    Like :func:`clustering_statistics`, but one extraction job per epoch.

    :func:`clustering_statistics` reads the merged ``epoch_statistics.json``, which
    exists only once every epoch of a run has finished - so a report on a run in
    progress shows blank statistics columns however many epochs have completed.
    This takes the per-epoch files instead
    (``ChunkedClusteringExpResult.out_epoch_statistics``) and builds an
    :class:`EpochStatisticsJob` for each, so a column fills in as soon as its own
    epoch is done.

    The columns are the same and so is ``epoch_offset``; only the plumbing differs.
    Prefer this for any run that will be looked at before it finishes.

    :param epoch_statistics: ``{epoch: statistics.json}`` for the run
    """
    jobs = {
        int(epoch): EpochStatisticsJob(path, reference_counts, **distance_kwargs)
        for epoch, path in epoch_statistics.items()
    }
    return StatisticsSource(
        scalars={epoch: job.out_scalars for epoch, job in jobs.items()},
        distances=(
            {epoch: job.out_distance for epoch, job in jobs.items()}
            if reference_counts is not None
            else None
        ),
        frequencies={epoch: job.out_frequencies for epoch, job in jobs.items()},
        name=name,
        epoch_offset=epoch_offset,
    )


@dataclass
class Row:
    """One line of the table before grouping."""

    params: dict = field(default_factory=dict)
    epoch: int | None = None
    result: DecodeRecogResult | None = None
    statistics: StatisticsSource | None = None
    values: dict = field(default_factory=dict)

    @property
    def descriptor(self) -> str | None:
        return self.result.descriptor if self.result is not None else None


@dataclass
class Column:
    """
    One column of the table.

    :param key: identifier, also what ``highlight`` is called with
    :param header: header text, or several lines to stack via a nested ``tabular``
    :param value: reads the cell value off a ``Row``; may return a plain value, a
        Variable, a delayed expression, or None
    :param fmt: ``str.format`` spec applied to the resolved value, or a callable
        taking the value and returning the cell text (see :func:`scientific_above`)
    :param scale: multiplied onto numeric values before formatting, e.g. 100 to turn
        the fractions ``JiwerScoringJob`` reports into percent
    :param align: column alignment letter
    :param span: merge equal values across a row group with ``\\multirow``. Ignored
        for a group whose values are not all equal.
    :param block: visual block; a change of block draws a ``||`` rule
    :param rule_before: overrides the rule this column is preceded by ("|" or "||")
    :param header_align: alignment of the header cell, "c" unless set
    :param raw: do not LaTeX-escape string values of this column
    :param missing: text for a cell that has no value (yet)
    """

    key: str
    header: str | Sequence[str]
    value: Callable[[Row], Any]
    fmt: str | Callable[[Any], str] = "{}"
    scale: float | None = None
    align: str = "r"
    span: bool = False
    block: str = "param"
    rule_before: str | None = None
    header_align: str | None = None
    raw: bool = False
    missing: str | None = None


# -- column sources ---------------------------------------------------------------


def _from_param(name: str) -> Callable[[Row], Any]:
    def source(row: Row):
        return row.params.get(name)

    return source


def _from_result(attribute: str) -> Callable[[Row], Any]:
    def source(row: Row):
        return getattr(row.result, attribute) if row.result is not None else None

    return source


def _from_value(name: str) -> Callable[[Row], Any]:
    def source(row: Row):
        return row.values.get(name)

    return source


def _from_epoch(row: Row):
    return row.epoch


def _statistics_epoch(row: Row, epoch: int | str):
    """Translate a column's ``epoch=`` setting into a key of the statistics file."""
    if epoch != "row":
        return epoch
    if row.epoch is None:
        return None
    return row.epoch + row.statistics.epoch_offset


def _from_statistic(stat_key: str, epoch: int | str = "row") -> Callable[[Row], Any]:
    def source(row: Row):
        if row.statistics is None:
            return None
        stat_epoch = _statistics_epoch(row, epoch)
        if stat_epoch is None:
            return None
        return _EpochLookup(row.statistics.scalars, stat_epoch, stat_key)

    return source


def _from_distance(epoch: int | str = "row") -> Callable[[Row], Any]:
    def source(row: Row):
        if row.statistics is None or row.statistics.distances is None:
            return None
        stat_epoch = _statistics_epoch(row, epoch)
        if stat_epoch is None:
            return None
        return _EpochLookup(row.statistics.distances, stat_epoch)

    return source


def _from_frequency(phoneme: str, epoch: int | str = "row") -> Callable[[Row], Any]:
    def source(row: Row):
        if row.statistics is None or row.statistics.frequencies is None:
            return None
        stat_epoch = _statistics_epoch(row, epoch)
        if stat_epoch is None:
            return None
        return _EpochLookup(row.statistics.frequencies, stat_epoch, phoneme)

    return source


class _EpochDelta(DelayedBase):
    """A statistic at one epoch minus the same statistic at a baseline epoch.

    Exists because some of these statistics are a small signal riding on a large
    additive constant. ``mean_log_likelihood_per_frame`` is the clearest case: it
    carries the Gaussian normalisation, which is hundreds of nats and identical in
    every epoch, so a run whose objective improves by 0.3 nats shows six identical
    leading digits and reads as "not converging" when it is in fact moving. The
    difference drops the constant and leaves the part that changed.
    """

    def __init__(self, variable, epoch, baseline, key=None):
        super().__init__(variable, None)
        self.epoch = epoch
        self.baseline = baseline
        self.key = key

    def _lookups(self):
        return (
            _EpochLookup(self.a, self.epoch, self.key),
            _EpochLookup(self.a, self.baseline, self.key),
        )

    def is_set(self):
        return all(lookup.is_set() for lookup in self._lookups())

    def get(self):
        current, base = self._lookups()
        return current.get() - base.get()


def _from_statistic_delta(
    stat_key: str, baseline: int | str = "first", epoch: int | str = "row"
) -> Callable[[Row], Any]:
    def source(row: Row):
        if row.statistics is None:
            return None
        stat_epoch = _statistics_epoch(row, epoch)
        if stat_epoch is None:
            return None
        base = baseline
        if base == "first":
            scalars = row.statistics.scalars
            # One Variable per epoch is the per-epoch extraction; a single Variable
            # holds them all. Only the first form lets the baseline be found without
            # resolving anything, which is what keeps this usable mid-run.
            if isinstance(scalars, dict):
                if not scalars:
                    return None
                base = min(scalars)
            else:
                base = 1
        return _EpochDelta(row.statistics.scalars, stat_epoch, base, stat_key)

    return source


def _from_frequencies(epoch: int | str = "row") -> Callable[[Row], Any]:
    """The whole ``{label: fraction}`` distribution, for a column that reduces it."""

    def source(row: Row):
        if row.statistics is None or row.statistics.frequencies is None:
            return None
        stat_epoch = _statistics_epoch(row, epoch)
        if stat_epoch is None:
            return None
        return _EpochLookup(row.statistics.frequencies, stat_epoch)

    return source


def _distribution_perplexity(frequencies: dict) -> float:
    """``exp(H)`` of a label distribution: how many labels it effectively uses."""
    total = sum(v for v in frequencies.values() if v and v > 0)
    if not total:
        return float("nan")
    entropy = 0.0
    for value in frequencies.values():
        if value and value > 0:
            share = value / total
            entropy -= share * math.log(share)
    return math.exp(entropy)


# -- column factories -------------------------------------------------------------

#: Headers for the hyperparameters the guided-k-means configs sweep over. Anything
#: not listed here falls back to the key with underscores turned into spaces.
PARAM_HEADERS = {
    "lm_scale": ("LM/Trans.", "scale"),
    "lm_order": ("LM", "order"),
    "transition_scale": ("Trans.", "scale"),
    "emission_scale": ("Emission", "scale"),
    "distance_scale": ("Distance", "scale"),
    "loop_probability": ("Loop", "prob."),
    "silence_loop_probability": ("Sil. loop", "prob."),
    "subsampling": ("Sub-", "sampling"),
    "num_clusters": (r"\#", "clusters"),
    "num_chunks": (r"\#", "chunks"),
    "initialization": ("Init.",),
    "corpus": ("Corpus",),
}


def param(name: str, header: str | Sequence[str] | None = None, **kwargs) -> Column:
    """A column reading ``name`` out of a row's ``params``."""
    if header is None:
        header = PARAM_HEADERS.get(name, (latex_escape(name.replace("_", " ")),))
    kwargs.setdefault("span", True)
    kwargs.setdefault("block", "param")
    return Column(key=name, header=header, value=_from_param(name), **kwargs)


def loop_probability(
    param_name: str = "lambda",
    transition_scale: float = 1.0,
    header: str | Sequence[str] | None = None,
    **kwargs,
) -> Column:
    """
    A column restating a per-phoneme penalty as the loop probability behind it.

    The transition model reaches the search only through its product
    ``lambda = transition_scale * logit(p)`` (derived and verified against RASR in
    :mod:`...config.vq_unsupervised_frames`), so sweeping lambda is the honest
    axis - a grid over loop probability x transition scale would re-run the same
    lambda at most of its points. A reader, though, sets the two numbers the
    config file actually contains, so a table is easier to use in those units.
    This inverts the relation, ``p = sigmoid(lambda / transition_scale)``, at one
    scale per column; two such columns show the same experiment as the two
    settings that would reproduce it.

    :param param_name: the key holding the penalty, e.g. ``"lambda"`` for the
        training setting or ``"decode_lambda"`` for the one decoded at
    :param transition_scale: the scale the probability is quoted at; the pair
        ``(p, transition_scale)`` reproduces the row's lambda
    """
    if transition_scale <= 0.0:
        raise ValueError(f"transition_scale must be positive, got {transition_scale}")
    if header is None:
        header = ("Loop prob.", f"weight = {transition_scale:g}")

    def source(row: Row):
        lam = row.params.get(param_name)
        if isinstance(lam, bool) or not isinstance(lam, (int, float)):
            return None
        # sigmoid, written so that a large penalty cannot overflow the exp
        z = math.exp(-abs(lam) / transition_scale)
        return 1.0 / (1.0 + z) if lam >= 0 else z / (1.0 + z)

    kwargs.setdefault("fmt", "{:.3f}")
    kwargs.setdefault("span", True)
    kwargs.setdefault("block", "param")
    return Column(
        key=f"{param_name}_loop_ts{transition_scale:g}",
        header=header,
        value=source,
        **kwargs,
    )


def score(attribute: str, header: str | Sequence[str], **kwargs) -> Column:
    """A column reading ``attribute`` off the row's ``DecodeRecogResult``."""
    kwargs.setdefault("block", "score")
    return Column(key=attribute, header=header, value=_from_result(attribute), **kwargs)


def statistic(
    stat_key: str,
    header: str | Sequence[str] | None = None,
    epoch: int | str = "row",
    **kwargs,
) -> Column:
    """
    A column reading ``stat_key`` out of the row's ``epoch_statistics.json``.

    :param stat_key: key inside one epoch's statistics, e.g. ``average_am_score``
    :param epoch: ``"row"`` uses the row's own epoch, ``"last"`` the last epoch the
        statistics file has, an int a fixed epoch. A fixed epoch is constant within a
        group, so pair it with ``span=True`` to get one merged cell per group.
    """
    if header is None:
        header = (latex_escape(stat_key.replace("_", " ")),)
    kwargs.setdefault("block", "stat")
    kwargs.setdefault("span", epoch != "row")
    return Column(key=stat_key, header=header, value=_from_statistic(stat_key, epoch), **kwargs)


def prior_l1(
    header: str | Sequence[str] = ("L1", "distance"),
    epoch: int | str = "row",
    **kwargs,
) -> Column:
    """A column with the prior distance from ``PhonemePriorDistanceJob``."""
    kwargs.setdefault("fmt", "{:.3f}")
    kwargs.setdefault("block", "stat")
    kwargs.setdefault("span", epoch != "row")
    return Column(key="l1", header=header, value=_from_distance(epoch), **kwargs)


def phoneme_fraction(
    phoneme: str = "[SILENCE]",
    header: str | Sequence[str] | None = None,
    epoch: int | str = "row",
    **kwargs,
) -> Column:
    """
    A column with one phoneme's share of the recognized frames.

    :param phoneme: key in ``relative_phoneme_frequencies``, e.g. ``"[SILENCE]"``
    :param epoch: as in :func:`statistic`
    """
    if header is None:
        label = "Sil." if phoneme == "[SILENCE]" else latex_escape(phoneme)
        header = (label, "frac.")
    kwargs.setdefault("fmt", "{:.3f}")
    kwargs.setdefault("block", "stat")
    kwargs.setdefault("span", epoch != "row")
    return Column(
        key=f"frac_{phoneme}", header=header, value=_from_frequency(phoneme, epoch), **kwargs
    )


def scientific_above(
    threshold: float = 1e9, digits: int = 2, plain: str = "{:,.0f}"
) -> Callable[[Any], str]:
    """
    Thousands separators for ordinary values, LaTeX scientific notation for huge ones.

    ``{:,.0f}`` is right for a score around $10^5$ and unreadable for one around
    $10^{20}$ - and a 26-digit cell also drags the whole table past the page. The
    large values are not noise: a Gaussian whose covariance collapses onto a few
    frames sends ``0.5 log det`` towards $-\infty$, so a score of that size is a
    real symptom worth showing rather than truncating.

    :param threshold: absolute value at which to switch to scientific notation
    :param digits: mantissa decimals
    :param plain: format string used below the threshold
    """

    def render(value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        if not math.isfinite(value) or abs(value) < threshold:
            return plain.format(value)
        exponent = math.floor(math.log10(abs(value)))
        mantissa = value / 10.0**exponent
        return f"${mantissa:.{digits}f} \\times 10^{{{exponent}}}$"

    return render


def _percent(name: str) -> str:
    return name + r" [\%]"


#: Ready-made columns addressable by name in ``LatexTableReport(columns=[...])``.
#: Each entry is a factory, so every use gets its own ``Column``.
COLUMN_LIBRARY: dict = {
    # identification
    "experiment": lambda: Column(
        "experiment", "Experiment", _from_result("descriptor"), span=True, align="l"
    ),
    "corpus": lambda: Column("corpus", "Corpus", _from_result("corpus_name"), span=True, align="l"),
    # per-epoch axis
    "epoch": lambda: Column("epoch", "Epoch", _from_epoch, block="score"),
    # decoding scores; JiwerScoringJob reports PER in percent but del/ins/sub as
    # fractions of the reference length, hence the scale
    "per": lambda: score("per", _percent("PER"), fmt="{:.1f}"),
    "del": lambda: Column(
        "del", _percent("Del"), _from_result("deletion"), fmt="{:.1f}", scale=100, block="score"
    ),
    "ins": lambda: Column(
        "ins", _percent("Ins"), _from_result("insertion"), fmt="{:.1f}", scale=100, block="score"
    ),
    "sub": lambda: Column(
        "sub", _percent("Sub"), _from_result("substitution"), fmt="{:.1f}", scale=100, block="score"
    ),
    # nothing in the pipeline computes a frame error rate yet; pass one in per row via
    # add_row(values={"fer": ...}) and this column will pick it up
    "fer": lambda: Column("fer", _percent("FER"), _from_value("fer"), fmt="{:.1f}", block="score"),
    # LM-free frame cluster error rate: fraction of non-silence frames where the
    # model's MAP assignment c*_x = argmax_c p(c)*table[c,x] does NOT match the
    # GMM label.  Pass via add_row(values={"frame_err": FrameClusterAccuracyJob(...).out_error_rate}).
    "frame_err": lambda: Column(
        "frame_err", _percent("Frame err."), _from_value("frame_err"),
        fmt="{:.1f}", scale=100, block="score",
    ),
    # GMM-alignment reference: PER and edit rates measured against the forced-alignment
    # phoneme sequence rather than the text+lexicon transcription. Set when
    # decode_and_score receives gmm_segment_ref.
    "per_gmm": lambda: score("per_gmm", _percent("PER (GMM-ali)"), fmt="{:.1f}"),
    "del_gmm": lambda: Column(
        "del_gmm", _percent("Del (GMM-ali)"), _from_result("deletion_gmm"),
        fmt="{:.1f}", scale=100, block="score",
    ),
    "ins_gmm": lambda: Column(
        "ins_gmm", _percent("Ins (GMM-ali)"), _from_result("insertion_gmm"),
        fmt="{:.1f}", scale=100, block="score",
    ),
    "sub_gmm": lambda: Column(
        "sub_gmm", _percent("Sub (GMM-ali)"), _from_result("substitution_gmm"),
        fmt="{:.1f}", scale=100, block="score",
    ),
    # scores of the guiding recognition itself, over the clustering corpus, as
    # produced by chunked_clustering(score_reference=...). Fill them in with
    # add_row(values=exp_result.guided_score_row(epoch)). Same units as the
    # decoding scores: PER in percent, the edits as fractions.
    "guided_per": lambda: Column(
        "guided_per", ("Guided", _percent("PER")), _from_value("guided_per"),
        fmt="{:.1f}", block="guided",
    ),
    "guided_del": lambda: Column(
        "guided_del", ("Guided", _percent("Del")), _from_value("guided_del"),
        fmt="{:.1f}", scale=100, block="guided",
    ),
    "guided_ins": lambda: Column(
        "guided_ins", ("Guided", _percent("Ins")), _from_value("guided_ins"),
        fmt="{:.1f}", scale=100, block="guided",
    ),
    "guided_sub": lambda: Column(
        "guided_sub", ("Guided", _percent("Sub")), _from_value("guided_sub"),
        fmt="{:.1f}", scale=100, block="guided",
    ),
    # Hypothesis/reference length, 1 - Del + Ins exactly, as for the free-
    # segmentation decode: below one where the search merges segments.
    "guided_hyp_ref": lambda: Column(
        "guided_hyp_ref", ("Guided", "hyp/ref"), _from_value("guided_hyp_ref"),
        fmt="{:.3f}", block="guided",
    ),
    # statistics
    "l1": prior_l1,
    "silence": lambda: phoneme_fraction("[SILENCE]"),
    "am_score": lambda: statistic(
        "average_am_score", ("Avg. AM", "score"), fmt=scientific_above()
    ),
    "lm_score": lambda: statistic(
        "average_lm_score", ("Avg. LM", "score"), fmt=scientific_above()
    ),
    "transition_score": lambda: statistic(
        "average_transition_score", ("Avg. trans.", "score"), fmt=scientific_above()
    ),
    "total_score": lambda: statistic(
        "average_total_score", ("Avg. total", "score"), fmt=scientific_above()
    ),
    "normed_score": lambda: statistic(
        "average_total_normed_score", ("Avg. score", "per frame"), fmt="{:,.2f}"
    ),
    # Forward-backward runs. FBStatisticsCounter replaces the Viterbi traceback
    # counters on that path, so none of the average_*_score keys above exist for
    # them - these are the corresponding numbers, and a report on an FB run wants
    # this group where a Viterbi one wants am/lm/transition.
    #
    # `log_likelihood` is the quantity to read first: it is the actual EM
    # objective, so it is monotone under a correct update and a run that stops
    # improving it has converged in the only sense EM defines.
    "log_likelihood": lambda: statistic(
        "mean_log_likelihood_per_frame", ("Avg. log", "lik./frame"), fmt="{:,.3f}"
    ),
    # The collapse detector on the FB path, standing in for the AM/LM split that
    # a Viterbi run gets: it is the mean per-frame entropy of the label
    # posteriors, so a search that is not separating labels acoustically leaves
    # it pinned near log(num_labels) - 3.69 nats at 40 phonemes - however well
    # the total score is doing.
    "posterior_entropy": lambda: statistic(
        "mean_posterior_entropy", ("Posterior", "entropy"), fmt="{:.3f}"
    ),
    "dead_clusters": lambda: statistic(
        "dead_cluster_count", ("Dead", "clusters"), fmt="{:,}"
    ),
    # The convergence signal the raw log-likelihood hides. That statistic carries
    # the Gaussian normalisation -- hundreds of nats, constant across epochs -- so
    # a run improving by 0.3 nats prints six identical leading digits and looks
    # stuck. Against the run's own first epoch the constant cancels and only the
    # movement is left, which is the number worth reading during training.
    "log_likelihood_gain": lambda: Column(
        "log_likelihood_gain",
        ("Log lik.", "gain"),
        _from_statistic_delta("mean_log_likelihood_per_frame"),
        fmt="{:+.3f}",
        block="stat",
    ),
    # How many labels the posteriors effectively use, as exp(entropy) of the label
    # distribution. Reads against the cluster count: near it, usage is spread;
    # far below, a few labels are absorbing the corpus. Complements
    # "dead_clusters", which only catches labels that have gone to exactly zero
    # and so stays at 0 through a collapse that merely concentrates mass.
    "cluster_perplexity": lambda: Column(
        "cluster_perplexity",
        ("Label", "perplexity"),
        _from_frequencies(),
        fmt=lambda d: f"{_distribution_perplexity(d):.1f}",
        block="stat",
    ),
    "max_cluster_fraction": lambda: Column(
        "max_cluster_fraction",
        ("Largest", "label"),
        _from_frequencies(),
        fmt=lambda d: (f"{max(d.values()) * 100:.1f}" if d else "-"),
        block="stat",
    ),
    # add_row(values={"mi": MixtureDiagnosticsJob(...).out_mi}) fills this in.
    # Reference-free, and unaffected by which search produced the posteriors: it
    # is a function of the mixture weights alone, so it reads the same for a
    # Viterbi and a forward-backward run.
    "mi": lambda: Column(
        "mi", ("I(L;C)", "nats"), _from_value("mi"), fmt="{:.3f}", block="stat"
    ),
    "loop_frequency": lambda: statistic("relative_loop_frequency", ("Loop", "freq."), fmt="{:.3f}"),
    "loop_count": lambda: statistic("absolute_loop_count", ("Loop", "count"), fmt="{:,}"),
    "segment_duration": lambda: statistic(
        "average_segment_duration", ("Avg. segment", "duration"), fmt="{:.2f}"
    ),
    "visited_phonemes": lambda: statistic(
        "fraction_visited_phonemes", ("Visited", "phonemes"), fmt="{:.3f}"
    ),
    # Duration off the posteriors, for the forward-backward path, which has no
    # Viterbi path to count runs along. "segment_duration" above reads the soft
    # estimate (an expectation under a frame-independence assumption, so an
    # upper bound on the boundary rate and a lower bound on duration); this is
    # the run length of the argmax label sequence, which errs the other way.
    # Read them together: far apart means the posteriors are smeared, close
    # means the segmentation is committed. On this corpus the GMM alignment's
    # own mean is 3.99 frames, so that is the number both should approach.
    "segment_duration_argmax": lambda: statistic(
        "average_segment_duration_argmax", ("Avg. dur.", "(argmax)"), fmt="{:.2f}"
    ),
    "boundary_rate": lambda: statistic(
        "boundary_rate_argmax", ("Boundary", "rate"), fmt="{:.3f}"
    ),
    # Emission vs LM, from setup.emission_balance.EmissionBalanceJob; pass its
    # Variables via add_row(values={...}) under these keys. All in nats after the
    # AM scale. "segment_gap" is the acoustic evidence for a label-identity
    # decision; "boundary_payback" is what the acoustics gain by placing a true
    # boundary, read against "boundary_price", the LM's (and the topology's)
    # charge for one more phoneme. "payable_boundaries" is the share of true
    # boundaries whose payback clears the price.
    "segment_gap": lambda: Column(
        "segment_gap", ("Seg.", "gap"), _from_value("segment_gap"),
        fmt="{:.3f}", block="stat"),
    "boundary_payback": lambda: Column(
        "boundary_payback", ("Bnd.", "payback"), _from_value("boundary_payback"),
        fmt="{:.3f}", block="stat"),
    "boundary_price": lambda: Column(
        "boundary_price", ("Bnd.", "price"), _from_value("boundary_price"),
        fmt="{:.3f}", block="stat"),
    "payable_boundaries": lambda: Column(
        "payable_boundaries", _percent("Payable"), _from_value("payable_boundaries"),
        fmt="{:.1f}", scale=100, block="stat"),
    # A second decode_and_score run on cv *frames* with free segmentation (a loop
    # in the recog config) instead of segment means. Pass via add_row(values=...)
    # from its DecodeRecogResult: "matched_per" takes .per (already a
    # percentage), the error types .deletion/.insertion/.substitution, and
    # "matched_hyp_ref" the delayed 1 - deletion + insertion, which is the
    # hypothesis/reference length ratio exactly. Unlike the standard decode,
    # which is handed the boundaries and so pins hypothesis length, these can
    # show a segmentation error - as deletions, and as a ratio below one.
    "matched_per": lambda: Column(
        "matched_per", ("Free-seg.", "PER [\\%]"), _from_value("matched_per"),
        fmt="{:.1f}", block="matched"),
    "matched_del": lambda: Column(
        "matched_del", ("Free-seg.", "Del [\\%]"), _from_value("matched_del"),
        fmt="{:.1f}", scale=100, block="matched"),
    "matched_ins": lambda: Column(
        "matched_ins", ("Free-seg.", "Ins [\\%]"), _from_value("matched_ins"),
        fmt="{:.1f}", scale=100, block="matched"),
    "matched_sub": lambda: Column(
        "matched_sub", ("Free-seg.", "Sub [\\%]"), _from_value("matched_sub"),
        fmt="{:.1f}", scale=100, block="matched"),
    "matched_hyp_ref": lambda: Column(
        "matched_hyp_ref", ("Free-seg.", "hyp/ref"), _from_value("matched_hyp_ref"),
        fmt="{:.3f}", block="matched"),
}


def column_from_name(name: str) -> Column:
    """The library column called ``name``, or a parameter column of that name."""
    factory = COLUMN_LIBRARY.get(name)
    return factory() if factory is not None else param(name)


def relabel(column: str | Column, header: str | Sequence[str], **kwargs) -> Column:
    """
    A ready-made column under a different header, everything else kept.

    The headers in :data:`COLUMN_LIBRARY` name a column the same way in every table
    of the setup, so they carry a qualification the reader of a single table does
    not need: in a table whose caption already says free segmentation,
    ``Free-seg. PER`` is better off as plain ``PER``. Renaming through this keeps
    the value source, format, scale and block; re-declaring the column by hand
    does not -- ``param("matched_del", header="Del")`` reads the wrong place
    (``params`` rather than the row's ``values``) and silently drops the x100
    scale, so the cell comes out a hundred times too small or empty::

        columns=[relabel("matched_per", r"PER [\%]"), "epoch"]

    :param column: a name in :data:`COLUMN_LIBRARY`, or a ``Column`` to copy
    :param header: the new header, one string or one string per stacked line
    :param kwargs: further ``Column`` fields to override along with the header
    """
    if isinstance(column, str):
        column = column_from_name(column)
    return replace(column, header=header, **kwargs)


class TableReport:
    """
    Collects rows, selects and resolves their cells, and leaves rendering to a
    subclass.

    Everything that decides *what* is in the table lives here: the rows, the column
    library, grouping, sorting, epoch filtering, dropping empty rows, and resolving a
    cell through its (possibly still-unfinished) sisyphus Variable. Subclasses add
    only *how* it is printed -- :class:`LatexTableReport` and
    :class:`TextTableReport` differ in nothing else, which is what makes the two
    views of one run guaranteed to agree.

    The instance is the report value handed to ``tk.register_report``: sisyphus calls
    it to (re)write the file, and finds the Variables it depends on by walking the
    rows, so cells fill in as the jobs behind them finish. Cells whose job has not run
    yet render as ``missing``.

    A subclass implements ``__call__`` (returning the file contents) and may override
    :meth:`_escape`, which is applied to string cell values.
    """

    #: Conventional extension for this format, used by :meth:`register` to derive the
    #: ``.deps`` stem.
    SUFFIX = ""

    def __init__(
        self,
        columns: Sequence[str | Column] | None = None,
        group_by: Sequence[str] | None = None,
        sort_by: Sequence[str] | None = None,
        epochs: Sequence[int] | None = None,
        drop_empty_rows: bool = False,
        row_filter: Callable[[Row], bool] | None = None,
        caption: str | None = None,
        highlight: Callable[[Row, str], bool] | None = None,
        missing: str = "",
    ):
        """
        :param columns: column names and/or ``Column`` objects; None derives them
        :param group_by: parameter keys defining a row group; None uses all parameters,
            so consecutive epochs of one experiment end up in the same group
        :param sort_by: parameter keys to sort rows by; None keeps insertion order
        :param drop_empty_rows: omit rows whose every computed cell is still missing.
            A run of ``num_epochs`` registers a row per epoch whether or not that
            epoch has finished, so a table of a run in progress otherwise ends in a
            block of blank lines - and a float cannot break across pages, so a long
            enough table is silently truncated rather than continued. With this, the
            table shows the epochs that exist and grows as the run does. Rows are
            judged by what renders, so a row is kept as soon as any one of its cells
            has a value.
        :param epochs: keep only rows on these epochs, e.g. ``(0, 9)`` for a table
            that just contrasts the start and end of training
        :param row_filter: keep only rows this returns True for, applied on top of
            ``epochs``
        :param caption: table caption. Written for the richest format in use, so it
            may contain LaTeX; :class:`TextTableReport` strips the markup it knows.
        :param highlight: called with ``(row, column_key)``; True marks that cell, in
            whatever way the format has
        :param missing: text for cells whose value is not available
        """
        self._columns = list(columns) if columns is not None else None
        self.group_by = list(group_by) if group_by is not None else None
        self.sort_by = list(sort_by) if sort_by is not None else None
        self.epochs = tuple(epochs) if epochs is not None else None
        self.drop_empty_rows = drop_empty_rows
        self.row_filter = row_filter
        self.caption = caption
        self.highlight = highlight
        self.missing = missing

        self.rows: list[Row] = []

    def _settings(self) -> dict:
        """Constructor keywords reproducing this report, for :meth:`view`."""
        return dict(
            columns=self._columns,
            group_by=self.group_by,
            sort_by=self.sort_by,
            epochs=self.epochs,
            drop_empty_rows=self.drop_empty_rows,
            row_filter=self.row_filter,
            caption=self.caption,
            highlight=self.highlight,
            missing=self.missing,
        )

    @staticmethod
    def _escape(text: str) -> str:
        """Applied to string cell values; the plain-text format needs no escaping."""
        return text

    # -- building -----------------------------------------------------------------

    def add_row(
        self,
        result: DecodeRecogResult | None = None,
        params: dict | None = None,
        epoch: int | None = None,
        statistics: StatisticsSource | tk.Path | None = None,
        values: dict | None = None,
    ) -> Row:
        """
        Add one line to the table.

        :param result: the decoding result, source of the PER/Del/Ins/Sub columns
        :param params: hyperparameters of this run; also the default grouping key
        :param epoch: which epoch's centroids were decoded, also the default epoch for
            the statistics columns
        :param statistics: a ``StatisticsSource``, or the clustering job's
            ``out_statistics`` path to build one with the default settings
        :param values: any further per-row values, addressed by column key
        """
        if isinstance(statistics, tk.Path):
            statistics = clustering_statistics(statistics)
        if statistics is not None and statistics.name is None and result is not None:
            statistics.name = result.descriptor
        row = Row(
            params=dict(params or {}),
            epoch=epoch,
            result=result,
            statistics=statistics,
            values=dict(values or {}),
        )
        self.rows.append(row)
        return row

    def add_rows(self, results: Iterable[DecodeRecogResult], **kwargs) -> None:
        """Add one row per result, all sharing the given keyword arguments."""
        for result in results:
            self.add_row(result=result, **kwargs)

    def view(self, **overrides) -> "TableReport":
        """
        A second report over the same rows, with some settings changed.

        Typical use is a condensed version of a table next to the full one::

            summary = report.view(epochs=(0, 9), caption="First and last epoch")

        The row list is shared rather than copied, so it does not matter whether this
        is called before or after the rows are added.

        :param overrides: any constructor keyword argument to change
        """
        settings = self._settings()
        settings.update(overrides)
        other = type(self)(**settings)
        other.rows = self.rows
        return other

    def as_text(self, **overrides) -> "TextTableReport":
        """
        The same rows and the same column selection, rendered as plain text.

        The point of the shared base: this is not a second table that happens to
        agree, it is the same selection logic with a different printer, so a
        discrepancy between the two is impossible rather than merely unlikely.
        Settings this format has no use for (float placement, LaTeX rules) are
        dropped, and anything :class:`TextTableReport` accepts can be overridden.

        :param overrides: constructor keywords for the text report
        """
        settings = {
            key: value
            for key, value in self._settings().items()
            if key in TableReport.__init__.__code__.co_varnames
        }
        settings.update(overrides)
        other = TextTableReport(**settings)
        other.rows = self.rows
        return other

    def register(
        self,
        output_path: str,
        required: Any = True,
        register_dependencies: bool = True,
        update_frequency: int = 300,
    ):
        """
        Register the table as a sisyphus report under ``output_path``.

        Also registers the statistics Variables as outputs. Sisyphus only schedules
        jobs that a registered output depends on, and a report registered with
        ``required=True`` does not pull its inputs into the graph - the decoding jobs
        are registered by the configs themselves, but the statistics jobs created here
        would otherwise never run and their columns would stay empty. A Variable is
        only registered once per process, so registering several views of the same
        rows does not produce several copies of the same ``.deps`` tree.

        :param output_path: path under ``output/``, conventionally ending in this
            format's ``SUFFIX``
        :param required: passed through to ``tk.register_report``. Sisyphus treats it as
            the paths the report waits on, not as a flag: ``True`` holds no paths, so
            the report counts as done at once, is written a single time and is only
            rewritten when the manager restarts. Pass the report itself
            (``required=report``) to have it rewritten every ``update_frequency``
            seconds while its inputs are being computed.
        :param register_dependencies: set False to skip the ``.deps`` outputs, when
            something else in the config already keeps the statistics jobs alive
        :param update_frequency: seconds between rewrites, see ``required``
        """
        suffix = self.SUFFIX
        stem = (
            output_path[: -len(suffix)]
            if suffix and output_path.endswith(suffix)
            else output_path
        )

        if register_dependencies:
            for index, row in enumerate(self.rows):
                if row.statistics is None:
                    continue
                name = row.statistics.name or f"{index:03d}"
                for suffix, variable in (
                    ("statistics", row.statistics.scalars),
                    ("prior_distance", row.statistics.distances),
                    ("phoneme_frequencies", row.statistics.frequencies),
                ):
                    if variable is None:
                        continue
                    # One Variable for the whole run, or one per epoch.
                    entries = (
                        [(f"{suffix}.{epoch:03d}", v) for epoch, v in sorted(variable.items())]
                        if isinstance(variable, dict)
                        else [(suffix, variable)]
                    )
                    for label, entry in entries:
                        if entry is None or entry.get_path() in _REGISTERED_DEPENDENCIES:
                            continue
                        _REGISTERED_DEPENDENCIES.add(entry.get_path())
                        tk.register_output(f"{stem}.deps/{name}.{label}", entry)

        return tk.register_report(
            output_path, values=self, required=required, update_frequency=update_frequency
        )

    # -- selection ----------------------------------------------------------------

    def _column_from_name(self, name: str) -> Column:
        return column_from_name(name)

    def _visible_rows(self) -> list:
        rows = self.rows
        if self.epochs is not None:
            rows = [row for row in rows if row.epoch in self.epochs]
        if self.row_filter is not None:
            rows = [row for row in rows if self.row_filter(row)]
        return list(rows)

    def _param_keys(self, rows: list) -> list:
        keys = []
        for row in rows:
            for key in row.params:
                if key not in keys:
                    keys.append(key)
        return keys

    def _auto_column_names(self, rows: list) -> list:
        param_keys = self._param_keys(rows)
        varying = [
            key for key in param_keys if len({_hashable(row.params.get(key)) for row in rows}) > 1
        ]
        names = list(varying or param_keys)
        if any(row.epoch is not None for row in rows):
            names.append("epoch")
        if any(row.result is not None for row in rows):
            names += ["per", "del", "ins", "sub"]
        if any(row.statistics is not None and row.statistics.distances is not None for row in rows):
            names.append("l1")
        if any(row.statistics is not None for row in rows):
            names += ["am_score", "lm_score"]
        if not names:
            names = ["experiment"]
        return names

    def _resolve_columns(self, rows: list) -> list:
        specs = self._columns if self._columns is not None else self._auto_column_names(rows)
        columns = []
        previous_block = None
        for spec in specs:
            column = spec if isinstance(spec, Column) else self._column_from_name(spec)
            if column.rule_before is None:
                rule = "||" if previous_block is not None and column.block != previous_block else "|"
                column = replace(column, rule_before=rule)
            previous_block = column.block
            columns.append(column)
        return columns

    def _sorted_rows(self, rows: list) -> list:
        if not self.sort_by:
            return list(rows)

        def sort_key(row: Row):
            key = []
            for name in self.sort_by:
                value = row.params.get(name)
                if value is None:
                    key.append((2, 0.0, ""))
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    key.append((0, float(value), ""))
                else:
                    key.append((1, 0.0, str(value)))
            key.append((0, float(row.epoch if row.epoch is not None else 0), ""))
            return key

        return sorted(rows, key=sort_key)

    def _groups(self, rows: list) -> list:
        group_by = self.group_by if self.group_by is not None else self._param_keys(rows)

        def group_key(row: Row):
            return tuple(_hashable(row.params.get(name)) for name in group_by)

        return [list(group) for _, group in itertools.groupby(rows, key=group_key)]

    def _cell_text(self, column: Column, row: Row) -> str:
        value = column.value(row)
        if isinstance(value, DelayedBase):
            if not value.is_set():
                value = MISSING
            else:
                try:
                    value = value.get()
                except (KeyError, IndexError, TypeError):
                    # the job ran but does not have this epoch/key, e.g. the last set
                    # of centroids, which no recognition pass produced statistics for
                    value = MISSING
        if value is MISSING or value is None:
            return column.missing if column.missing is not None else self.missing
        if isinstance(value, (int, float)) and not isinstance(value, bool) and column.scale is not None:
            value = value * column.scale
        if isinstance(value, str) and not column.raw:
            value = self._escape(value)
        try:
            return column.fmt(value) if callable(column.fmt) else column.fmt.format(value)
        except (ValueError, TypeError, IndexError, KeyError):
            return str(value)

    def _row_has_data(self, columns: list, row: Row) -> bool:
        """
        True if any computed cell of ``row`` renders as something.

        Judged from the rendered text rather than from the Variables, so what counts
        as present here is exactly what would show in the table. Cells that are not
        computed at all - parameters, the epoch index - are ignored, or every row
        would qualify.
        """
        for column in columns:
            if not isinstance(column.value(row), DelayedBase):
                continue
            missing = column.missing if column.missing is not None else self.missing
            if self._cell_text(column, row) != missing:
                return True
        return False

    def _prepared(self):
        """
        ``(groups, columns)`` after filtering, column resolution and sorting, or
        ``(None, None)`` when there is nothing to show.

        The single place the two formats get their content from, so a row that one
        of them drops is dropped by both.
        """
        rows = self._visible_rows()
        if not rows:
            return None, None
        columns = self._resolve_columns(rows)
        if self.drop_empty_rows:
            rows = [row for row in rows if self._row_has_data(columns, row)]
            if not rows:
                return [], columns
        return self._groups(self._sorted_rows(rows)), columns

    def __call__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self()


class LatexTableReport(TableReport):
    """
    Renders the rows as a LaTeX ``table``.

    Adds only the float wrapper, the ``tabular`` body, ``\multirow`` spanning and
    ``\cellcolor`` highlighting. What goes in the table is :class:`TableReport`'s.
    """

    SUFFIX = ".tex"

    def __init__(
        self,
        columns: Sequence[str | Column] | None = None,
        group_by: Sequence[str] | None = None,
        sort_by: Sequence[str] | None = None,
        epochs: Sequence[int] | None = None,
        drop_empty_rows: bool = False,
        row_filter: Callable[[Row], bool] | None = None,
        caption: str | None = None,
        label: str | None = None,
        placement: str = "h!",
        highlight: Callable[[Row, str], bool] | None = None,
        highlight_color: str = "FFCE93",
        highlight_text_color: str = "333333",
        missing: str = "",
        group_rule: str = r"\hline \hline",
        preamble_comment: bool = True,
    ):
        """
        :param label: ``\\label`` for the table
        :param placement: float placement specifier
        :param highlight_color: cell background, as an HTML colour
        :param highlight_text_color: text colour inside a highlighted cell
        :param group_rule: rule drawn between row groups
        :param preamble_comment: emit a comment naming the packages the table needs

        Every other parameter is :class:`TableReport`'s.
        """
        super().__init__(
            columns=columns,
            group_by=group_by,
            sort_by=sort_by,
            epochs=epochs,
            drop_empty_rows=drop_empty_rows,
            row_filter=row_filter,
            caption=caption,
            highlight=highlight,
            missing=missing,
        )
        self.label = label
        self.placement = placement
        self.highlight_color = highlight_color
        self.highlight_text_color = highlight_text_color
        self.group_rule = group_rule
        self.preamble_comment = preamble_comment

    def _settings(self) -> dict:
        settings = super()._settings()
        settings.update(
            label=self.label,
            placement=self.placement,
            highlight_color=self.highlight_color,
            highlight_text_color=self.highlight_text_color,
            group_rule=self.group_rule,
            preamble_comment=self.preamble_comment,
        )
        return settings

    @staticmethod
    def _escape(text: str) -> str:
        return latex_escape(text)

    def _decorate(self, text: str, row: Row, column: Column) -> str:
        if self.highlight is None or not self.highlight(row, column.key):
            return text
        return (
            r"\cellcolor[HTML]{%s}{\color[HTML]{%s} %s}"
            % (self.highlight_color, self.highlight_text_color, text)
        )

    @staticmethod
    def _header_cell(header: str | Sequence[str]) -> str:
        lines = [header] if isinstance(header, str) else list(header)
        if len(lines) == 1:
            return lines[0]
        return r"\begin{tabular}[c]{@{}c@{}}" + r"\\ ".join(lines) + r"\end{tabular}"

    @staticmethod
    def _clines(spanning: list) -> str:
        """``\\cline`` over each run of columns that is not merged by ``\\multirow``."""
        parts = []
        start = None
        for index, is_spanning in enumerate(spanning, start=1):
            if not is_spanning and start is None:
                start = index
            elif is_spanning and start is not None:
                parts.append(r"\cline{%d-%d}" % (start, index - 1))
                start = None
        if start is not None:
            parts.append(r"\cline{%d-%d}" % (start, len(spanning)))
        return " ".join(parts)

    def _tabular_spec(self, columns: list) -> str:
        spec = "|"
        for index, column in enumerate(columns):
            spec += column.align
            spec += columns[index + 1].rule_before if index + 1 < len(columns) else "|"
        return spec

    def _header_line(self, columns: list) -> str:
        cells = []
        for index, column in enumerate(columns):
            left = "|" if index == 0 else ""
            right = columns[index + 1].rule_before if index + 1 < len(columns) else "|"
            cells.append(
                r"\multicolumn{1}{%s%s%s}{%s}"
                % (left, column.header_align or "c", right, self._header_cell(column.header))
            )
        return " & ".join(cells) + r" \\ \hline \hline"

    def __call__(self) -> str:
        groups, columns = self._prepared()
        if columns is None:
            return "% no rows registered for this table\n"
        if not groups:
            return "% every row is still empty\n"

        lines = []
        for group_index, group in enumerate(groups):
            texts = [[self._cell_text(column, row) for column in columns] for row in group]
            # a column only merges if it really is constant across the group; a
            # parameter that varies inside a group would otherwise be silently dropped
            spanning = [
                column.span and len({text[index] for text in texts}) == 1
                for index, column in enumerate(columns)
            ]
            cline = self._clines(spanning)

            for row_index, row in enumerate(group):
                cells = []
                for index, column in enumerate(columns):
                    if spanning[index]:
                        text = texts[0][index]
                        if row_index > 0:
                            cells.append("")
                        elif len(group) > 1 and text:
                            cells.append(r"\multirow{%d}{*}{%s}" % (len(group), text))
                        else:
                            cells.append(text)
                    else:
                        cells.append(self._decorate(texts[row_index][index], row, column))
                line = " & ".join(cells) + r" \\"
                if row_index < len(group) - 1 and cline:
                    line += " " + cline
                lines.append(line)

            lines[-1] += " " + (self.group_rule if group_index < len(groups) - 1 else r"\hline")

        out = []
        if self.preamble_comment:
            packages = [r"\usepackage{multirow}"]
            if self.highlight is not None:
                packages.append(r"\usepackage[table]{xcolor}")
            out.append("% requires " + ", ".join(packages))
        out.append(r"\begin{table}[%s]" % self.placement)
        out.append(r"\centering")
        if self.caption is not None:
            out.append(r"\caption{%s}" % self.caption)
        if self.label is not None:
            out.append(r"\label{%s}" % self.label)
        out.append(r"\begin{tabular}{%s}" % self._tabular_spec(columns))
        out.append(r"\hline")
        out.append(self._header_line(columns))
        out.extend(lines)
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        return "\n".join(out) + "\n"


#: Replaced before any command handling, because these are literal strings rather
#: than commands with arguments.
_DELATEX_LITERALS = (
    (r"\begin{tabular}[c]{@{}c@{}}", " "), (r"\end{tabular}", " "),
    (r"\%", "%"), (r"\_", "_"), (r"\&", "&"), (r"\#", "#"), (r"\$", "$"),
)

#: Commands whose name is noise but whose argument is content: dropped, argument
#: kept. ``\mathcal{N}`` has to become ``N`` here, before the braces go, or it
#: would read as ``mathcalN``.
_DELATEX_WRAPPERS = (
    "mathcal", "mathrm", "mathbf", "mathit", "mathsf", "textbf", "textit",
    "texttt", "emph", "text", "operatorname",
)

#: Commands that are pure spacing or sizing, with no argument worth keeping.
_DELATEX_SPACING = ("left", "right", "quad", "qquad", "," , ";", "!", "/")

#: Math operators with a readable ASCII equivalent. Without these the generic
#: fallback would render ``\sim`` as ``sim``, which reads as a word.
_DELATEX_SYMBOLS = {
    "sim": "~", "approx": "~=", "times": "x", "cdot": ".", "pm": "+-",
    "to": "->", "rightarrow": "->", "leftarrow": "<-", "ll": "<<", "gg": ">>",
    "leq": "<=", "geq": ">=", "neq": "!=", "infty": "inf", "ldots": "...",
    "dots": "...", "cdots": "...",
}


def delatex(text: str) -> str:
    """Strip the LaTeX a caption or header written for the .tex output carries.

    Best-effort and lossy by design: the aim is a caption a person can read in a
    terminal, not a faithful de-rendering. A command this does not know keeps its
    name and loses its backslash, so ``\\sigma`` reads as ``sigma`` rather than
    disappearing -- wrong-looking is recoverable, silently dropped is not.
    """
    for needle, replacement in _DELATEX_LITERALS:
        text = text.replace(needle, replacement)
    # Line breaks before the generic command pass, which would eat the backslashes.
    text = re.sub(r"\\\\\s*", " ", text)
    for command in _DELATEX_WRAPPERS:
        # Innermost first, so nesting unwraps rather than leaving a stray name.
        pattern = r"\\" + command + r"\s*\{([^{}]*)\}"
        while re.search(pattern, text):
            text = re.sub(pattern, r"\1", text)
        text = re.sub(r"\\" + command + r"(?![a-zA-Z])", " ", text)
    for command in _DELATEX_SPACING:
        text = re.sub(r"\\" + re.escape(command) + r"(?![a-zA-Z])", " ", text)
    for command, symbol in _DELATEX_SYMBOLS.items():
        text = re.sub(r"\\" + command + r"(?![a-zA-Z])", symbol, text)
    # Everything left keeps its name: \tau -> tau, \Sigma -> Sigma.
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("\\", " ")
    for char in "${}":
        text = text.replace(char, "")
    return " ".join(text.split())


class TextTableReport(TableReport):
    """
    Renders the rows as a fixed-width plain-text table.

    Exists so a run can be read without a LaTeX pass -- during training above all,
    where the question is whether the numbers are moving at all. It is deliberately
    a sibling of :class:`LatexTableReport` rather than a reimplementation: rows,
    columns, grouping, sorting, epoch filtering and cell resolution all come from
    :class:`TableReport`, so the two cannot drift apart. ``report.as_text()`` gives
    the text twin of a table already built for LaTeX.

    Differences from the LaTeX output, all of them presentational:

    * stacked headers are flattened to one line and LaTeX markup is stripped
    * a spanning column shows its value on the group's first row and is blank
      underneath, which is what ``\multirow`` looks like
    * a highlighted cell is marked with ``*`` instead of coloured
    """

    SUFFIX = ".txt"

    def __init__(
        self,
        columns: Sequence[str | Column] | None = None,
        group_by: Sequence[str] | None = None,
        sort_by: Sequence[str] | None = None,
        epochs: Sequence[int] | None = None,
        drop_empty_rows: bool = False,
        row_filter: Callable[[Row], bool] | None = None,
        caption: str | None = None,
        highlight: Callable[[Row, str], bool] | None = None,
        missing: str = "-",
        title: str | None = None,
        separator: str = "  ",
        max_caption_width: int = 100,
    ):
        """
        :param missing: defaults to ``-`` rather than the empty string, so a cell
            that has no value yet is visibly pending instead of looking like a
            column that was never asked for -- the common case while a run is in
            flight, which is when this format gets read
        :param title: heading above the table; the caption is used when unset
        :param separator: column separator
        :param max_caption_width: wrap the caption at this width

        Every other parameter is :class:`TableReport`'s.
        """
        super().__init__(
            columns=columns,
            group_by=group_by,
            sort_by=sort_by,
            epochs=epochs,
            drop_empty_rows=drop_empty_rows,
            row_filter=row_filter,
            caption=caption,
            highlight=highlight,
            missing=missing,
        )
        self.title = title
        self.separator = separator
        self.max_caption_width = max_caption_width

    def _settings(self) -> dict:
        settings = super()._settings()
        settings.update(
            title=self.title,
            separator=self.separator,
            max_caption_width=self.max_caption_width,
        )
        return settings

    @staticmethod
    def _header_cell(header: str | Sequence[str]) -> str:
        lines = [header] if isinstance(header, str) else list(header)
        return delatex(" ".join(lines))

    def _decorate(self, text: str, row: Row, column: Column) -> str:
        if self.highlight is None or not self.highlight(row, column.key):
            return text
        return f"{text}*"

    def _wrap(self, text: str) -> list[str]:
        words, lines, current = text.split(), [], ""
        for word in words:
            if current and len(current) + 1 + len(word) > self.max_caption_width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)
        return lines

    def __call__(self) -> str:
        groups, columns = self._prepared()
        if columns is None:
            return "no rows registered for this table\n"
        if not groups:
            return "every row is still empty\n"

        headers = [self._header_cell(column.header) for column in columns]
        # Blank the repeats of a spanning column, as \multirow does, but only when
        # the value really is constant across the group - the same test the LaTeX
        # renderer applies, so the two agree on what is merged.
        body: list[list[str] | None] = []
        for group_index, group in enumerate(groups):
            texts = [[self._cell_text(column, row) for column in columns] for row in group]
            spanning = [
                column.span and len({text[index] for text in texts}) == 1
                for index, column in enumerate(columns)
            ]
            for row_index, row in enumerate(group):
                cells = []
                for index, column in enumerate(columns):
                    if spanning[index] and row_index > 0:
                        cells.append("")
                    elif spanning[index]:
                        cells.append(texts[0][index])
                    else:
                        cells.append(self._decorate(texts[row_index][index], row, column))
                body.append(cells)
            if group_index < len(groups) - 1:
                body.append(None)          # group separator

        widths = [
            max(len(headers[i]), *(len(r[i]) for r in body if r is not None))
            for i in range(len(columns))
        ]

        def line(cells: list[str]) -> str:
            out = []
            for index, (cell, width) in enumerate(zip(cells, widths)):
                out.append(cell.ljust(width) if columns[index].align == "l"
                           else cell.rjust(width))
            return self.separator.join(out).rstrip()

        rule = self.separator.join("-" * width for width in widths)
        out = []
        title = self.title if self.title is not None else (
            delatex(self.caption) if self.caption else None)
        if title:
            out.extend(self._wrap(title))
            out.append("")
        out.append(line(headers))
        out.append(rule)
        for cells in body:
            out.append(rule.replace("-", " ").rstrip() if cells is None else line(cells))
        out.append("")
        return "\n".join(out) + "\n"


def create_latex_report(
    recog_results: Sequence[DecodeRecogResult],
    params: Sequence[dict] | None = None,
    epochs: Sequence[int] | None = None,
    statistics: StatisticsSource | tk.Path | None = None,
    **kwargs,
) -> LatexTableReport:
    """
    Build a report from a flat list of decoding results, mirroring ``create_report``.

    Without ``params`` every result becomes its own group, labelled by its descriptor -
    useful as a drop-in for the plain-text report. Configs that know their
    hyperparameters are better off calling ``LatexTableReport.add_row`` in the sweep
    loop, which is what gives the grouped-by-hyperparameter layout.

    :param recog_results: the decoding results, one per row
    :param params: per-result hyperparameters, same length as ``recog_results``
    :param epochs: per-result epoch, same length as ``recog_results``
    :param statistics: statistics source shared by all rows
    :param kwargs: forwarded to ``LatexTableReport``
    """
    report = LatexTableReport(**kwargs)
    for index, result in enumerate(recog_results):
        row_params = dict(params[index]) if params is not None else {"experiment": result.descriptor}
        report.add_row(
            result=result,
            params=row_params,
            epoch=epochs[index] if epochs is not None else None,
            statistics=statistics,
        )
    return report
