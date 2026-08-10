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
* any other string, which becomes a column reading that key out of ``params``,
* a ``Column`` built by hand or by ``param()`` / ``statistic()`` / ``prior_l1()``
  when the defaults do not fit.

Leaving ``columns=None`` derives them: the hyperparameters that actually vary, then
epoch, the decoding scores, and the statistics columns if a statistics source was
given.

Rendering the table needs ``\usepackage{multirow}`` in the document, plus
``\usepackage[table]{xcolor}`` when ``highlight`` is used.
"""

__all__ = [
    "Row",
    "Column",
    "LatexTableReport",
    "StatisticsSource",
    "clustering_statistics",
    "param",
    "statistic",
    "prior_l1",
    "score",
    "COLUMN_LIBRARY",
    "PARAM_HEADERS",
    "create_latex_report",
    "latex_escape",
]

import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase
from sisyphus.tools import try_get

from .constants import PHONEME_UNIGRAM_PRIORS
from .decode_config import DecodeRecogResult
from .statistics_jobs import ExtractEpochStatisticsJob, PhonemePriorDistanceJob


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

    def get(self):
        values = try_get(self.a)
        epoch = max(values) if self.epoch == "last" else self.epoch
        value = values[epoch]
        return value if self.key is None else value[self.key]


@dataclass(eq=False)
class StatisticsSource:
    """
    The Variables derived from one clustering job's ``epoch_statistics.json``.

    :param scalars: ``{epoch: {key: value}}``, from ``ExtractEpochStatisticsJob``
    :param distances: ``{epoch: distance}``, from ``PhonemePriorDistanceJob``, or None
    :param name: used when registering the Variables as outputs; filled in from the
        first row's descriptor when left unset
    :param epoch_offset: added to a row's epoch to get the statistics key, see
        ``clustering_statistics``
    """

    scalars: tk.Variable
    distances: tk.Variable | None = None
    name: str | None = None
    epoch_offset: int = 0


def clustering_statistics(
    statistics: tk.Path,
    reference_counts: tk.Path | None = PHONEME_UNIGRAM_PRIORS,
    name: str | None = None,
    epoch_offset: int = 0,
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
    :param distance_kwargs: forwarded to ``PhonemePriorDistanceJob``
        (``exclude_phonemes``, ``renormalize``, ``order``)
    """
    scalars = ExtractEpochStatisticsJob(statistics).out_statistics
    distances = None
    if reference_counts is not None:
        distances = PhonemePriorDistanceJob(statistics, reference_counts, **distance_kwargs).out_distances
    return StatisticsSource(scalars=scalars, distances=distances, name=name, epoch_offset=epoch_offset)


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
    :param fmt: ``str.format`` spec applied to the resolved value
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
    fmt: str = "{}"
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
    # statistics
    "l1": prior_l1,
    "am_score": lambda: statistic("average_am_score", ("Avg. AM", "score"), fmt="{:,.0f}"),
    "lm_score": lambda: statistic("average_lm_score", ("Avg. LM", "score"), fmt="{:,.0f}"),
    "total_score": lambda: statistic("average_total_score", ("Avg. total", "score"), fmt="{:,.0f}"),
    "normed_score": lambda: statistic(
        "average_total_normed_score", ("Avg. score", "per frame"), fmt="{:,.2f}"
    ),
    "loop_frequency": lambda: statistic("relative_loop_frequency", ("Loop", "freq."), fmt="{:.3f}"),
    "loop_count": lambda: statistic("absolute_loop_count", ("Loop", "count"), fmt="{:,}"),
    "segment_duration": lambda: statistic(
        "average_segment_duration", ("Avg. segment", "duration"), fmt="{:.2f}"
    ),
    "visited_phonemes": lambda: statistic(
        "fraction_visited_phonemes", ("Visited", "phonemes"), fmt="{:.3f}"
    ),
}


class LatexTableReport:
    """
    Collects rows and renders them as a LaTeX ``table``.

    The instance is the report value handed to ``tk.register_report``: sisyphus calls
    it to (re)write the file, and finds the Variables it depends on by walking the
    rows, so cells fill in as the jobs behind them finish. Cells whose job has not run
    yet render as ``missing``.
    """

    def __init__(
        self,
        columns: Sequence[str | Column] | None = None,
        group_by: Sequence[str] | None = None,
        sort_by: Sequence[str] | None = None,
        epochs: Sequence[int] | None = None,
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
        :param columns: column names and/or ``Column`` objects; None derives them
        :param group_by: parameter keys defining a row group; None uses all parameters,
            so consecutive epochs of one experiment end up in the same group
        :param sort_by: parameter keys to sort rows by; None keeps insertion order
        :param epochs: keep only rows on these epochs, e.g. ``(0, 9)`` for a table
            that just contrasts the start and end of training
        :param row_filter: keep only rows this returns True for, applied on top of
            ``epochs``
        :param caption: table caption, emitted verbatim (write it as LaTeX)
        :param label: ``\\label`` for the table
        :param placement: float placement specifier
        :param highlight: called with ``(row, column_key)``; True colours that cell
        :param highlight_color: cell background, as an HTML colour
        :param highlight_text_color: text colour inside a highlighted cell
        :param missing: text for cells whose value is not available
        :param group_rule: rule drawn between row groups
        :param preamble_comment: emit a comment naming the packages the table needs
        """
        self._columns = list(columns) if columns is not None else None
        self.group_by = list(group_by) if group_by is not None else None
        self.sort_by = list(sort_by) if sort_by is not None else None
        self.epochs = tuple(epochs) if epochs is not None else None
        self.row_filter = row_filter
        self.caption = caption
        self.label = label
        self.placement = placement
        self.highlight = highlight
        self.highlight_color = highlight_color
        self.highlight_text_color = highlight_text_color
        self.missing = missing
        self.group_rule = group_rule
        self.preamble_comment = preamble_comment

        self.rows: list[Row] = []

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

    def view(self, **overrides) -> "LatexTableReport":
        """
        A second report over the same rows, with some settings changed.

        Typical use is a condensed version of a table next to the full one::

            summary = report.view(epochs=(0, 9), caption="First and last epoch")

        The row list is shared rather than copied, so it does not matter whether this
        is called before or after the rows are added.

        :param overrides: any constructor keyword argument to change
        """
        settings = dict(
            columns=self._columns,
            group_by=self.group_by,
            sort_by=self.sort_by,
            epochs=self.epochs,
            row_filter=self.row_filter,
            caption=self.caption,
            label=self.label,
            placement=self.placement,
            highlight=self.highlight,
            highlight_color=self.highlight_color,
            highlight_text_color=self.highlight_text_color,
            missing=self.missing,
            group_rule=self.group_rule,
            preamble_comment=self.preamble_comment,
        )
        settings.update(overrides)
        other = LatexTableReport(**settings)
        other.rows = self.rows
        return other

    def register(self, output_path: str, required: bool = True, register_dependencies: bool = True):
        """
        Register the table as a sisyphus report under ``output_path``.

        Also registers the statistics Variables as outputs. Sisyphus only schedules
        jobs that a registered output depends on, and a report registered with
        ``required=True`` does not pull its inputs into the graph - the decoding jobs
        are registered by the configs themselves, but the statistics jobs created here
        would otherwise never run and their columns would stay empty. A Variable is
        only registered once per process, so registering several views of the same
        rows does not produce several copies of the same ``.deps`` tree.

        :param output_path: path under ``output/``, conventionally ending in ``.tex``
        :param required: passed through to ``tk.register_report``
        :param register_dependencies: set False to skip the ``.deps`` outputs, when
            something else in the config already keeps the statistics jobs alive
        """
        stem = output_path[: -len(".tex")] if output_path.endswith(".tex") else output_path

        if register_dependencies:
            for index, row in enumerate(self.rows):
                if row.statistics is None:
                    continue
                name = row.statistics.name or f"{index:03d}"
                for suffix, variable in (
                    ("statistics", row.statistics.scalars),
                    ("prior_distance", row.statistics.distances),
                ):
                    if variable is None or variable.get_path() in _REGISTERED_DEPENDENCIES:
                        continue
                    _REGISTERED_DEPENDENCIES.add(variable.get_path())
                    tk.register_output(f"{stem}.deps/{name}.{suffix}", variable)

        return tk.register_report(output_path, values=self, required=required)

    # -- rendering ----------------------------------------------------------------

    def _column_from_name(self, name: str) -> Column:
        factory = COLUMN_LIBRARY.get(name)
        return factory() if factory is not None else param(name)

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
            value = latex_escape(value)
        try:
            return column.fmt.format(value)
        except (ValueError, TypeError, IndexError, KeyError):
            return str(value)

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
        rows = self._visible_rows()
        if not rows:
            return "% no rows registered for this table\n"

        columns = self._resolve_columns(rows)
        groups = self._groups(self._sorted_rows(rows))

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

    def __str__(self) -> str:
        return self()


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
