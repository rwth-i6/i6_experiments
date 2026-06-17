"""
Global results registry + a single combined summary table for the pHMM phon
recognition experiments.

Both the lexicon-constrained search and the merged lexicon-free search (all assembled in
``config_01_phon_phmm`` -- and the analogous ``config_02_phon_ctc``) push their
per-(model, epoch, variant, dataset) scores here via :func:`add_result`. A single
``tk.register_report`` then renders all of them into ONE table at the top-level experiment
prefix (``.../ls960_phmm_eow_phon/summary.report``).

Each row can carry up to two metrics: ``WER`` (word error rate) and ``PER`` (phoneme error
rate). The lexicon-free search reports **PER only** (its word-level WER is meaningless because
the hypothesis is a phoneme stream); the lexicon-constrained search reports **WER** (primary)
and **PER** (auxiliary, for a like-for-like phoneme comparison against the lexicon-free path).

The table is a *live* sisyphus report (see :class:`sisyphus.graph.OutputReport`):
it is re-rendered periodically while the manager runs, so entries fill in as the
underlying sclite jobs finish. Unfinished WERs show ``--`` instead of blocking the
whole table (we guard every ``Variable`` with ``is_set()`` before ``.get()`` --
unset variables raise ``VariableNotSet`` by default). This mirrors the global-dict
convention used by :mod:`.storage`.
"""
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

# Datasets are shown in this fixed order; only those actually present are printed.
_DATASET_ORDER = ["dev-clean", "dev-other", "test-clean", "test-other"]
# Metrics are shown in this fixed order; only those actually present are printed. "ms" is the
# average label duration (milliseconds) used by the duration-analysis section.
_METRIC_ORDER = ["WER", "PER", "ms"]

# prefix -> OrderedDict(row_key -> row dict). One table per top-level experiment prefix.
_tables: "Dict[str, OrderedDict]" = {}
# prefixes whose report target has already been registered (register exactly once).
_registered: "set[str]" = set()


def add_result(
    prefix: str,
    *,
    search_type: str,
    model: str,
    variant: str,
    epoch: int,
    wers: Optional[Dict[str, tk.Variable]] = None,
    pers: Optional[Dict[str, tk.Variable]] = None,
):
    """
    Add (or extend) one result row for the summary table at ``prefix``.

    :param prefix: top-level experiment prefix (the table lives at ``prefix + "/summary.report"``).
    :param search_type: ``"lexicon"`` or ``"lexicon-free"`` -- groups the rows into sections.
    :param model: AM model name (the per-model subdir of the detailed reports).
    :param variant: search-config description, e.g. ``"4gram-word-lm"`` or
        ``"neural:phon_trafo12x512_3ep lm0p50 beam32 st14p00"``.
    :param epoch: AM checkpoint epoch.
    :param wers: mapping ``dataset -> WER tk.Variable`` (omit for lexicon-free search).
    :param pers: mapping ``dataset -> PER tk.Variable`` (the only metric for lexicon-free search;
        an auxiliary metric for lexicon-constrained search).
    """
    table = _tables.setdefault(prefix, OrderedDict())
    row_key = (search_type, model, variant, epoch)
    row = table.setdefault(
        row_key,
        {
            "search_type": search_type,
            "model": model,
            "variant": variant,
            "epoch": epoch,
            "metrics": {},  # metric -> {dataset -> tk.Variable}
        },
    )
    if wers:
        row["metrics"].setdefault("WER", {}).update(wers)
    if pers:
        row["metrics"].setdefault("PER", {}).update(pers)
    _ensure_registered(prefix)


def add_duration_result(
    prefix: str,
    *,
    model: str,
    epoch: int,
    durations: Dict[str, Dict[str, tk.Variable]],
):
    """
    Add phoneme / silence **duration** rows to the summary table at ``prefix``.

    Durations are rendered as their own ``duration-analysis`` section -- one row per duration
    *category* (e.g. ``phon_eow`` / ``sil_leading``), NOT as extra WER columns -- with the average
    label duration in ms per dataset. The category is the row's ``variant``, so it lines up under the
    same ``model / variant / epoch`` label column as the recognition rows.

    :param durations: ``{category -> {dataset -> ms tk.Variable}}`` (e.g. from
        :class:`...pipeline.PhonemeDurationStatsJob.out_vars`).
    """
    for category, per_dataset in durations.items():
        table = _tables.setdefault(prefix, OrderedDict())
        row_key = ("duration", model, category, epoch)
        row = table.setdefault(
            row_key,
            {
                "search_type": "duration",
                "model": model,
                "variant": category,
                "epoch": epoch,
                "metrics": {},
            },
        )
        row["metrics"].setdefault("ms", {}).update(per_dataset)
    _ensure_registered(prefix)


def _ensure_registered(prefix: str):
    if prefix in _registered:
        return
    _registered.add(prefix)
    table = _tables[prefix]
    # `partial(_format_table, prefix, table)` closes over the SAME dict object that add_result keeps
    # mutating, so rows added after registration are still rendered. `required=table` lets
    # sisyphus discover the WER Variables (extract_paths walks the nested dict).
    tk.register_report(
        prefix + "/summary.report",
        partial(_format_table, prefix, table),
        required=table,
        update_frequency=120,
    )


def _resolve(var) -> Optional[float]:
    """Resolve a WER Variable to a float, or None if not finished yet."""
    try:
        if isinstance(var, DelayedBase):
            if hasattr(var, "is_set") and not var.is_set():
                return None
            var = var.get()
        return float(var)
    except Exception:
        return None


def _present_datasets(rows: List[dict]) -> List[str]:
    present = {ds for row in rows for metric in row["metrics"].values() for ds in metric}
    cols = [ds for ds in _DATASET_ORDER if ds in present]
    # append any unexpected dataset names deterministically
    cols += sorted(present - set(cols))
    return cols


def _present_metrics(rows: List[dict]) -> List[str]:
    present = {m for row in rows for m in row["metrics"]}
    cols = [m for m in _METRIC_ORDER if m in present]
    cols += sorted(present - set(cols))
    return cols


def _format_section(title: str, rows: List[dict]) -> str:
    if not rows:
        return ""
    datasets = _present_datasets(rows)
    metrics = _present_metrics(rows)
    # Columns are (metric, dataset) pairs, e.g. WER dev-clean, WER dev-other, PER dev-clean, ...
    columns = [(m, ds) for m in metrics for ds in datasets]
    # sort rows for stable output: by model, then variant, then epoch
    rows = sorted(rows, key=lambda r: (r["model"], r["variant"], r["epoch"]))

    col_labels = [f"{ds} {m}" for (m, ds) in columns]
    label_w = max([len("model / variant / epoch")] + [len(f'{r["model"]} | {r["variant"]} | ep{r["epoch"]}') for r in rows])
    col_w = max([9] + [len(lbl) for lbl in col_labels])

    lines = [f"### {title}  ({len(rows)} runs)"]
    header = "model / variant / epoch".ljust(label_w) + "  " + "  ".join(lbl.rjust(col_w) for lbl in col_labels)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        label = f'{r["model"]} | {r["variant"]} | ep{r["epoch"]}'.ljust(label_w)
        cells = []
        for (m, ds) in columns:
            val = _resolve(r["metrics"].get(m, {}).get(ds))
            cells.append(("--" if val is None else f"{val:.2f}").rjust(col_w))
        lines.append(label + "  " + "  ".join(cells))
    return "\n".join(lines)


def _format_table(prefix: str, table: "OrderedDict") -> str:
    rows = list(table.values())
    n_done = sum(1 for r in rows for metric in r["metrics"].values() for v in metric.values() if _resolve(v) is not None)
    n_total = sum(len(metric) for r in rows for metric in r["metrics"].values())

    # Title from the top-level experiment dir, e.g. "ls960_ctc_eow_phon" / "ls960_phmm_eow_phon".
    experiment = prefix.rstrip("/").split("/")[-1]
    out = [
        "=" * 80,
        f"{experiment} -- recognition summary (lexicon search + lexicon-free search)",
        f"live report; '--' = not finished yet; {n_done}/{n_total} scores available "
        f"(WER = word error rate; PER = phoneme error rate over EOW-phoneme labels; "
        f"ms = average label duration in milliseconds)",
        "=" * 80,
        "",
    ]
    for search_type, title in [
        ("lexicon", "lexicon-search"),
        ("lexicon-free", "lexicon-free-search"),
        ("duration", "duration-analysis"),
    ]:
        section = _format_section(title, [r for r in rows if r["search_type"] == search_type])
        if section:
            out.append(section)
            out.append("")
    return "\n".join(out) + "\n"
