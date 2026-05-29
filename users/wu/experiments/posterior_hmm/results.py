"""
Global results registry + a single combined summary table for the pHMM phon
recognition experiments.

Both the lexicon baseline (``config_02_phon_phmm``) and the lexicon-free baseline
(``config_03_phon_phmm_lexfree``) push their per-(model, epoch, variant, dataset)
WERs here via :func:`add_result`. A single ``tk.register_report`` then renders all
of them into ONE table at the top-level experiment prefix
(``.../ls960_phmm_eow_phon/summary.report``).

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
    wers: Dict[str, tk.Variable],
):
    """
    Add (or extend) one result row for the summary table at ``prefix``.

    :param prefix: top-level experiment prefix (the table lives at ``prefix + "/summary.report"``).
    :param search_type: ``"lexicon"`` or ``"lexicon-free"`` -- groups the rows into sections.
    :param model: AM model name (the per-model subdir of the detailed reports).
    :param variant: search-config description, e.g. ``"4gram-word-lm"`` or
        ``"neural:phon_trafo12x512_3ep lm0p50 beam32 st14p00"``.
    :param epoch: AM checkpoint epoch.
    :param wers: mapping ``dataset -> WER tk.Variable`` (as returned by the search helper).
    """
    table = _tables.setdefault(prefix, OrderedDict())
    row_key = (search_type, model, variant, epoch)
    row = table.setdefault(
        row_key,
        {"search_type": search_type, "model": model, "variant": variant, "epoch": epoch, "wers": {}},
    )
    row["wers"].update(wers)
    _ensure_registered(prefix)


def _ensure_registered(prefix: str):
    if prefix in _registered:
        return
    _registered.add(prefix)
    table = _tables[prefix]
    # `partial(_format_table, table)` closes over the SAME dict object that add_result keeps
    # mutating, so rows added after registration are still rendered. `required=table` lets
    # sisyphus discover the WER Variables (extract_paths walks the nested dict).
    tk.register_report(
        prefix + "/summary.report",
        partial(_format_table, table),
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
    present = {ds for row in rows for ds in row["wers"]}
    cols = [ds for ds in _DATASET_ORDER if ds in present]
    # append any unexpected dataset names deterministically
    cols += sorted(present - set(cols))
    return cols


def _format_section(title: str, rows: List[dict]) -> str:
    if not rows:
        return ""
    datasets = _present_datasets(rows)
    # sort rows for stable output: by model, then variant, then epoch
    rows = sorted(rows, key=lambda r: (r["model"], r["variant"], r["epoch"]))

    label_w = max([len("model / variant / epoch")] + [len(f'{r["model"]} | {r["variant"]} | ep{r["epoch"]}') for r in rows])
    col_w = max(9, *(len(ds) for ds in datasets))

    lines = [f"### {title}  ({len(rows)} runs)"]
    header = "model / variant / epoch".ljust(label_w) + "  " + "  ".join(ds.rjust(col_w) for ds in datasets)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        label = f'{r["model"]} | {r["variant"]} | ep{r["epoch"]}'.ljust(label_w)
        cells = []
        for ds in datasets:
            val = _resolve(r["wers"].get(ds))
            cells.append(("--" if val is None else f"{val:.2f}").rjust(col_w))
        lines.append(label + "  " + "  ".join(cells))
    return "\n".join(lines)


def _format_table(table: "OrderedDict") -> str:
    rows = list(table.values())
    n_done = sum(1 for r in rows for v in r["wers"].values() if _resolve(v) is not None)
    n_total = sum(len(r["wers"]) for r in rows)

    out = [
        "=" * 80,
        "pHMM EOW-phon -- recognition summary (lexicon search + lexicon-free search)",
        f"live report; '--' = not finished yet; {n_done}/{n_total} WERs available",
        "=" * 80,
        "",
    ]
    for search_type, title in [("lexicon", "lexicon-search"), ("lexicon-free", "lexicon-free-search")]:
        section = _format_section(title, [r for r in rows if r["search_type"] == search_type])
        if section:
            out.append(section)
            out.append("")
    return "\n".join(out) + "\n"
