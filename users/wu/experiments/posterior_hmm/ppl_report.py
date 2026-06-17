"""
Live summary report for phoneme- and word-LM perplexity.

Mirrors the recognition `results.py` live-report convention, but for LM perplexity
rather than WER/PER. Rows are (eval-set, LM); columns are:

  * phoneme PPL  (+eos / -eos): the standard per-phoneme-token perplexity of the two
    PHONEME LMs (neural Transformer + count n-gram). N/A for the word LM.
  * word PPL     (+eos / -eos): per-WORD perplexity. For the word LM this is its native
    perplexity; for the phoneme LMs it is the WORD-EQUIVALENT perplexity
    `exp(total_neg_log_prob / N_words)` -- the phoneme LM's total NLL over the text
    normalized by the word count, so all three LMs are comparable per word.

All numbers are reported WITHOUT unknowns (the phoneme LMs are closed-vocab so this is a
no-op for them; the word LM uses RASR's `*_without_unknowns` variants). "+eos" includes the
per-sentence `</s>` event in the NLL (and, for the per-phoneme/per-word native PPL, in the
normalizing token count); "-eos" drops it. See `ppl.py::evaluate_lm_perplexities`.
"""
from collections import OrderedDict
from functools import partial
from typing import Dict, Optional

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

# Eval sets shown in this order (only those present are printed).
_EVAL_ORDER = ["dev-clean", "dev-other", "test-clean", "test-other"]
# (key, header) for the four metric columns, in display order.
_METRIC_COLUMNS = [
    ("phon_ppl_eos", "phonPPL +eos"),
    ("phon_ppl_noeos", "phonPPL -eos"),
    ("word_ppl_eos", "wordPPL +eos"),
    ("word_ppl_noeos", "wordPPL -eos"),
]

# eval_set -> OrderedDict(lm_label -> row dict)
_tables: "Dict[str, OrderedDict]" = {}
# per eval_set -> {stat_name -> tk.Variable} (N_words, N_phon, N_sent)
_stats: "Dict[str, Dict[str, tk.Variable]]" = {}
_registered = False
_REPORT_PATH: Optional[str] = None


def set_report_path(path: str):
    """Set where the single combined perplexity report is registered."""
    global _REPORT_PATH
    _REPORT_PATH = path


def add_eval_set_stats(eval_set: str, *, n_words=None, n_phon=None, n_sent=None):
    """Register the per-eval-set token/word/sentence counts (shown in the section header)."""
    d = _stats.setdefault(eval_set, {})
    if n_words is not None:
        d["N_words"] = n_words
    if n_phon is not None:
        d["N_phon"] = n_phon
    if n_sent is not None:
        d["N_sent"] = n_sent
    _ensure_registered()


def add_ppl_result(eval_set: str, lm_label: str, *, order: int, metrics: Dict[str, tk.Variable]):
    """
    Add one perplexity row.

    :param eval_set: e.g. "dev-other" / "test-other".
    :param lm_label: e.g. "neural trafo 12x512" / "count 8-gram" / "official word 4-gram".
    :param order: sort order within the eval-set section (lower = first).
    :param metrics: subset of {phon_ppl_eos, phon_ppl_noeos, word_ppl_eos, word_ppl_noeos}
        -> tk.Variable. Missing keys render as "n/a".
    """
    table = _tables.setdefault(eval_set, OrderedDict())
    table[lm_label] = {"label": lm_label, "order": order, "metrics": dict(metrics)}
    _ensure_registered()


def _ensure_registered():
    global _registered
    if _registered or _REPORT_PATH is None:
        return
    _registered = True
    tk.register_report(
        _REPORT_PATH,
        partial(_format_report),
        required={"tables": _tables, "stats": _stats},
        update_frequency=120,
    )


def _resolve(var) -> Optional[float]:
    try:
        if isinstance(var, DelayedBase):
            if hasattr(var, "is_set") and not var.is_set():
                return None
            var = var.get()
        return float(var)
    except Exception:
        return None


def _fmt_stat(var) -> str:
    v = _resolve(var)
    return "?" if v is None else f"{int(round(v))}"


def _format_section(eval_set: str, table: "OrderedDict") -> str:
    rows = sorted(table.values(), key=lambda r: (r["order"], r["label"]))
    stats = _stats.get(eval_set, {})
    nw, nph, nse = stats.get("N_words"), stats.get("N_phon"), stats.get("N_sent")
    exponent = ""
    rw, rp = _resolve(nw), _resolve(nph)
    if rw and rp:
        exponent = f"; phonemes/word={rp / rw:.2f}"
    head = (
        f"### {eval_set}  (N_words={_fmt_stat(nw)}, N_phon={_fmt_stat(nph)}, "
        f"N_sent={_fmt_stat(nse)}{exponent})"
    )

    col_headers = [h for (_, h) in _METRIC_COLUMNS]
    label_w = max([len("LM")] + [len(r["label"]) for r in rows])
    col_w = max([12] + [len(h) for h in col_headers])

    lines = [head]
    header = "LM".ljust(label_w) + "  " + "  ".join(h.rjust(col_w) for h in col_headers)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        cells = []
        for (key, _) in _METRIC_COLUMNS:
            if key not in r["metrics"]:
                cells.append("n/a".rjust(col_w))
            else:
                v = _resolve(r["metrics"][key])
                cells.append(("--" if v is None else f"{v:.2f}").rjust(col_w))
        lines.append(r["label"].ljust(label_w) + "  " + "  ".join(cells))
    return "\n".join(lines)


def _format_report(*args) -> str:
    out = [
        "=" * 88,
        "phoneme & word LM perplexity  (eval on transcripts; WITHOUT unknowns)",
        "live report; '--' = not finished yet; 'n/a' = metric not defined for this LM",
        "  phonPPL = per-phoneme-token PPL (phoneme LMs only); wordPPL = per-word PPL",
        "  (word-equivalent = exp(total_neg_log_prob / N_words) for the phoneme LMs)",
        "  +eos includes the per-sentence </s> event; -eos drops it",
        "  held-out: test-other is fully unseen; dev-other is the neural LM's CV set",
        "=" * 88,
        "",
    ]
    present = [es for es in _EVAL_ORDER if es in _tables]
    present += sorted(set(_tables) - set(present))
    for eval_set in present:
        out.append(_format_section(eval_set, _tables[eval_set]))
        out.append("")
    return "\n".join(out) + "\n"
