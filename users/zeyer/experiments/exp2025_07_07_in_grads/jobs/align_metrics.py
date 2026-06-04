"""Shared word-boundary alignment metrics.

Used by both the grad-align WBE job and the forced-align baseline metric job
so every alignment reports the same richer set, not just mean WBE:
- mean WBE (as before; macro mean over utterances of the per-utt mean),
- start / end mean-absolute-error,
- accuracy at tolerance collars (fraction of boundaries within 20/50/100 ms;
  micro over all boundaries -- the standard forced-alignment metric),
- edge vs interior WBE (first+last word, dominated by leading/trailing silence,
  vs the interior words) -- the WBE mean is edge-dominated, so this separates them.

All predictions are 1:1 word-aligned to the reference (same word count), so a
boundary's prediction and reference are paired and accuracy needs no detection
matching. (Hypothesis-mode alignment, where words differ, would need F1@collar;
that is a separate path.)
"""

from typing import Dict, List, Sequence, Tuple


def per_utt_boundary_errors(
    pred_word_start_ends: Sequence[Tuple[float, float]],
    ref_word_start_ends: Sequence[Tuple[float, float]],
) -> Dict[str, List[float]]:
    """Per-word absolute boundary errors (seconds) for one utterance.

    :param pred_word_start_ends: per-word (start, end) seconds, predicted.
    :param ref_word_start_ends: per-word (start, end) seconds, reference. Same length.
    :return: dict with per-word lists ``start_abs``, ``end_abs``, ``wbe``
        (``wbe[w] = 0.5*(start_abs[w]+end_abs[w])``).
    """
    n = len(ref_word_start_ends)
    assert n == len(pred_word_start_ends) and n > 0, f"{len(pred_word_start_ends)=} {n=}"
    start_abs = [abs(p[0] - r[0]) for p, r in zip(pred_word_start_ends, ref_word_start_ends)]
    end_abs = [abs(p[1] - r[1]) for p, r in zip(pred_word_start_ends, ref_word_start_ends)]
    wbe = [0.5 * (s + e) for s, e in zip(start_abs, end_abs)]
    return {"start_abs": start_abs, "end_abs": end_abs, "wbe": wbe}


def aggregate_corpus(
    utt_errors: Sequence[Dict[str, List[float]]],
    collars_sec: Sequence[float] = (0.02, 0.05, 0.1),
) -> Dict[str, float]:
    """Aggregate per-utterance errors to corpus-level scalar metrics.

    WBE / start-MAE / end-MAE and edge/interior are macro means over utterances
    (matches the original WBE convention).
    Collar accuracies are micro over all boundaries (the standard).
    """
    n_utt = len(utt_errors)
    assert n_utt > 0
    # macro: per-utt mean, then mean over utts.
    utt_wbe = [sum(u["wbe"]) / len(u["wbe"]) for u in utt_errors]
    utt_start = [sum(u["start_abs"]) / len(u["start_abs"]) for u in utt_errors]
    utt_end = [sum(u["end_abs"]) / len(u["end_abs"]) for u in utt_errors]
    # edge = first + last word; interior = the rest.
    utt_edge, utt_interior = [], []
    for u in utt_errors:
        w = u["wbe"]
        m = len(w)
        edge = [w[0]] + ([w[-1]] if m > 1 else [])
        interior = w[1:-1]
        utt_edge.append(sum(edge) / len(edge))
        if interior:
            utt_interior.append(sum(interior) / len(interior))
    # micro: all boundary errors across the corpus for the collar accuracies.
    all_bnd = [b for u in utt_errors for b in (u["start_abs"] + u["end_abs"])]

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    out = {
        "wbe": mean(utt_wbe),
        "start_mae": mean(utt_start),
        "end_mae": mean(utt_end),
        "edge_wbe": mean(utt_edge),
        "interior_wbe": mean(utt_interior),
        "n_utt": float(n_utt),
        "n_words": float(sum(len(u["wbe"]) for u in utt_errors)),
    }
    for c in collars_sec:
        out[f"acc_{int(round(c * 1000))}ms"] = mean([1.0 if b <= c else 0.0 for b in all_bnd])
    return out
