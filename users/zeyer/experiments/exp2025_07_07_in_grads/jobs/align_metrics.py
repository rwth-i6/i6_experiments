"""Shared word-boundary alignment metrics.

Used by both the grad-align WBE job and the forced-align baseline metric job
so every alignment reports the same richer set, not just mean WBE:
- mean WBE (aggregated over all words (micro) or over utterances (macro); see ``aggregate_corpus``),
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
    # signed (pred - ref): + = predicted boundary later than reference (delayed emission).
    start_signed = [p[0] - r[0] for p, r in zip(pred_word_start_ends, ref_word_start_ends)]
    end_signed = [p[1] - r[1] for p, r in zip(pred_word_start_ends, ref_word_start_ends)]
    wbe = [0.5 * (s + e) for s, e in zip(start_abs, end_abs)]
    return {
        "start_abs": start_abs,
        "end_abs": end_abs,
        "wbe": wbe,
        "start_signed": start_signed,
        "end_signed": end_signed,
    }


def aggregate_corpus(
    utt_errors: Sequence[Dict[str, List[float]]],
    collars_sec: Sequence[float] = (0.02, 0.05, 0.1),
    aggregation: str = "macro",
) -> Dict[str, float]:
    """Aggregate per-utterance errors to corpus-level scalar metrics.

    :param aggregation: how to reduce the per-word errors (WBE / start-MAE / end-MAE / edge /
        interior / the signed offsets) to a single corpus number:

        - ``"micro"``: mean over ALL words in the corpus, each word weighted equally -- the standard
          "average over the words".
        - ``"macro"``: mean per utterance, then mean over utterances (weights each utterance equally;
          the original convention). Kept as a selectable option.

        Collar accuracies are ALWAYS micro over all boundaries (the standard), independent of this.
    """
    n_utt = len(utt_errors)
    assert n_utt > 0
    assert aggregation in ("macro", "micro"), aggregation

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def agg(per_utt_lists):
        # per_utt_lists: one list of per-word values per utterance.
        # micro = pool every word and average; macro = per-utt mean, then mean over utts.
        if aggregation == "micro":
            return mean([v for lst in per_utt_lists for v in lst])
        return mean([mean(lst) for lst in per_utt_lists if lst])

    wbe_l = [u["wbe"] for u in utt_errors]
    start_l = [u["start_abs"] for u in utt_errors]
    end_l = [u["end_abs"] for u in utt_errors]
    start_sg_l = [u["start_signed"] for u in utt_errors]
    end_sg_l = [u["end_signed"] for u in utt_errors]
    # per-word |word-center offset| (= |0.5*(start+end) signed|): width-robust accuracy.
    center_abs_l = [[abs(0.5 * (s + e)) for s, e in zip(u["start_signed"], u["end_signed"])] for u in utt_errors]
    # edge = first + last word; interior = the rest.
    edge_l = [([w[0]] + ([w[-1]] if len(w) > 1 else [])) for w in wbe_l]
    interior_l = [w[1:-1] for w in wbe_l]
    # micro: all boundary errors across the corpus for the collar accuracies (always micro).
    all_bnd = [b for u in utt_errors for b in (u["start_abs"] + u["end_abs"])]

    start_signed_mean = agg(start_sg_l)
    end_signed_mean = agg(end_sg_l)
    out = {
        "wbe": agg(wbe_l),
        "start_mae": agg(start_l),
        "end_mae": agg(end_l),
        "start_signed_mean": start_signed_mean,
        "end_signed_mean": end_signed_mean,
        "signed_mean": agg(start_sg_l + end_sg_l),
        # center (midpoint) signed offset, word-width signed error, |center| (width-robust accuracy):
        "center_offset": (start_signed_mean + end_signed_mean) / 2,
        "width_signed_err": end_signed_mean - start_signed_mean,
        "center_abs": agg(center_abs_l),
        "edge_wbe": agg(edge_l),
        "interior_wbe": agg(interior_l),
        "n_utt": float(n_utt),
        "n_words": float(sum(len(w) for w in wbe_l)),
    }
    for c in collars_sec:
        out[f"acc_{int(round(c * 1000))}ms"] = mean([1.0 if b <= c else 0.0 for b in all_bnd])
    return out


def collapse_phones_to_words(
    pred_phone_start_ends: Sequence[Tuple[float, float]],
    phone_starts_samples: Sequence[float],
    phone_stops_samples: Sequence[float],
    word_starts_samples: Sequence[float],
    word_stops_samples: Sequence[float],
    time_scale: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Collapse per-phone PREDICTED boundaries to per-word boundaries.

    A phone belongs to the word whose GT span contains the phone's GT midpoint
    (inter-word silences h#/pau fall in no word and are dropped). The word's
    predicted boundary = (predicted start of its first phone, predicted end of
    its last phone); its reference boundary = the GT word span.

    :param pred_phone_start_ends: per-phone (start, end) seconds, PREDICTED,
        same order/length as the GT phone segments.
    :param phone_starts_samples / phone_stops_samples: GT phone segment bounds (samples).
    :param word_starts_samples / word_stops_samples: GT word segment bounds (samples).
    :param time_scale: samples -> seconds factor (``dataset_offset_factors / samplerate``).
    :return: ``(pred_word_start_ends, ref_word_start_ends)``, 1:1 per word (seconds).
    """
    n_ph = len(pred_phone_start_ends)
    assert n_ph == len(phone_starts_samples) == len(phone_stops_samples), (
        f"{n_ph=} {len(phone_starts_samples)=} {len(phone_stops_samples)=}"
    )
    phone_mid = [0.5 * (s + e) for s, e in zip(phone_starts_samples, phone_stops_samples)]
    pred_word, ref_word = [], []
    for ws, we in zip(word_starts_samples, word_stops_samples):
        idxs = [i for i, m in enumerate(phone_mid) if ws <= m < we]
        if not idxs:  # fallback: any GT overlap
            idxs = [i for i in range(n_ph) if phone_starts_samples[i] < we and phone_stops_samples[i] > ws]
        if not idxs:  # degenerate (e.g. zero-width word span in TIMIT): nearest phone by GT midpoint
            wmid = 0.5 * (ws + we)
            idxs = [min(range(n_ph), key=lambda i: abs(phone_mid[i] - wmid))]
        pred_word.append((pred_phone_start_ends[idxs[0]][0], pred_phone_start_ends[idxs[-1]][1]))
        ref_word.append((ws * time_scale, we * time_scale))
    return pred_word, ref_word


def levenshtein_word_matches(hyp_words: Sequence[str], ref_words: Sequence[str]) -> List[Tuple[int, int]]:
    """Levenshtein-align two word sequences; return the identity matches.

    Standard edit-distance DP (match 0, sub/ins/del 1), backtraced preferring
    matches. Only pairs with equal words are returned
    (substitutions carry no usable boundary correspondence).

    :return: list of (hyp_idx, ref_idx), strictly increasing in both.
    """
    n, m = len(hyp_words), len(ref_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        hw = hyp_words[i - 1]
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (0 if hw == ref_words[j - 1] else 1)
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    matches = []
    i, j = n, m
    while i > 0 and j > 0:
        if hyp_words[i - 1] == ref_words[j - 1] and d[i][j] == d[i - 1][j - 1]:
            matches.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


def hyp_aggregate_corpus(
    utt_stats: Sequence[Dict[str, object]],
    collars_sec: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
    aggregation: str = "macro",
) -> Dict[str, float]:
    """Aggregate hypothesis-mode alignment stats to corpus metrics.

    Identity-gated F1@collar (WhisperX protocol): a hyp word is a TP at collar c
    iff it Levenshtein-matches a ref word by identity AND both its boundaries
    are within c of that ref word's. Precision over hyp words, recall over ref
    words. Plus the Levenshtein-matched mean WBE (Rousso protocol; aggregated per
    ``aggregation`` -- micro over matched words or macro over utterances, like
    ``aggregate_corpus``) and match/coverage rates.

    :param utt_stats: per utt: ``n_hyp``, ``n_ref``,
        ``match_errs`` (per matched word: (start_abs, end_abs) seconds).
    """
    n_utt = len(utt_stats)
    assert n_utt > 0
    n_hyp = sum(u["n_hyp"] for u in utt_stats)
    n_ref = sum(u["n_ref"] for u in utt_stats)
    all_match_errs = [e for u in utt_stats for e in u["match_errs"]]
    n_match = len(all_match_errs)

    assert aggregation in ("macro", "micro"), aggregation
    # matched WBE: micro = mean over ALL matched words; macro = per-utt mean, then mean over utts.
    if aggregation == "micro":
        matched_wbe = float(sum(0.5 * (s + e) for s, e in all_match_errs) / n_match) if n_match else float("nan")
    else:
        utt_wbe = [
            sum(0.5 * (s + e) for s, e in u["match_errs"]) / len(u["match_errs"]) for u in utt_stats if u["match_errs"]
        ]
        matched_wbe = float(sum(utt_wbe) / len(utt_wbe)) if utt_wbe else float("nan")

    out = {
        "n_utt": float(n_utt),
        "n_hyp_words": float(n_hyp),
        "n_ref_words": float(n_ref),
        "n_matched_words": float(n_match),
        "match_rate_ref": float(n_match / n_ref) if n_ref else float("nan"),
        "match_rate_hyp": float(n_match / n_hyp) if n_hyp else float("nan"),
        "matched_wbe": matched_wbe,
    }
    for c in collars_sec:
        tp = sum(1 for s, e in all_match_errs if s <= c and e <= c)
        prec = tp / n_hyp if n_hyp else float("nan")
        rec = tp / n_ref if n_ref else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ms = int(round(c * 1000))
        out[f"precision_{ms}ms"] = float(prec)
        out[f"recall_{ms}ms"] = float(rec)
        out[f"f1_{ms}ms"] = float(f1)
    return out
