from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional, Any
import gzip
import json
import re
import ast
import statistics

import sisyphus
from sisyphus import tk, Job


# ---------- Utilities ----------

def _smart_open(path: str, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)  # keep apostrophes by default

def normalize_text(
    s: str,
    lowercase: bool = True,
    remove_punct: bool = False,
    strip_extra_spaces: bool = True,
) -> str:
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = _PUNCT_RE.sub(" ", s)
    if strip_extra_spaces:
        s = re.sub(r"\s+", " ", s).strip()
    return s

def word_tokenize(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return s.split()


def edit_distance_sdi(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance and count S, D, I along an optimal path.
    Returns (distance, S, D, I), where distance == S + D + I.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, 0, m
    if m == 0:
        return n, 0, n, 0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            hj = hyp[j - 1]
            cost_sub = 0 if ri == hj else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,            # deletion
                dp[i][j - 1] + 1,            # insertion
                dp[i - 1][j - 1] + cost_sub  # substitution/match
            )

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                if ref[i - 1] != hyp[j - 1]:
                    S += 1
                i -= 1
                j -= 1
            else:
                if i > 0:
                    D += 1
                    i -= 1
                elif j > 0:
                    I += 1
                    j -= 1
    dist = S + D + I
    return dist, S, D, I


def _load_py_dict(path: tk.Path) -> Dict[str, List[Tuple[float, str]]]:
    with _smart_open(str(path), "rt") as f:
        data_str = f.read()
    try:
        data = ast.literal_eval(data_str)
    except Exception:
        data = eval(data_str, {"nan": float("nan"), "inf": float("inf")})
    if not isinstance(data, dict):
        raise ValueError(f"File {path} does not contain a dict.")
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError(f"Key {k!r} is not a string.")
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"Value for key {k!r} must be list/tuple of (score, text).")
    return data


def _coerce_ref_variants(v: Any) -> List[Tuple[float, str]]:
    if isinstance(v, str):
        return [(0.0, v)]
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return [(0.0, "")]
        out = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], str):
                out.append((float(item[0]), item[1]))
            elif isinstance(item, str):
                out.append((0.0, item))
            else:
                raise ValueError(f"Unsupported ref entry format: {item!r}")
        return out
    raise ValueError(f"Unsupported ref value type: {type(v)}")


def _full_hist(values: List[int]) -> Dict[str, int]:
    d: Dict[int, int] = {}
    for v in values:
        d[v] = d.get(v, 0) + 1
    # stringify keys for JSON stability
    return {str(k): v for k, v in sorted(d.items(), key=lambda x: int(x[0]))}

def _binned_hist(values: List[int]) -> Dict[str, int]:
    """
    Buckets: 0, 1, 2, 3–5, 6–10, 11+
    """
    bins = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3-5": 0,
        "6-10": 0,
        "11+": 0,
    }
    for v in values:
        if v <= 0:
            bins["0"] += 1
        elif v == 1:
            bins["1"] += 1
        elif v == 2:
            bins["2"] += 1
        elif 3 <= v <= 5:
            bins["3-5"] += 1
        elif 6 <= v <= 10:
            bins["6-10"] += 1
        else:
            bins["11+"] += 1
    return bins

def _topk_coverage(values: List[int], ks=(0, 1, 2, 3, 5, 10)) -> Dict[str, float]:
    """
    Fraction of items with rank_gap <= k. Keys are strings for JSON.
    """
    if not values:
        return {str(k): None for k in ks}
    n = len(values)
    out = {}
    for k in ks:
        cnt = sum(1 for v in values if v <= k)
        out[str(k)] = cnt / n
    return out
# ---------- The Job ----------

class OracleWerJob(Job):
    """
    Compute oracle WER (with S/D/I) over an N-best list against (possibly multiple) reference variants.

    Inputs:
      hyp_path: { seq_tag: [(score, hyp_text), ...], ... }
      ref_path: { seq_tag: [(score, ref_text), ...] } or { seq_tag: [ref_text, ...] } or { seq_tag: ref_text }

    Outputs:
      - oracle_wer.json : overall stats + score-gap and rank-gap stats
      - per_utt.csv     : seq_tag, edits, S, D, I, ref_len, wer,
                          best_hyp_score, best_hyp_text, best_ref_text,
                          best_score_hyp_score, best_score_hyp_text, abs_score_gap, is_same_hyp,
                          score_rank_oracle, rank_gap
      - oracle_best.py  : dict like input but with only the oracle 1-best hyp per seq_tag
    """

    def __init__(
        self,
        hyp_path: tk.Path,
        ref_path: tk.Path,
        *,
        lowercase: bool = True,
        remove_punct: bool = False,
        score_higher_is_better: bool = True,
        output_gzip: bool = True,
        name: Optional[str] = None,
        version: int = 1,
    ):
        self.hyp_path = hyp_path
        self.ref_path = ref_path
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.score_higher_is_better = score_higher_is_better
        self.output_gzip = output_gzip

        suffix = ".gz" if output_gzip else ""
        base = "oracle" if name is None else f"oracle_{name}"
        self.out_report = self.output_path(f"{base}_wer.json")
        self.out_per_utt = self.output_path(f"{base}_per_utt.csv{suffix}")
        self.out_best_nbest = self.output_path(f"{base}_best.py{suffix}")

    def tasks(self):
        yield sisyphus.Task("run", mini_task=True)


    # ---helpers---
    def _is_better_score(self, a: float, b: float) -> bool:
        return (a > b) if self.score_higher_is_better else (a < b)

    def _best_by_score(self, hyps: List[Tuple[float, str]]) -> Tuple[float, str]:
        assert hyps, "empty hyps"
        best_score, best_text = hyps[0]
        for s, t in hyps[1:]:
            if self._is_better_score(s, best_score):
                best_score, best_text = s, t
        return best_score, best_text

    def _rank_of_hyp(self, hyps: List[Tuple[float, str]], target: Tuple[float, str]) -> int:
        """
        Rank hypotheses by score (1 = best). Ties are broken by original index to ensure stability.
        'target' is matched by (score, text).
        """
        # Prepare sortable tuples: (key, idx, score, text)
        sortable = []
        for idx, (s, t) in enumerate(hyps):
            key = (-s if self.score_higher_is_better else s)
            sortable.append((key, idx, s, t))
        sortable.sort()  # ascending by key -> best first
        # Find first position where both (score,text) match
        for pos, (_key, _idx, s, t) in enumerate(sortable, start=1):
            if s == target[0] and t == target[1]:
                return pos
        # If not found (shouldn't happen), return len+1 as sentinel
        return len(hyps) + 1


    def run(self):
        hyp_dict = _load_py_dict(self.hyp_path)
        ref_dict_raw = _load_py_dict(self.ref_path)

        hyp_keys = set(hyp_dict.keys())
        ref_keys = set(ref_dict_raw.keys())
        if hyp_keys != ref_keys:
            missing = sorted(ref_keys - hyp_keys)
            extra = sorted(hyp_keys - ref_keys)
            raise AssertionError(
                "Hyp/Ref key sets differ. "
                f"Missing_in_hyp:{len(missing)} ({missing[:5]}...), "
                f"Extra_in_hyp:{len(extra)} ({extra[:5]}...)."
            )

        total_edits = 0
        total_ref_words = 0
        total_S = total_D = total_I = 0
        num_utts = 0

        # gap aggregates
        abs_gaps: List[float] = []
        rank_gaps: List[int] = []
        same_hyp_count = 0

        with _smart_open(str(self.out_per_utt), "wt") as f_csv:
            f_csv.write(
                "seq_tag,edits,S,D,I,ref_len,wer,"
                "best_hyp_score,best_hyp_text,best_ref_text,"
                "best_score_hyp_score,best_score_hyp_text,abs_score_gap,is_same_hyp,"
                "score_rank_oracle,rank_gap\n"
            )

            best_nbest_dict: Dict[str, List[Tuple[float, str]]] = {}

            for seq_tag in sorted(hyp_dict.keys()):
                hyps: List[Tuple[float, str]] = hyp_dict[seq_tag]
                refs_variants = _coerce_ref_variants(ref_dict_raw[seq_tag])

                # Score-best hyp (if any)
                score_best_score = None
                score_best_text = None
                if hyps:
                    score_best_score, score_best_text = self._best_by_score(hyps)

                if not hyps:
                    # No hyp case
                    best_score = float("inf") if not self.score_higher_is_better else float("-inf")
                    best_hyp = ""
                    best_ref = refs_variants[0][1] if refs_variants else ""
                    ref_tokens = word_tokenize(normalize_text(best_ref, self.lowercase, self.remove_punct))
                    S = 0
                    D = len(ref_tokens)
                    I = 0
                    edits = D
                    ref_len = len(ref_tokens)
                    wer = 0.0 if ref_len == 0 else edits / ref_len
                    abs_gap_out = ""  # undefined
                    is_same = 0
                    oracle_rank = ""
                    rank_gap = ""
                else:
                    # Oracle-WER selection
                    best_pair = None  # (dist,S,D,I,ref_len,wer,hyp_score,hyp_text,ref_text)
                    for hyp_score, hyp_text in hyps:
                        hyp_norm = normalize_text(hyp_text, self.lowercase, self.remove_punct)
                        hyp_tok = word_tokenize(hyp_norm)
                        for _, ref_text in refs_variants:
                            ref_norm = normalize_text(ref_text, self.lowercase, self.remove_punct)
                            ref_tok = word_tokenize(ref_norm)
                            dist, s_cnt, d_cnt, i_cnt = edit_distance_sdi(ref_tok, hyp_tok)
                            ref_len = len(ref_tok)
                            wer = 0.0 if ref_len == 0 else dist / ref_len

                            if best_pair is None:
                                best_pair = (dist, s_cnt, d_cnt, i_cnt, ref_len, wer, hyp_score, hyp_text, ref_text)
                            else:
                                b_dist, b_s, b_d, b_i, b_ref_len, b_wer, b_score, b_hyp, b_ref = best_pair
                                better = False
                                if wer < b_wer - 1e-12:
                                    better = True
                                elif abs(wer - b_wer) <= 1e-12 and self._is_better_score(hyp_score, b_score):
                                    better = True
                                elif abs(wer - b_wer) <= 1e-12 and (not self._is_better_score(b_score, hyp_score)) and len(hyp_text) < len(b_hyp):
                                    better = True
                                if better:
                                    best_pair = (dist, s_cnt, d_cnt, i_cnt, ref_len, wer, hyp_score, hyp_text, ref_text)

                    edits, S, D, I, ref_len, wer, best_score, best_hyp, best_ref = best_pair  # type: ignore

                    # Score/Rank gaps to score-best hyp
                    abs_gap = abs(best_score - score_best_score)  # type: ignore
                    is_same = 1 if best_hyp == score_best_text else 0
                    oracle_rank = self._rank_of_hyp(hyps, (best_score, best_hyp))
                    rank_gap = oracle_rank - 1

                    abs_gaps.append(abs_gap)
                    rank_gaps.append(rank_gap)
                    same_hyp_count += is_same

                    abs_gap_out = abs_gap  # for CSV

                # Accumulate WER totals
                total_edits += edits
                total_ref_words += ref_len
                total_S += S
                total_D += D
                total_I += I
                num_utts += 1

                def _csv_escape(s: str) -> str:
                    s = s.replace('"', '""')
                    return f'"{s}"'

                f_csv.write(
                    f"{_csv_escape(seq_tag)},{edits},{S},{D},{I},{ref_len},"
                    f"{0.0 if ref_len==0 else edits/ref_len:.6f},"
                    f"{best_score},{_csv_escape(best_hyp)},{_csv_escape(best_ref)},"
                    f"{'' if score_best_score is None else score_best_score},"
                    f"{_csv_escape('' if score_best_text is None else score_best_text)},"
                    f"{abs_gap_out},{is_same},"
                    f"{oracle_rank},{rank_gap}\n"
                )

                best_nbest_dict[seq_tag] = [(best_score, best_hyp)]

        overall_wer = 0.0 if total_ref_words == 0 else total_edits / total_ref_words

        # Aggregate gaps
        if abs_gaps:
            avg_abs_gap = float(statistics.mean(abs_gaps))
            median_abs_gap = float(statistics.median(abs_gaps))
            same_rate = same_hyp_count / len(abs_gaps)
        else:
            avg_abs_gap = None
            median_abs_gap = None
            same_rate = None

        if rank_gaps:
            avg_rank_gap = float(statistics.mean(rank_gaps))
            median_rank_gap = float(statistics.median(rank_gaps))
        else:
            avg_rank_gap = None
            median_rank_gap = None
        # Build rank-gap histograms and coverage
        rank_gap_full_hist = _full_hist(rank_gaps) if rank_gaps else None
        rank_gap_histogram = _binned_hist(rank_gaps) if rank_gaps else None
        rank_gap_topk_coverage = _topk_coverage(rank_gaps) if rank_gaps else None

        report = {
            "overall_wer": overall_wer,
            "total_edits": int(total_edits),
            "total_ref_words": int(total_ref_words),
            "num_utts": int(num_utts),
            "total_S": int(total_S),
            "total_D": int(total_D),
            "total_I": int(total_I),
            "S_rate": (0.0 if total_ref_words == 0 else total_S / total_ref_words),
            "D_rate": (0.0 if total_ref_words == 0 else total_D / total_ref_words),

            # Score-gap stats
            "avg_abs_score_gap": avg_abs_gap,
            "median_abs_score_gap": median_abs_gap,
            "same_hyp_rate": same_rate,
            #"count_with_gap": len(abs_gaps),

            # Rank-gap stats
            "avg_rank_gap": avg_rank_gap,
            "median_rank_gap": median_rank_gap,
            #"count_with_rank": len(rank_gaps),

            # NEW: histogram summaries
            "rank_gap_histogram": rank_gap_histogram,     # compact buckets
            "rank_gap_full_hist": rank_gap_full_hist,     # exact value -> count
            "rank_gap_topk_coverage": rank_gap_topk_coverage,  # e.g., {"0":0.72,"1":0.86,...}

            "lowercase": self.lowercase,
            "remove_punct": self.remove_punct,
            "score_higher_is_better": self.score_higher_is_better,
        }

        with open(self.out_report, "w") as f_json:
            json.dump(report, f_json, ensure_ascii=False, indent=2)

        #text = repr(best_nbest_dict)
        def write_py_dict(d, io):
            io.write("{\n")
            for seq_tag, v in d.items():
                io.write(f"{seq_tag!r}: [\n")
                for score, hyp in v:
                    io.write(f"({score!r}, {hyp!r}),\n")
                io.write("],\n")
            io.write("}\n")

        if str(self.out_best_nbest).endswith(".gz"):
            with gzip.open(self.out_best_nbest, "wt") as f_out:
                write_py_dict(best_nbest_dict, f_out)
        else:
            with open(self.out_best_nbest, "w") as f_out:
                write_py_dict(best_nbest_dict, f_out)
