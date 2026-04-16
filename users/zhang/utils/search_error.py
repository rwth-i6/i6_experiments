from sisyphus import *
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import i6_core.util as util


class ComputeSearchErrorsJob(Job):
    def __init__(
        self,
        ground_truth_out: Optional[Path],
        recog_out: Optional[Path],
        verision: Optional[int] = 4,
    ):
        self.ground_truth_out = ground_truth_out
        self.recog_out = recog_out

        self.out_search_errors = self.output_path("search_errors")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):
        #d_gt = eval(util.uopen(self.ground_truth_out, "rt").read().replace('-inf', 'float("-inf")'),{"float":float})
        d_gt = eval(util.uopen(self.ground_truth_out, "rt").read())
        d_rec = eval(util.uopen(self.recog_out, "rt").read())
        assert isinstance(d_gt, dict)  # seq_tag -> bpe string
        assert isinstance(d_rec, dict)  # seq_tag -> bpe string

        num_seqs = 0
        num_search_errors = 0
        num_unequal = 0
        sent_oov = 0
        num_oov = 0
        num_words = 0

        # for each seq tag, calculate whether we have a search error
        for seq_tag in d_gt.keys():
            num_seqs += 1
            try:
                score_ground_truth, targets_ground_truth, oov = d_gt[seq_tag][0]
            except ValueError:
                score_ground_truth, targets_ground_truth = d_gt[seq_tag][0]
                oov = 0
            num_oov += oov
            try:
                score_search, targets_search = d_rec[seq_tag][0]
            except ValueError:
                score_search, targets_search = d_rec[seq_tag]
            num_words += len(targets_ground_truth.split())

            # we count as search error if the label seqs differ and the search score is worse than the ground truth score
            is_search_error = False
            targets_search = targets_search.replace("@@ ", "").strip()
            targets_ground_truth = targets_ground_truth.replace("@@ ", "").strip()
            #targets_search = targets_search.replace("<blank>", "")
            #targets_search = " ".join(targets_search.split())
            if list(targets_ground_truth) == list(targets_search):
                assert oov == 0, "Search reached a sequence with OOV?"
                equal_label_seq = True
            else:
                num_unequal += 1
                equal_label_seq = False
                if oov > 0: #Implicitly ignore OOV sentence for search error, the score comparision between an OOV gt and hyp makes no sense after all
                    sent_oov += 1

                elif score_ground_truth > score_search:# TODO add threshold
                    is_search_error = True
                    num_search_errors += 1

            with open("search_errors_log", "a") as f:
                log_txt = "Seq Tag: %s\n\tGround-truth score: %f\n\tSearch score: %f" % (
                    seq_tag,
                    score_ground_truth,
                    score_search,
                )
                log_txt += "\n\tGround-truth seq: %s\n\tSearch seq:       %s" % (
                    str(targets_ground_truth),
                    str(targets_search),
                )
                log_txt += "\n\tEqual label sequences: %s\n\t-> %s" % (
                    str(equal_label_seq),
                    "Search error!" if is_search_error else "No search error!",
                )
                log_txt += f"\n\tNum_oov: {oov}\n\n"
                f.write(log_txt)

        with open(self.out_search_errors.get_path(), "w+") as f:
            f.write("Search errors: %.2f%%" % ((num_search_errors / num_seqs) * 100) + "\n" +
                    "Search errors/total errors: %.2f%%" % ((num_search_errors / num_unequal) * 100) + "\n" +
                    "Sent_ER: %.2f%%" % ((num_unequal / num_seqs) * 100) + "\n" +
                    "Sent_OOV: %.2f%%" % ((sent_oov / num_seqs) * 100) + "\n" +
                    "OOV: %.2f%%" % ((num_oov / num_words) * 100) + "\n")




# --------- small, dependency-free edit distance (token-level) ----------
def levenshtein_tokens(a_tokens: List[str], b_tokens: List[str]) -> int:
    # classic DP, O(|a|*|b|)
    m, n = len(a_tokens), len(b_tokens)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        ai = a_tokens[i - 1]
        for j in range(1, n + 1):
            tmp = dp[j]
            cost = 0 if ai == b_tokens[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = tmp
    return dp[n]

# --------- recording tag extraction -----------------------------------
def extract_recording_tag(seq_tag: str,
                          dataset_hint: Optional[str] = None,
                          recording_regex: Optional[str] = None) -> str:
    """
    Return a recording tag for aggregation.

    Priority:
      1) If recording_regex is provided, use the first capturing group.
      2) If dataset_hint == 'librispeech', return first two path parts.
      3) Else, return prefix before the last '/'.
    """
    if recording_regex:
        m = re.match(recording_regex, seq_tag)
        if m and m.groups():
            return m.group(1)

    parts = seq_tag.split("/")
    if dataset_hint and dataset_hint.lower() == "librispeech":
        if len(parts) >= 2:
            return "/".join(parts[:1] + parts[1].split("-")[:-1])
        return seq_tag  # fallback

    # generic: prefix until last '/'
    if "/" in seq_tag:
        return seq_tag.rsplit("/", 1)[0]
    return seq_tag

# --------- stats helpers -----------------------------------------------
def _mean(xs: List[float]) -> float:
    return float(sum(xs)) / max(len(xs), 1)

def _std(xs: List[float], ddof: int = 0) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (n - ddof)
    return math.sqrt(var)

@dataclass
class DiagnosticsConfig:
    dataset_hint: Optional[str] = None          # "librispeech" or None
    recording_regex: Optional[str] = None       # e.g. r'^(test-other/\d+-\d+)-\d+/.+$'
    # If you want to flip gap definition, change here; by request we keep (GT - Hyp).
    gap_definition: str = "gt_minus_hyp"

class SearchErrorDiagnosticsJob(Job):
    """
    Inputs:
      - gt_json: path to GT dict {seq_tag: [score, normalized_text]}
      - hyp_json: path to Hyp dict {seq_tag: [score, normalized_text]}
      - missing_gt_json: path to dict whose keys are seq_tags with search errors
    Outputs:
      - diagnostics JSON with stats + sanity check
      - optional human-readable txt summary
    """
    def __init__(self,
                 gt_json: tk.Path,
                 hyp_json: tk.Path,
                 missing_gt_json: tk.Path,
                 config: Optional[DiagnosticsConfig] = None):
        self.gt_json = gt_json
        self.hyp_json = hyp_json
        self.missing_gt_json = missing_gt_json
        self.out_json = self.output_path("output.json")
        self.out_txt = self.output_path("output.txt")
        self.config = config or DiagnosticsConfig()

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # 1) load
        with open(self.gt_json, "r", encoding="utf-8") as f:
            gt: Dict[str, Tuple[float, str]] = json.load(f)
        with open(self.hyp_json, "r", encoding="utf-8") as f:
            hyp: Dict[str, Tuple[float, str]] = json.load(f)
        with open(self.missing_gt_json, "r", encoding="utf-8") as f:
            missing_gt: Dict[str, object] = json.load(f)

        gt_keys = set(gt.keys())
        hyp_keys = set(hyp.keys())
        common = sorted(gt_keys & hyp_keys)

        # Stats over DIFFERENT segments only
        gaps: List[float] = []
        edits: List[int] = []
        perseq = {}

        num_equal = 0
        num_different = 0
        equal_examples = []
        diff_examples = []

        max_gap = (-float("inf"), None)  # most positive gap
        min_gap_pos = (float("inf"), None)  # smallest positive gap (>0)
        max_edit = (-1, None)
        min_edit_pos = (10 ** 9, None)

        # NEW: sign stats over ALL COMMON segments (including equal)
        neg_count = 0
        pos_count = 0
        zero_count = 0

        # Track most negative gap among DIFFERENT segments for the requested example
        most_negative_gap = (float("inf"), None)  # store the *lowest* value (most negative)

        for k in common:
            gt_score, gt_text = gt[k]
            hyp_score, hyp_text = hyp[k]
            gap_all = float(gt_score) - float(hyp_score)  # always defined

            # --- sign counters over ALL common segments ---
            if gap_all < 0:
                neg_count += 1
            elif gap_all > 0:
                pos_count += 1
            else:
                zero_count += 1

            # Skip stats if identical strings; still counted above
            if gt_text == hyp_text:
                num_equal += 1
                if len(equal_examples) < 10:
                    equal_examples.append(k)
                continue

            # DIFFERENT → contributes to gap/edit stats
            num_different += 1
            if len(diff_examples) < 10:
                diff_examples.append(k)

            gap = gap_all  # GT - Hyp
            gt_toks = gt_text.split()
            hyp_toks = hyp_text.split()
            edit = levenshtein_tokens(gt_toks, hyp_toks)

            gaps.append(gap)
            edits.append(edit)
            perseq[k] = {
                "gap": gap,
                "edit_distance_tokens": edit,
                "gt_score": float(gt_score),
                "hyp_score": float(hyp_score),
                "gt_text": gt_text,
                "hyp_text": hyp_text,
            }

            if gap > max_gap[0]:
                max_gap = (gap, k)
            if gap > 0 and gap < min_gap_pos[0]:
                min_gap_pos = (gap, k)
            if gap < most_negative_gap[0]:
                most_negative_gap = (gap, k)
            if edit > max_edit[0]:
                max_edit = (edit, k)
            if edit > 0 and edit < min_edit_pos[0]:
                min_edit_pos = (edit, k)

        # per-recording counts from missing_gt_dict
        missing_keys = set(missing_gt.keys())
        per_recording_counts: Dict[str, int] = {}
        for k in sorted(missing_keys):
            rec_tag = extract_recording_tag(
                k, dataset_hint=self.config.dataset_hint,
                recording_regex=self.config.recording_regex
            )
            per_recording_counts[rec_tag] = per_recording_counts.get(rec_tag, 0) + 1

        # Sanity check
        reference_total = max(len(gt_keys), 1)
        search_error_rate = len(missing_keys) / reference_total

        diff_keys = set(perseq.keys())
        all_diffs_in_missing = diff_keys.issubset(missing_keys)
        all_missing_are_diffs = missing_keys.issubset(diff_keys)
        diff_but_not_missing = sorted(list(diff_keys - missing_keys))
        missing_but_not_diff = sorted(list(missing_keys - diff_keys))

        # Aggregates over DIFFERENT segments only
        gap_stats = {
            "count": len(gaps),
            "avg": _mean(gaps) if gaps else 0.0,
            "min": min(gaps) if gaps else 0.0,
            "max": max(gaps) if gaps else 0.0,
            "std": _std(gaps) if gaps else 0.0,
        }
        edit_stats = {
            "count": len(edits),
            "avg": _mean(edits) if edits else 0.0,
            "min": min(edits) if edits else 0,
            "max": max(edits) if edits else 0,
            "std": _std(edits) if edits else 0.0,
        }

        # NEW: sign stats over ALL COMMON segments (counts + percentages)
        total_common = max(len(common), 1)
        gap_sign_over_all_common = {
            "total_common": len(common),
            "negative": {
                "count": neg_count,
                "percent": neg_count / total_common
            },
            "positive": {
                "count": pos_count,
                "percent": pos_count / total_common
            },
            "zero": {
                "count": zero_count,
                "percent": zero_count / total_common
            },
        }

        # Extremes over DIFFERENT segments only, plus a negative example
        extremes = {
            "max_score_gap": {
                "seq_tag": max_gap[1],
                **(perseq.get(max_gap[1], {}) if max_gap[1] is not None else {})
            },
            "min_score_gap_positive": {
                "seq_tag": min_gap_pos[1],
                **(perseq.get(min_gap_pos[1], {}) if min_gap_pos[1] is not None and min_gap_pos[0] != float(
                    "inf") else {})
            },
            "most_negative_score_gap": {
                "seq_tag": most_negative_gap[1],
                **(perseq.get(most_negative_gap[1], {}) if most_negative_gap[1] is not None else {})
            },
            "max_edit_distance": {
                "seq_tag": max_edit[1],
                **(perseq.get(max_edit[1], {}) if max_edit[1] is not None else {})
            },
            "min_edit_distance_positive": {
                "seq_tag": min_edit_pos[1],
                **(perseq.get(min_edit_pos[1], {}) if min_edit_pos[1] is not None and min_edit_pos[
                    0] != 10 ** 9 else {})
            },
        }

        sanity = {
            "missing_gt_count": len(missing_keys),
            "ref_total": reference_total,
            "search_error_rate": search_error_rate,
            "loop_counts": {
                "num_equal": num_equal,
                "num_different": num_different,
                "equal_examples": equal_examples,
                "different_examples": diff_examples,
            },
            "diff_keys_between_gt_and_hyp": {
                "count": len(diff_keys),
                "example_mismatches": sorted(list(diff_keys))[:20],
            },
            "all_diffs_listed_in_missing_gt": all_diffs_in_missing,
            "all_missing_gt_are_diffs": all_missing_are_diffs,
            "diff_but_not_in_missing_gt": diff_but_not_missing[:50],
            "missing_gt_but_not_diff": missing_but_not_diff[:50],
        }

        out_payload = {
            "config": {
                "dataset_hint": self.config.dataset_hint,
                "recording_regex": self.config.recording_regex,
                "gap_definition": self.config.gap_definition,
            },
            "summary_over_different_segments_only": {
                "score_gap": gap_stats,
                "edit_distance_tokens": edit_stats,
            },
            "gap_sign_over_all_common_segments": gap_sign_over_all_common,  # NEW
            "extremes_over_different_segments_only": extremes,
            "per_recording_search_error_counts": per_recording_counts,
            "sanity_check": sanity,
        }

        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)

        if self.out_txt:
            with open(self.out_txt, "w", encoding="utf-8") as f:
                f.write("# Search Error Diagnostics (different segments only)\n")
                f.write(json.dumps(out_payload["summary_over_different_segments_only"], ensure_ascii=False, indent=2))
                f.write("\n\n## Gap sign over ALL common segments\n")
                f.write(json.dumps(out_payload["gap_sign_over_all_common_segments"], ensure_ascii=False, indent=2))
                f.write("\n\n## Extremes (different segments only)\n")
                f.write(json.dumps(out_payload["extremes_over_different_segments_only"], ensure_ascii=False, indent=2))
                f.write("\n\n## Per-recording counts (top 50)\n")
                top = sorted(per_recording_counts.items(), key=lambda x: (-x[1], x[0]))[:50]
                for k, v in top:
                    f.write(f"{k}\t{v}\n")
                f.write("\n\n## Sanity check\n")
                f.write(json.dumps(out_payload["sanity_check"], ensure_ascii=False, indent=2))