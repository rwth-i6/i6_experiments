"""
Phoneme Error Rate (PER) / Token Edit Distance Evaluation Job for Phoneme-Level UNMT.
Supports both RETURNN search output dicts (search_out.py.gz) and plain line-by-line hypothesis files,
and optionally computes CTC-collapsed PER (collapsing repeated tokens and dropping 'BLK') when
evaluating hypotheses in the EN+Blank domain.
"""

from typing import Dict, Iterator, List, Optional, Tuple
import json
import logging

from sisyphus import Job, Task, tk
from .bleu_eval import _load_text_or_dict, clean_mt_tokens


def _edit_distance_ops(hyp: List[str], ref: List[str]) -> Tuple[int, int, int]:
    """
    Computes (substitutions, deletions, insertions) to transform `ref` into `hyp`
    via standard Wagner-Fischer dynamic programming.
    """
    m, n = len(ref), len(hyp)
    if m == 0:
        return (0, 0, n)
    if n == 0:
        return (0, m, 0)

    # Track (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(n + 1)]
    for i in range(1, m + 1):
        curr = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * n
        r_tok = ref[i - 1]
        for j in range(1, n + 1):
            if r_tok == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                sub_c = prev[j - 1][0] + 1
                del_c = prev[j][0] + 1
                ins_c = curr[j - 1][0] + 1
                best = min(sub_c, del_c, ins_c)
                if best == sub_c:
                    _, s, d, ins = prev[j - 1]
                    curr[j] = (best, s + 1, d, ins)
                elif best == del_c:
                    _, s, d, ins = prev[j]
                    curr[j] = (best, s, d + 1, ins)
                else:
                    _, s, d, ins = curr[j - 1]
                    curr[j] = (best, s, d, ins + 1)
        prev = curr
    _, s, d, ins = prev[n]
    return (s, d, ins)


def _ctc_collapse(tokens: List[str], blank_token: str = "BLK") -> List[str]:
    out: List[str] = []
    prev = None
    for tok in tokens:
        if tok != prev:
            if tok != blank_token:
                out.append(tok)
            prev = tok
    return out


class ComputePerJob(Job):
    """
    Computes corpus-level Phoneme Error Rate (PER = (S + D + I) / N_ref * 100%)
    between decoded phoneme sequences and reference phoneme sequences.
    If `collapse_blank_token` is provided (e.g. 'BLK' for EN+Blank outputs), also reports
    collapsed phoneme error rate (`collapsed_per`) alongside exact frame-level token error rate.
    """
    _version = 1

    def __init__(
        self,
        hyp_file: tk.Path,
        ref_file: tk.Path,
        remove_unk: bool = True,
        collapse_blank_token: Optional[str] = None,
    ):
        super().__init__()
        self.hyp_file = hyp_file
        self.ref_file = ref_file
        self.remove_unk = remove_unk
        self.collapse_blank_token = collapse_blank_token

        self.out_per = self.output_var("per", pickle=False)
        self.out_summary_json = self.output_path("per_summary.json")
        self.out_report = self.output_path("per_report.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1}, mini_task=True)

    def run(self):
        hyp_dict, hyp_lines = _load_text_or_dict(self.hyp_file.get_path())
        ref_dict, ref_lines = _load_text_or_dict(self.ref_file.get_path())

        if hyp_dict is not None and ref_dict is not None:
            common = [k for k in ref_dict if k in hyp_dict]
            aligned_hyps = [hyp_dict[k] for k in common]
            aligned_refs = [ref_dict[k] for k in common]
        else:
            aligned_hyps = hyp_lines
            aligned_refs = ref_lines

        assert len(aligned_hyps) == len(aligned_refs) and len(aligned_refs) > 0, (
            f"Hypothesis/Reference line mismatch in ComputePerJob: {len(aligned_hyps)} vs {len(aligned_refs)}"
        )

        tot_s, tot_d, tot_i, tot_ref = 0, 0, 0, 0
        col_s, col_d, col_i, col_ref = 0, 0, 0, 0

        for h_raw, r_raw in zip(aligned_hyps, aligned_refs):
            h_toks = clean_mt_tokens(h_raw, remove_unk=self.remove_unk).split()
            r_toks = clean_mt_tokens(r_raw, remove_unk=False).split()
            s, d, ins = _edit_distance_ops(h_toks, r_toks)
            tot_s += s
            tot_d += d
            tot_i += ins
            tot_ref += len(r_toks)

            if self.collapse_blank_token:
                hc = _ctc_collapse(h_toks, self.collapse_blank_token)
                rc = _ctc_collapse(r_toks, self.collapse_blank_token)
                cs, cd, ci = _edit_distance_ops(hc, rc)
                col_s += cs
                col_d += cd
                col_i += ci
                col_ref += len(rc)

        per = (tot_s + tot_d + tot_i) / max(1, tot_ref) * 100.0
        self.out_per.set(round(per, 2))

        summary: Dict[str, object] = {
            "per": round(per, 2),
            "substitutions": int(tot_s),
            "deletions": int(tot_d),
            "insertions": int(tot_i),
            "ref_tokens": int(tot_ref),
            "num_sentences": len(aligned_refs),
        }
        if self.collapse_blank_token:
            col_per = (col_s + col_d + col_i) / max(1, col_ref) * 100.0
            summary["collapsed_per"] = round(col_per, 2)
            summary["collapsed_ref_tokens"] = int(col_ref)

        with open(self.out_summary_json.get_path(), "w", encoding="utf-8") as fj:
            json.dump(summary, fj, indent=2)

        with open(self.out_report.get_path(), "w", encoding="utf-8") as fr:
            fr.write("==========================================================\n")
            fr.write("          UNMT Phoneme Error Rate (PER) Report            \n")
            fr.write("==========================================================\n")
            fr.write(f"Sentences evaluated : {len(aligned_refs)}\n")
            fr.write(f"PER (%)             : {per:.2f}% (S={tot_s}, D={tot_d}, I={tot_i}, N={tot_ref})\n")
            if self.collapse_blank_token:
                fr.write(
                    f"CTC-Collapsed PER   : {summary['collapsed_per']:.2f}% "
                    f"(collapsed blank={self.collapse_blank_token!r}, N_col={col_ref})\n"
                )
