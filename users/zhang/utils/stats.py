from __future__ import annotations
from typing import Dict, List, Tuple, Any

import ast
import gzip
import json
import os
from collections import defaultdict

from sisyphus import Job, Task, tk


Hyp = Tuple[float, str]
NBestDict = Dict[str, List[Hyp]]


class NBestStatsAndDedupJob(Job):
    """
    Compute stats for an N-best output (i6-style py-dict) and produce a deduplicated N-best.

    Input format (pydict):
        {
            "utt1": [
                (-12.34, "hola que tal"),
                (-12.40, "hola que tal"),
                (-13.00, "hola que tal estas"),
                ...
            ],
            "utt2": [...],
            ...
        }

    Outputs:
        - out_stats.json: global + per-utterance stats
        - out_nbest.py.gz: deduplicated N-best in same dict format
    """

    def __init__(self, nbest_pydict: tk.Path):
        super().__init__()

        self.nbest_pydict = nbest_pydict

        # outputs
        self.out_stats = self.output_path("stats.json")
        self.out_nbest = self.output_path("nbest_dedup.py.gz")
        self.out_num_eff_hyps = self.output_var("num_eff_hyps")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        nbest: NBestDict = self._load_pydict(self.nbest_pydict.get_path())

        dedup_nbest: NBestDict = {}
        per_utt_stats: Dict[str, Dict[str, Any]] = {}

        total_utts = 0
        total_hyps_before = 0
        total_hyps_after = 0
        total_unique_global = 0  # counting per-utt unique (sum)

        # Optional: global unique across all utterances
        global_unique_hyps = set()

        for utt, hyps in nbest.items():
            total_utts += 1
            n_before = len(hyps)
            total_hyps_before += n_before

            dedup_hyps = self._dedup_hyps(hyps)
            n_after = len(dedup_hyps)
            total_hyps_after += n_after
            total_unique_global += n_after

            # update global unique set (by raw text)
            for _, txt in dedup_hyps:
                global_unique_hyps.add(txt)

            # per-utterance stats
            per_utt_stats[utt] = {
                "num_hyps_before": n_before,
                "num_hyps_after": n_after,
                "num_duplicates_removed": n_before - n_after,
                "dup_fraction": (n_before - n_after) / n_before if n_before > 0 else 0.0,
            }

            dedup_nbest[utt] = dedup_hyps

        global_stats = {
            "num_utts": total_utts,
            "total_hyps_before": total_hyps_before,
            "total_hyps_after": total_hyps_after,
            "total_duplicates_removed": total_hyps_before - total_hyps_after,
            "avg_hyps_per_utt_before": total_hyps_before / total_utts if total_utts > 0 else 0.0,
            "avg_hyps_per_utt_after": total_hyps_after / total_utts if total_utts > 0 else 0.0,
            "sum_unique_hyps_over_utts": total_unique_global,
            "num_global_distinct_hyps": len(global_unique_hyps),
        }
        self.out_num_eff_hyps.set(total_unique_global)
        stats_out = {
            "global": global_stats,
            "per_utt": per_utt_stats,
        }

        # write stats
        with open(self.out_stats.get_path(), "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2, ensure_ascii=False)

        # write dedup N-best in pydict form
        with gzip.open(self.out_nbest.get_path(), "wt", encoding="utf-8") as f:
            # keep it compatible with the usual Returnn / i6 style
            f.write("{\n")
            for seq_tag, nbest_hyps in dedup_nbest.items():
                f.write(f"{seq_tag!r}: [\n")
                for score, hyp in nbest_hyps:
                    f.write(f"  ({score!r}, {hyp!r}),\n")
                f.write("],\n")
            f.write("}\n")

    @staticmethod
    def _load_pydict(path: str) -> NBestDict:
        """Load an i6-style pydict from .py or .py.gz using ast.literal_eval."""
        if path.endswith(".gz"):
            opener = gzip.open
            mode = "rt"
        else:
            opener = open
            mode = "r"

        with opener(path, mode, encoding="utf-8") as f:
            content = f.read()
        return ast.literal_eval(content)

    @staticmethod
    def _dedup_hyps(hyps: List[Hyp]) -> List[Hyp]:
        """
        Deduplicate hypotheses by text; keep the entry with the best (highest) score for each text.
        Sort again by score descending.
        """
        best_by_text: Dict[str, Hyp] = {}

        for score, txt in hyps:
            # Optionally normalize text; here we keep it raw, but you could do:
            # key = txt.strip()
            key = txt
            if key not in best_by_text or score > best_by_text[key][0]:
                best_by_text[key] = (float(score), txt)

        # return list sorted by score descending
        return sorted(best_by_text.values(), key=lambda x: x[0], reverse=True)
