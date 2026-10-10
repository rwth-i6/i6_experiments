"""Per-source WER split of one sclite scoring, the corpus ctm and stm cut by seq tag pattern and rescored."""

import json
import os
import re
import subprocess
from typing import Dict, List, Optional

try:
    from sisyphus import Job, Task, tk
except ImportError:
    Job, Task, tk = object, None, None

OTHER_SOURCE = "other"


def assign_source(tag: str, sources: Dict[str, str]) -> str:
    """
    Names the corpus one sequence comes from.

    :param tag: the seq tag
    :param sources: per source its seq tag pattern, tried in order
    :return: the first source whose pattern matches the tag, OTHER_SOURCE when none does
    """
    for name, pattern in sources.items():
        if re.search(pattern, tag):
            return name
    return OTHER_SOURCE


def split_by_source(lines: List[str], sources: Dict[str, str]) -> Dict[str, List[str]]:
    """Group ctm or stm lines by the source of their first field, comment lines dropped."""
    out: Dict[str, List[str]] = {}
    for line in lines:
        if line.startswith(";;") or not line.strip():
            continue
        out.setdefault(assign_source(line.split(None, 1)[0], sources), []).append(line)
    return out


_DTL_KEYS = {
    "Percent Total Error": "num_errors",
    "Percent Substitution": "num_sub",
    "Percent Deletions": "num_del",
    "Percent Insertions": "num_ins",
    "Ref. words": "ref_words",
}


def parse_dtl(text: str, precision_ndigit: Optional[int] = 2) -> Dict[str, float]:
    """WER, sub, del and ins percentages recomputed from the absolute counts of a sclite dtl report."""
    counts: Dict[str, int] = {}
    for line in text.splitlines():
        for key, name in _DTL_KEYS.items():
            if line.startswith(key):
                m = re.search(r"\((\s*\d+)\)\s*$", line)
                assert m, line
                counts[name] = int(m.group(1))
    ref_words = counts["ref_words"]

    def pct(n: int) -> float:
        value = 100.0 * n / ref_words if ref_words > 0 else float("nan")
        return round(value, precision_ndigit) if precision_ndigit is not None else value

    return {
        "wer": pct(counts["num_errors"]),
        "sub": pct(counts["num_sub"]),
        "del": pct(counts["num_del"]),
        "ins": pct(counts["num_ins"]),
        "num_errors": counts["num_errors"],
        "ref_words": ref_words,
    }


class SourceSplitScliteJob(Job):
    """Score the ctm of one recognition against its stm once per source, sources given as seq tag patterns."""

    def __init__(self, *, ref: tk.Path, hyp: tk.Path, sources: Dict[str, str], sctk_binary_path: tk.Path):
        self.ref = ref
        self.hyp = hyp
        self.sources = dict(sources)
        self.sctk_binary_path = sctk_binary_path
        self.out_report_dir = self.output_path("reports", True)
        self.out_report = self.output_path("wer_by_source.json")
        self.out_wers = {name: self.output_var(f"wer_{name}") for name in list(self.sources) + [OTHER_SOURCE]}

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 1}, mini_task=True)

    def run(self):
        with open(self.ref.get_path(), "rt", errors="ignore") as f:
            ref_lines = split_by_source(f.readlines(), self.sources)
        with open(self.hyp.get_path(), "rt", errors="ignore") as f:
            hyp_lines = split_by_source(f.readlines(), self.sources)
        report_dir = self.out_report_dir.get_path()
        sclite = os.path.join(self.sctk_binary_path.get_path(), "sclite")
        report: Dict[str, Dict[str, float]] = {}
        for name in self.out_wers:
            if name not in ref_lines:
                self.out_wers[name].set(None)
                continue
            stm = os.path.join(report_dir, f"{name}.stm")
            ctm = os.path.join(report_dir, f"{name}.ctm")
            with open(stm, "wt") as f:
                f.writelines(ref_lines[name])
            with open(ctm, "wt") as f:
                f.write(";; <name> <track> <start> <duration> <word> <confidence>\n")
                f.writelines(hyp_lines.get(name, []))
            subprocess.check_call(
                [sclite, "-r", stm, "stm", "-h", ctm, "ctm", "-o", "dtl", "-o", "pra", "-n", name, "-O", report_dir]
            )
            with open(os.path.join(report_dir, f"{name}.dtl"), "rt", errors="ignore") as f:
                stats = parse_dtl(f.read())
            stats["num_seqs"] = len(ref_lines[name])
            report[name] = stats
            self.out_wers[name].set(stats["wer"])
        with open(self.out_report.get_path(), "wt") as f:
            json.dump(report, f, indent=1, sort_keys=True)
            f.write("\n")
