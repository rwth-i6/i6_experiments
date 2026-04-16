from sisyphus import tk, Job, Task
import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, List



class ReportDictJob(Job):
    """
    reports.
    """
    def __init__(
        self,
        *,
        outputs: Dict[str, tk.Path | tk.Variable],
    ):
        super(ReportDictJob, self).__init__()
        self.outputs = outputs  # type: Dict[str, tk.Path]
        self.out_report_dict = self.output_path("report.py")

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", rqmt={"cpu":1, "time":1})#mini_task=True)

    def run(self):
        """run"""
        with open(self.out_report_dict.get_path(), "wt") as out:
            out.write("{\n")
            for name, res in self.outputs.items():
                if isinstance(res, tk.Variable):
                    res = res.get()
                elif isinstance(res, tk.Path):
                    with open(res, "rt") as infile:
                        res = infile.read()
                out.write(f"\t{name!r}: \n\t'{res}',\n")
            out.write("}\n")

from i6_experiments.users.zeyer.datasets.score_results import ScoreResult

PRINTED = False  # keep as in your original module


class GetOutPutsJob(Job):
    """
    Collect WER reports and produce per-dataset S/D/I tables by parsing sclite .dtl files.
    Outputs:
      - wer_report.py (unchanged mapping for paths)
      - sdi_<dataset>.tsv (LM rows with Sub/Del/Ins counts)
    """
    def __init__(self, *, outputs: Dict[str, Dict[str, "ScoreResult"]]):
        super().__init__()
        global PRINTED
        self.outputs = outputs  # type: Dict[str, Dict[str, "ScoreResult"]]
        self.out_report_dict = self.output_path("wer_report.py")

        if not PRINTED:
            for k, v in self.outputs.items():
                for dataset, report in v.items():
                    print(f"{k}:\n{dataset}: {report}")
                    break
            PRINTED = True

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "time": 3})

    # ---------------------------
    # Helpers
    # ---------------------------

    def _find_dtl(self, report_dir: Path) -> Optional[Path]:
        """
        Heuristic: prefer a top-level *.dtl; otherwise first *.dtl anywhere.
        """
        if not report_dir.exists():
            return None
        top = list(report_dir.glob("*.dtl"))
        if top:
            # If multiple, a stable pick; optionally bias filenames with 'dtl' in stem (already is)
            return sorted(top)[0]
        for p in report_dir.rglob("*.dtl"):
            return p
        return None

    # Match both percent and count:  Percent Substitution =   3.3%   (1703)
    _RE_SUB = re.compile(r"Percent\s+Substitution\s*=\s*([0-9.]+)%\s*\(\s*(\d+)\s*\)")
    _RE_DEL = re.compile(r"Percent\s+Deletions\s*=\s*([0-9.]+)%\s*\(\s*(\d+)\s*\)")
    _RE_INS = re.compile(r"Percent\s+Insertions\s*=\s*([0-9.]+)%\s*\(\s*(\d+)\s*\)")

    def _parse_sdi_counts(self, dtl_path: Path) -> Optional[Dict[str, float]]:
        """
        Returns dict with percent and count for S/D/I, e.g.
        {'sub%': 3.3, 'sub#': 1703, 'del%': 0.4, 'del#': 191, 'ins%': 0.4, 'ins#': 211}
        """
        try:
            txt = dtl_path.read_text(errors="ignore")
        except Exception:
            return None

        def extract(rx):
            m = rx.search(txt)
            return (float(m.group(1)), int(m.group(2))) if m else (float("nan"), 0)

        sub_p, sub_c = extract(self._RE_SUB)
        del_p, del_c = extract(self._RE_DEL)
        ins_p, ins_c = extract(self._RE_INS)

        if any(x != x for x in (sub_p, del_p, ins_p)):  # NaN check
            return None

        return {
            "sub%": sub_p, "sub#": sub_c,
            "del%": del_p, "del#": del_c,
            "ins%": ins_p, "ins#": ins_c,
        }

    def _safe_name(self, s: str) -> str:
        return re.sub(r"[^\w.-]+", "_", s)

    # ---------------------------
    # Main
    # ---------------------------

    def run(self):
        # Keep the original wer_report.py for reference
        with open(self.out_report_dict.get_path(), "wt") as out:
            out.write("{\n")
            for lm, wer_dict in self.outputs.items():
                out.write(f"\t{lm!r}" + ":{\n")
                for dataset, score_res in wer_dict.items():
                    out.write(f"{dataset!r}: {score_res.report}\n")
                out.write("\t}\n")
            out.write("}\n")

        # Aggregate S/D/I per dataset across LMs
        per_dataset: Dict[str, Dict[str, Dict[str, int]]] = {}

        for lm, wer_dict in self.outputs.items():
            for dataset, score_res in wer_dict.items():
                report_dir = Path(str(score_res.report))
                dtl = self._find_dtl(report_dir)
                sdi = self._parse_sdi_counts(dtl) if dtl else None
                if sdi is None:
                    # If missing, fill zeros to keep table rectangular
                    sdi = {"sub": 0, "del": 0, "ins": 0}
                per_dataset.setdefault(dataset, {})[lm] = sdi

        # Emit one TSV per dataset
        for dataset, lm_map in sorted(per_dataset.items()):
            tsv_path = self.output_path(f"sdi_{self._safe_name(dataset)}.csv")
            with open(tsv_path.get_path(), "wt") as f:
                f.write(",".join(["LM", "Sub(%)", "Sub(#)", "Del(%)", "Del(#)", "Ins(%)", "Ins(#)"]) + "\n")
                for lm, sdi in sorted(lm_map.items()):
                    f.write(",".join([
                        lm,
                        f"{sdi['sub%']:.2f}", str(sdi['sub#']),
                        f"{sdi['del%']:.2f}", str(sdi['del#']),
                        f"{sdi['ins%']:.2f}", str(sdi['ins#']),
                    ]) + "\n")

        # Compute LM-wise averages over datasets
        # We macro-average percentages over datasets that have valid numbers,
        # and sum counts over those same datasets.
        lm_stats: Dict[str, Dict[str, float]] = {}
        lm_counts: Dict[str, int] = {}  # number of datasets contributing to averages per LM

        def _is_num(x) -> bool:
            try:
                return x is not None and x == x  # not NaN
            except Exception:
                return False

        for dataset, lm_map in per_dataset.items():
            for lm, vals in lm_map.items():
                subp, delp, insp = vals.get("sub%"), vals.get("del%"), vals.get("ins%")
                subn, deln, insn = vals.get("sub#"), vals.get("del#"), vals.get("ins#")
                # Only count datasets where all three percentages were parsed
                if _is_num(subp) and _is_num(delp) and _is_num(insp):
                    st = lm_stats.setdefault(lm, {"sub%_sum": 0.0, "del%_sum": 0.0, "ins%_sum": 0.0,
                                                  "sub#_sum": 0.0, "del#_sum": 0.0, "ins#_sum": 0.0})
                    st["sub%_sum"] += float(subp)
                    st["del%_sum"] += float(delp)
                    st["ins%_sum"] += float(insp)
                    if _is_num(subn): st["sub#_sum"] += float(subn)
                    if _is_num(deln): st["del#_sum"] += float(deln)
                    if _is_num(insn): st["ins#_sum"] += float(insn)
                    lm_counts[lm] = lm_counts.get(lm, 0) + 1

        # Emit LM-average TSV
        avg_path = self.output_path("sdi_average.csv")
        with open(avg_path.get_path(), "wt") as f:
            f.write(",".join([
                "LM",
                "Datasets(#)",
                "Sub(%)_avg", "Del(%)_avg", "Ins(%)_avg",
                "Sub(#)_sum", "Del(#)_sum", "Ins(#)_sum",
            ]) + "\n")
            for lm in sorted(lm_stats.keys() | lm_counts.keys()):
                k = lm_counts.get(lm, 0)
                st = lm_stats.get(lm, None)
                if not st or k == 0:
                    f.write(",".join([lm, "0", "NaN", "NaN", "NaN", "0", "0", "0"]) + "\n")
                    continue
                sub_avg = st["sub%_sum"] / k
                del_avg = st["del%_sum"] / k
                ins_avg = st["ins%_sum"] / k
                f.write(",".join([
                    lm,
                    str(k),
                    f"{sub_avg:.2f}", f"{del_avg:.2f}", f"{ins_avg:.2f}",
                    str(int(round(st["sub#_sum"]))),
                    str(int(round(st["del#_sum"]))),
                    str(int(round(st["ins#_sum"]))),
                ]) + "\n")


def _is_num(x) -> bool:
    try:
        return x is not None and x == x  # not NaN
    except Exception:
        return False


def _fmt(x) -> str:
    if not _is_num(x):
        return "NaN"
    return f"{float(x):.2f}"


def _fmt_int(x) -> str:
    if not _is_num(x):
        return "NaN"
    return str(int(round(float(x))))

