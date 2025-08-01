import collections
from typing import Dict, List, Sequence, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen
import re


class FastButInaccurateWer(Job):
    def __init__(
        self,
        *,
        ref: tk.Path,
        hyp_replacement_list: Sequence[Tuple[str, str]] = [],
        hyp: tk.Path,
    ):
        import Levenshtein

        self.ref = ref  # file with text on each line, i.e. LmDataset
        self.hyp = hyp
        self.hyp_replacement_list = hyp_replacement_list or []

        self.out_wer = self.output_var("wer")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 3
        return super().hash(d)

    def tasks(self):
        # yield Task("run", rqmt={"cpu": 16, "mem": 8, "time": 1})
        yield Task("run", mini_task=True)

    def run(self):
        import Levenshtein

        edit_dist = 0
        total = 0
        num_lines = 0
        with uopen(self.ref, "rt") as ref_file, uopen(self.hyp, "rt") as hyp_file:
            for ref_line, hyp_line in zip(ref_file, hyp_file):
                assert isinstance(ref_line, str) and isinstance(hyp_line, str), (
                    "Both reference and hypothesis lines must be strings."
                )
                # assert ref_line.strip() and hyp_line.strip(), "Reference and hypothesis lines must not be empty."
                for old, new in self.hyp_replacement_list:
                    hyp_line = hyp_line.replace(old, new)
                ref_line = ref_line.strip().split()
                hyp_line = hyp_line.strip().split()

                edit_dist += (cur_dist := Levenshtein.distance(ref_line, hyp_line))
                total += len(ref_line)
                num_lines += 1

                if num_lines % 1000 == 0:
                    print(f"Processed {num_lines} lines...")
                    print(f"Reference: {ref_line}")
                    print(f"Hypothesis: {hyp_line}")
                    print(f"Current edit distance: {cur_dist}")

        wer = 100 * edit_dist / total if total > 0 else -1
        self.out_wer.set(wer)
