"""Recognition post-processing jobs: n-best LM rescoring."""

__all__ = ["NbestKenLmRescoreJob"]

import ast
import math

from sisyphus import Job, Task


class NbestKenLmRescoreJob(Job):
    """Rescore an n-best list with an n-gram LM via ``kenlm``, best hypothesis per sequence, once per ``lm_scale``."""

    def __init__(
        self,
        *,
        nbest_file,
        arpa_lm,
        lm_scales,
        am_scale: float = 1.0,
        word_ins_penalty: float = 0.0,
    ):
        self.nbest_file = nbest_file
        self.arpa_lm = arpa_lm
        self.lm_scales = tuple(float(s) for s in lm_scales)
        self.am_scale = float(am_scale)
        self.word_ins_penalty = float(word_ins_penalty)

        self.out_search_results = {
            ls: self.output_path(f"search_out.lm{ls}.py") for ls in self.lm_scales
        }

        # loading the librispeech 4-gram arpa (gz) into kenlm is ram-heavy
        self.rqmt = {"cpu": 1, "mem": 30, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import kenlm

        with open(self.nbest_file.get_path(), "rt") as f:
            nbest = ast.literal_eval(f.read())

        model = kenlm.Model(self.arpa_lm.get_path())
        ln10 = math.log(10.0)  # kenlm scores are log10, convert to natural log to match am_score units

        scored = {}
        for seq_tag, entries in nbest.items():
            lst = []
            for am_score, text in entries:
                # upper() because the librispeech arpa lm is trained on uppercase text
                lm_ln = model.score(text.upper(), bos=True, eos=True) * ln10
                lst.append((float(am_score), lm_ln, len(text.split()), text))
            scored[seq_tag] = lst

        for lm_scale in self.lm_scales:
            results = {}
            for seq_tag, lst in scored.items():
                best_text, best_score = "", None
                for am_score, lm_ln, n_words, text in lst:
                    score = (
                        self.am_scale * am_score
                        + lm_scale * lm_ln
                        + self.word_ins_penalty * n_words
                    )
                    if best_score is None or score > best_score:
                        best_score, best_text = score, text
                results[seq_tag] = best_text

            # emit the returnn search-output dict format (seq_tag to hyp) so downstream scoring reads it
            with open(self.out_search_results[lm_scale].get_path(), "wt") as f:
                f.write("{\n")
                for seq_tag, text in results.items():
                    f.write(f"{repr(str(seq_tag))}: {repr(text)},\n")
                f.write("}\n")
