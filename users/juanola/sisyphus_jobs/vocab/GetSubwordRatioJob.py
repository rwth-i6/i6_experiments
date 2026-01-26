from sisyphus import Job, tk, Task

import i6_core.util as util
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob


class GetSubwordRatioJob(Job):
    """
    Apply subword codes on a text file and get ratio
    """

    def __init__(
            self,
            text_file: tk.Path,
            spm_model: tk.Path,
            mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param vocab as str
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file

        self.subword_text = ApplySentencepieceToTextJob(
            text_file=self.text_file,
            sentencepiece_model=spm_model,
            enable_unk=False,
        ).out_sentencepiece_text

        self.out_ratio = self.output_var("subword_to_word_ratio")

        self.mini_task = mini_task
        self.rqmt = {"cpu": 2, "mem": 4, "time": 2}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        # Compute BPE-to-original token ratio:
        total_orig_tokens = 0
        total_subword_tokens = 0
        with util.uopen(self.text_file, "rt") as fin, util.uopen(self.subword_text, "rt") as fout:
            for orig_line, bpe_line in zip(fin, fout):
                orig_tokens = orig_line.strip().split()
                bpe_tokens = bpe_line.strip().split()
                total_orig_tokens += len(orig_tokens)
                total_subword_tokens += len(bpe_tokens)

        # avoid division by zero
        if total_orig_tokens > 0:
            self.out_ratio.set(total_subword_tokens / total_orig_tokens)
        else:
            self.out_ratio.set(1.0)  # fallback to 1.0 if no tokens

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)
