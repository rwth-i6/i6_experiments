from sisyphus import Job, tk, Task
from typing import Optional, Any, Dict
import i6_core.util as util
import os
import sentencepiece
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

class ApplySentencepieceToWordOutputJob(Job):
    """
    Apply sentencepiece model on a text file, basically a wrapper for spm.encode
    """

    def __init__(
        self,
        *,
        search_py_output: tk.Path,
        sentencepiece_model: tk.Path,
        enable_unk: bool = True,
        gzip_output: bool = True,
    ):
        """
        :param search_py_output: words recog_out file to convert to sentencepiece
        :param sentencepiece_model: path to the trained sentencepiece model
        :param enable_unk: whether enable unk to map OOV symbol to the unknown symbol set in training or keep it as is
        :param gzip_output: use gzip on the output text
        """
        self.search_py_output = search_py_output
        self.sentencepiece_model = sentencepiece_model
        self.enable_unk = enable_unk
        self.out_search_results = self.output_path("search_results.py" + (".gz" if gzip_output else ""))

        self.rqmt: Optional[Dict[str, Any]] = {"cpu": 1, "mem": 2.0, "time": 2.0}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, mini_task=self.rqmt is None)

    def run(self):
        spm = sentencepiece.SentencePieceProcessor(model_file=self.sentencepiece_model.get_path())
        if self.enable_unk:
            spm.set_encode_extra_options("unk")

        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_search_results.get_path())

        def _transform_text(s: str):
            return " ".join(spm.encode(s.rstrip("\n"), out_type=str))

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, _transform_text(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, _transform_text(entry)))
            out.write("}\n")

def RecogOut_words_to_spm(data: RecogOutput, spm:SentencePieceModel):
    """words to spms"""
    spms = ApplySentencepieceToWordOutputJob(search_py_output=data.output, sentencepiece_model=spm.model_file, gzip_output=True).out_search_results
    return RecogOutput(output=spms)