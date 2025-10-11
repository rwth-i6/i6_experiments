import tempfile

from sisyphus import Job, tk, Task, gs
from typing import Optional, Any, Dict, Type
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
import subprocess as sp
import shutil
import os
import sys
import sentencepiece
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
from returnn_common.datasets_old_2022_10.interface import VocabConfig
import i6_core.util as util

class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    __sis_hash_exclude__ = {"gzip_output": False}

    def __init__(
            self,
            text_file: tk.Path,
            bpe_codes: tk.Path,
            bpe_vocab: Optional[tk.Path] = None,
            subword_nmt_repo: Optional[tk.Path] = None,
            gzip_output: bool = False,
            mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param bpe_codes: bpe codes file, e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param bpe_vocab: if provided, then merge operations that produce OOV are reverted,
            use e.g. ReturnnTrainBpeJob.out_bpe_dummy_count_vocab
        :param subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        :param gzip_output: use gzip on the output text
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        self.gzip_output = gzip_output

        self.out_bpe_text = self.output_path("words_to_bpe.txt.gz" if gzip_output else "words_to_bpe.txt")

        self.mini_task = mini_task
        self.rqmt = {"cpu": 2, "mem": 4, "time": 12}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            input_file = self.text_file.get_path()
            tmp_infile = os.path.join(tmp, "in_text.txt")
            tmp_outfile = os.path.join(tmp, "out_text.txt")
            with util.uopen(tmp_infile, "wt") as out:
                sp.call(["zcat", "-f", input_file], stdout=out)
            cmd = [
                sys.executable,
                os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py"),
                "--input",
                tmp_infile,
                "--codes",
                self.bpe_codes.get_path(),
                "--output",
                tmp_outfile,
            ]

            if self.bpe_vocab:
                cmd += ["--vocabulary", self.bpe_vocab.get_path()]

            util.create_executable("apply_bpe.sh", cmd)
            sp.run(cmd, check=True)

            if self.gzip_output:
                with util.uopen(tmp_outfile, "rt") as fin, util.uopen(self.out_bpe_text, "wb") as fout:
                    sp.call(["gzip"], stdin=fin, stdout=fout)
            else:
                shutil.copy(tmp_outfile, self.out_bpe_text.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)

class GetSubwordRatioJob(Job):
    """
    Apply subword codes on a text file and get ratio
    """
    def __init__(
            self,
            text_file: tk.Path,
            vocab: [str | VocabConfig],
            subword_nmt_repo: Optional[tk.Path] = None,
            apply_job: Type[Job] = ApplyBPEToTextJob,
            mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param vocab as str
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        if isinstance(vocab, str):
            vocab = get_vocab_by_str(vocab)
        if apply_job == ApplyBPEToTextJob:
            apply_job: Type[ApplyBPEToTextJob]
            self.subword_text  = apply_job(
                text_file=self.text_file,
                bpe_codes=vocab.codes,
                bpe_vocab=tk.Path(vocab.vocab.get_path() [:-5] + "dummy_count.vocab"), #
                subword_nmt_repo=self.subword_nmt_repo,
                gzip_output=True,
                mini_task=False,
            ).out_bpe_text
        elif apply_job == ApplySentencepieceToTextJob:
            apply_job: Type[ApplySentencepieceToTextJob]
            self.subword_text  = apply_job(
                text_file=self.text_file,
                sentencepiece_model=vocab.model_file,
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

class GetBpeRatioJob(Job):
    """
    Apply BPE codes on a text file
    """
    def __init__(
            self,
            text_file: tk.Path,
            vocab: str,
            subword_nmt_repo: Optional[tk.Path] = None,
            mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param bpe_codes: bpe codes file, e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param bpe_vocab: if provided, then merge operations that produce OOV are reverted,
            use e.g. ReturnnTrainBpeJob.out_bpe_dummy_count_vocab
        :param subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """

        self.text_file = text_file
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        vocab = get_vocab_by_str(vocab)
        self.bpe_text  = ApplyBPEToTextJob(
            text_file=self.text_file,
            bpe_codes=vocab.codes,
            bpe_vocab=tk.Path(vocab.vocab.get_path() [:-5] + "dummy_count.vocab"), #
            subword_nmt_repo=self.subword_nmt_repo,
            gzip_output=True,
            mini_task=False,
        ).out_bpe_text
        self.out_ratio = self.output_var("bpe_to_word_ratio")

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
        total_bpe_tokens = 0
        with util.uopen(self.text_file, "rt") as fin, util.uopen(self.bpe_text, "rt") as fout:
            for orig_line, bpe_line in zip(fin, fout):
                orig_tokens = orig_line.strip().split()
                bpe_tokens = bpe_line.strip().split()
                total_orig_tokens += len(orig_tokens)
                total_bpe_tokens += len(bpe_tokens)

        # avoid division by zero
        if total_orig_tokens > 0:
            self.out_ratio.set(total_bpe_tokens / total_orig_tokens)
        else:
            self.out_ratio.set(1.0)  # fallback to 1.0 if no tokens

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)




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

        self.rqmt: Optional[Dict[str, Any]] = {"cpu": 1, "mem": 6.0, "time": 2.0}

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
    spms = ApplySentencepieceToWordOutputJob(search_py_output=data.output, sentencepiece_model=spm.model_file, enable_unk=False ,gzip_output=True).out_search_results
    return RecogOutput(output=spms)