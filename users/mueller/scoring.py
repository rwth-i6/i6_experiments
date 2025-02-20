from __future__ import annotations

import os
import logging
import re
import tempfile
import subprocess as sp
from typing import Optional, Dict, List, Tuple


import sisyphus
from sisyphus import tk, gs

import i6_core.util as util
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from .datasets.librispeech import _get_corpus_text_dict


class ComputeWERJob(sisyphus.Job):
    """
    Computes WER using the calculate-word-error-rate.py tool from RETURNN
    """

    def __init__(self, hypothesis, reference, returnn_python_exe=None, returnn_root=None):
        """

        :param Path hypothesis: python-style search output from RETURNN (e.g. SearchBPEtoWordsJob)
        :param Path reference: python-style text dictionary (use e.g. i6_core.corpus.convert.CorpusToTextDictJob)
        :param str|Path returnn_python_exe: RETURNN python executable
        :param str|Path returnn_root: RETURNN source root
        """
        self.hypothesis = hypothesis
        self.reference = reference

        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        self.out_wer = self.output_var("wer")
        
        self.rqmt = {"cpu": 16, "mem": 40, "time": 12}

    def run(self):
        call = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(str(self.returnn_root), "tools/calculate-word-error-rate.py"),
            "--expect_full",
            "--hyps",
            self.hypothesis.get_path(),
            "--refs",
            self.reference.get_path(),
            "--out",
            self.out_wer.get_path(),
        ]
        logging.info("run %s" % " ".join(call))
        sp.check_call(call)

    def tasks(self):
        yield sisyphus.Task("run", rqmt=self.rqmt)

# ---------------------------------------------------

def _score_recog(dataset: DatasetConfig, recog_output: RecogOutput, alias_name: str = None):
    """score"""
    # We use sclite now.
    # Could also use ReturnnComputeWERJob.
    from i6_core.returnn.search import SearchWordsDummyTimesToCTMJob
    from i6_core.text.convert import TextDictToStmJob
    from i6_core.recognition.scoring import ScliteJob
    from .datasets.task import ScoreResult

    hyp_words = recog_output.output
    corpus_name = dataset.get_main_name()

    corpus_text_dict = _get_corpus_text_dict(corpus_name)
    # Arbitrary seg length time. The jobs SearchWordsDummyTimesToCTMJob and TextDictToStmJob
    # serialize two points after decimal, so long seqs (>1h or so) might be problematic,
    # and no reason not to just use a high value here to avoid this problem whenever we get to it.
    seg_length_time = 1000.0
    search_ctm = SearchWordsDummyTimesToCTMJob(
        recog_words_file=hyp_words, seq_order_file=corpus_text_dict, seg_length_time=seg_length_time
    ).out_ctm_file
    stm_file = TextDictToStmJob(text_dict=corpus_text_dict, seg_length_time=seg_length_time).out_stm_path

    score_job = CustomScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )
    
    if alias_name:
        score_job.add_alias(alias_name)

    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)

class CustomScliteJob(sisyphus.Job):
    """
    Run the Sclite scorer from the SCTK toolkit

    Outputs:
        - out_report_dir: contains the report files with detailed scoring information
        - out_*: the job also outputs many variables, please look in the init code for a list
    """

    __sis_hash_exclude__ = {"sctk_binary_path": None, "precision_ndigit": 1}

    def __init__(
        self,
        ref: tk.Path,
        hyp: tk.Path,
        cer: bool = False,
        sort_files: bool = False,
        additional_args: Optional[List[str]] = None,
        sctk_binary_path: Optional[tk.Path] = None,
        precision_ndigit: Optional[int] = 1,
    ):
        """
        :param ref: reference stm text file
        :param hyp: hypothesis ctm text file
        :param cer: compute character error rate
        :param sort_files: sort ctm and stm before scoring
        :param additional_args: additional command line arguments passed to the Sclite binary call
        :param sctk_binary_path: set an explicit binary path.
        :param precision_ndigit: number of digits after decimal point for the precision
            of the percentages in the output variables.
            If None, no rounding is done.
            In sclite, the precision was always one digit after the decimal point
            (https://github.com/usnistgov/SCTK/blob/f48376a203ab17f/src/sclite/sc_dtl.c#L343),
            thus we recalculate the percentages here.
        """
        self.set_vis_name("Sclite - %s" % ("CER" if cer else "WER"))

        self.ref = ref
        self.hyp = hyp
        self.cer = cer
        self.sort_files = sort_files
        self.additional_args = additional_args
        self.sctk_binary_path = sctk_binary_path
        self.precision_ndigit = precision_ndigit

        self.out_report_dir = self.output_path("reports", True)

        self.out_wer = self.output_var("wer")
        self.out_num_errors = self.output_var("num_errors")
        self.out_percent_correct = self.output_var("percent_correct")
        self.out_num_correct = self.output_var("num_correct")
        self.out_percent_substitution = self.output_var("percent_substitution")
        self.out_num_substitution = self.output_var("num_substitution")
        self.out_percent_deletions = self.output_var("percent_deletions")
        self.out_num_deletions = self.output_var("num_deletions")
        self.out_percent_insertions = self.output_var("percent_insertions")
        self.out_num_insertions = self.output_var("num_insertions")
        self.out_percent_word_accuracy = self.output_var("percent_word_accuracy")
        self.out_ref_words = self.output_var("ref_words")
        self.out_hyp_words = self.output_var("hyp_words")
        self.out_aligned_words = self.output_var("aligned_words")
        
        self.rqmt = {"cpu": 16, "mem": 40, "time": 12}

    def tasks(self):
        if self.rqmt:
            yield sisyphus.Task("run", resume="run", rqmt=self.rqmt)
        else:
            yield sisyphus.Task("run", resume="run", mini_task=True)

    def run(self, output_to_report_dir=True):
        if self.sort_files:
            sort_stm_args = ["sort", "-k1,1", "-k4,4n", self.ref.get_path()]
            (fd_stm, tmp_stm_file) = tempfile.mkstemp(suffix=".stm")
            res = sp.run(sort_stm_args, stdout=sp.PIPE)
            os.write(fd_stm, res.stdout)
            os.close(fd_stm)

            sort_ctm_args = ["sort", "-k1,1", "-k3,3n", self.hyp.get_path()]
            (fd_ctm, tmp_ctm_file) = tempfile.mkstemp(suffix=".ctm")
            res = sp.run(sort_ctm_args, stdout=sp.PIPE)
            os.write(fd_ctm, res.stdout)
            os.close(fd_ctm)

        if self.sctk_binary_path:
            sclite_path = os.path.join(self.sctk_binary_path.get_path(), "sclite")
        else:
            sclite_path = os.path.join(gs.SCTK_PATH, "sclite") if hasattr(gs, "SCTK_PATH") else "sclite"
        output_dir = self.out_report_dir.get_path() if output_to_report_dir else "."
        stm_file = tmp_stm_file if self.sort_files else self.ref.get_path()
        ctm_file = tmp_ctm_file if self.sort_files else self.hyp.get_path()

        args = [
            sclite_path,
            "-r",
            stm_file,
            "stm",
            "-h",
            ctm_file,
            "ctm",
            "-o",
            "all",
            "-o",
            "dtl",
            "-o",
            "lur",
            "-n",
            "sclite",
            "-O",
            output_dir,
        ]
        if self.cer:
            args.append("-c")
        if self.additional_args is not None:
            args += self.additional_args

        sp.check_call(args)

        if output_to_report_dir:  # run as real job
            with open(f"{output_dir}/sclite.dtl", "rt", errors="ignore") as f:
                # Example:
                """
                Percent Total Error       =    5.3%   (2709)
                ...
                Percent Word Accuracy     =   94.7%
                ...
                Ref. words                =           (50948)
                """

                # key -> percentage, absolute
                output_variables: Dict[str, Tuple[Optional[tk.Variable], Optional[tk.Variable]]] = {
                    "Percent Total Error": (self.out_wer, self.out_num_errors),
                    "Percent Correct": (self.out_percent_correct, self.out_num_correct),
                    "Percent Substitution": (self.out_percent_substitution, self.out_num_substitution),
                    "Percent Deletions": (self.out_percent_deletions, self.out_num_deletions),
                    "Percent Insertions": (self.out_percent_insertions, self.out_num_insertions),
                    "Percent Word Accuracy": (self.out_percent_word_accuracy, None),
                    "Ref. words": (None, self.out_ref_words),
                    "Hyp. words": (None, self.out_hyp_words),
                    "Aligned words": (None, self.out_aligned_words),
                }

                outputs_absolute: Dict[str, int] = {}
                for line in f:
                    key: Optional[str] = ([key for key in output_variables if line.startswith(key)] or [None])[0]
                    if not key:
                        continue
                    pattern = rf"^{re.escape(key)}\s*=\s*((\S+)%)?\s*(\(\s*(\d+)\))?$"
                    m = re.match(pattern, line)
                    assert m, f"Could not parse line: {line!r}, does not match to pattern r'{pattern}'"
                    absolute_s = m.group(4)
                    if not absolute_s:
                        assert not output_variables[key][1], f"Expected absolute value for {key}"
                        continue
                    outputs_absolute[key] = int(absolute_s)
                    if key == "Aligned words":
                        break  # that should be the last key, can stop now

                assert "Ref. words" in outputs_absolute, "Expected absolute numbers for Ref. words"
                num_ref_words = outputs_absolute["Ref. words"]
                assert "Percent Total Error" in outputs_absolute, "Expected absolute numbers for Percent Total Error"
                outputs_absolute["Percent Word Accuracy"] = num_ref_words - outputs_absolute["Percent Total Error"]

                outputs_percentage: Dict[str, float] = {}
                for key, absolute in outputs_absolute.items():
                    if num_ref_words > 0:
                        percentage = 100.0 * absolute / num_ref_words
                    else:
                        percentage = float("nan")
                    outputs_percentage[key] = (
                        round(percentage, self.precision_ndigit) if self.precision_ndigit is not None else percentage
                    )

                for key, (percentage_var, absolute_var) in output_variables.items():
                    if percentage_var is not None:
                        assert key in outputs_percentage, f"Expected percentage value for {key}"
                        percentage_var.set(outputs_percentage[key])
                    if absolute_var is not None:
                        assert key in outputs_absolute, f"Expected absolute value for {key}"
                        absolute_var.set(outputs_absolute[key])

    def calc_wer(self):
        wer = None

        old_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            self.run(output_to_report_dir=False)
            dtl_file = "sclite.dtl"
            with open(dtl_file, "rt", errors="ignore") as f:
                for line in f:
                    if line.startswith("Percent Total Error"):
                        errors = float("".join(line.split()[5:])[1:-1])
                    if line.startswith("Ref. words"):
                        wer = 100.0 * errors / float("".join(line.split()[3:])[1:-1])
                        break
        os.chdir(old_dir)

        return wer