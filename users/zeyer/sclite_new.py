from typing import Optional, List, Dict, Tuple
from sisyphus import Job, Task, tk, gs
import tempfile
import subprocess as sp
import os
import re


class ScliteJob(Job):
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
        precision_ndigit: int = 1,
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

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

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
                assert "Percent Total Error" in outputs_absolute, "Expected Percent Total Error"
                outputs_absolute["Percent Word Accuracy"] = 100 - outputs_absolute["Percent Total Error"]

                outputs_percentage: Dict[str, float] = {}
                num_ref_words = outputs_absolute["Ref. words"]
                for key, absolute in outputs_absolute.items():
                    if num_ref_words > 0:
                        percentage = 100.0 * absolute / num_ref_words
                    else:
                        percentage = float("nan")
                    outputs_percentage[key] = round(percentage, self.precision_ndigit)

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
