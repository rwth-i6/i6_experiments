import shutil

from sisyphus import *

import os
import subprocess as sp
from typing import Dict, Iterator, List, Optional

from i6_core.util import create_executable, get_executable_path


class EspnetBeamSearchJob(Job):
    """
    Runs beam search using espnet models and data loaders with some given search arguments
    It also allows specifying the beam search algorithm to use

    ref: https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh#L1577
    """

    __sis_hash_exclude__ = {"cpu_rqmt": 2, "cpu_type": None}

    def __init__(
        self,
        beam_search_script: tk.Path,
        data_path: tk.Path,
        search_args: Dict,
        python_exe: tk.Path,
        mem_rqmt: float = 8,
        time_rqmt: float = 4,
        cpu_rqmt: int = 2,
        cpu_type: Optional[str] = None,
    ):
        self.beam_search_script = beam_search_script
        self.data_path = data_path
        self.search_args = search_args
        self.python_exe = python_exe

        self.output_dir = self.output_path("search_output")
        self.log_dir = self.output_path("search_log")
        self.out_hyp = self.output_path("hyp")

        self.rqmt = dict(mem=mem_rqmt, time=time_rqmt)
        self.rqmt["cpu"] = cpu_rqmt
        if cpu_type is not None:
            self.rqmt["cpu_type"] = cpu_type
        if search_args["device"] == "cuda":
            self.rqmt["gpu"] = 1

    def tasks(self) -> Iterator[Task]:
        yield Task("link_data_folder", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def link_data_folder(self):
        os.symlink(os.path.join(self.data_path.get_path(), "data"), "./data")
        os.symlink(os.path.join(self.data_path.get_path(), "downloads"), "./downloads")

    def get_cmd(self) -> List[str]:
        cmd = [
            self.python_exe.get_path(),
            self.beam_search_script.get_path(),
            "--data_path",
            self.data_path.get_path(),
            "--log_dir",
            self.log_dir.get_path(),
            "--output_dir",
            self.output_dir.get_path(),
        ]
        for k, v in self.search_args.items():
            assert v is not None, f"search_args[{k}] is None"
            cmd.append(f"--{k}")
            cmd.append(str(v))
        return cmd

    def run(self):
        cmd = self.get_cmd()
        create_executable("run.sh", cmd)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
        env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
        sp.check_call(cmd, env=env)

        shutil.copy(os.path.join(self.output_dir.get_path(), "1best_recog/text"), self.out_hyp.get_path())

    @classmethod
    def hash(cls, kwargs):
        d = {
            "beam_search_script": kwargs["beam_search_script"],
            "data_path": kwargs["data_path"],
            "search_args": kwargs["search_args"],
            "cpu_rqmt": kwargs["cpu_rqmt"],
            "cpu_type": kwargs["cpu_type"],
        }
        return super().hash(d)


class ConvertHypRefToDict(Job):

    __sis_hash_exclude__ = {"upper_case": False}

    def __init__(self, filename: tk.Path, upper_case: bool = False):
        self.filename = filename
        self.upper_case = upper_case
        self.out_dict = self.output_path("dict")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        res = {}
        with open(self.filename) as f:
            for line in f:
                splits = line.strip().split(" ", 1)
                assert len(splits) <= 2
                if len(splits) == 1:
                    res[splits[0]] = ""
                else:
                    res[splits[0]] = splits[1] if not self.upper_case else splits[1].upper()
        with open(self.out_dict.get_path(), "w") as f:
            f.write("{\n")
            for k, v in res.items():
                f.write(f'"{k}": "{v}",\n')
            f.write("}\n")


class EspnetScliteScoringJob(Job):
    """
    ref: https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh#L1706
    """

    __sis_hash_exclude__ = {"upper_case": False}

    def __init__(self, hyp: tk.Path, ref: tk.Path, sclite_exe: tk.Path, upper_case: bool = False):
        self.hyp = hyp
        self.ref = ref
        self.sclite_exe = sclite_exe
        self.upper_case = upper_case

        self.out_hyp_trn = self.output_path("hyp.trn")
        self.out_ref_trn = self.output_path("ref.trn")
        self.out_wer_report = self.output_path("reports", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def _convert_to_trn(self, filename: tk.Path, out_trn: tk.Path):
        with open(filename.get_path()) as f:
            with open(out_trn.get_path(), "w") as f_out:
                for line in f:
                    splits = line.strip().split(" ", 1)
                    if len(splits) == 1:
                        f_out.write(f"({splits[0]})\n")
                    else:
                        text = splits[1].upper() if self.upper_case else splits[1]
                        f_out.write(f"{text} ({splits[0]})\n")

    def run(self):
        self._convert_to_trn(self.hyp, self.out_hyp_trn)
        self._convert_to_trn(self.ref, self.out_ref_trn)
        sclite_path = os.path.join(self.sclite_exe.get_path(), "sclite")
        call = [
            sclite_path,
            "-r",
            self.out_ref_trn.get_path(),
            "trn",
            "-h",
            self.out_hyp_trn.get_path(),
            "trn",
            "-i",
            "rm",
            "-o",
            "all",
            "-o",
            "dtl",
            "-o",
            "lur",
            "-n",
            "sclite",
            "-O",
            self.out_wer_report.get_path(),
        ]
        sp.check_call(call)

    @classmethod
    def hash(cls, kwargs):
        d = {
            "hyp": kwargs["hyp"],
            "ref": kwargs["ref"],
        }
        return super().hash(d)

class EspnetCalculateRtfJob(Job):

    def __init__(self, log_file: tk.Path, rtf_script: tk.Path):
        self.log_file = log_file
        self.rtf_script = rtf_script
        self.out_rtf = self.output_var("rtf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        cmd = [
            "python3",
            self.rtf_script.get_path(),
            self.log_file.get_path(),
        ]
        process = sp.Popen(cmd, stdout=sp.PIPE)
        res, _ = process.communicate()
        rtf = float(res.decode("utf-8").strip())
        self.out_rtf.set(rtf)