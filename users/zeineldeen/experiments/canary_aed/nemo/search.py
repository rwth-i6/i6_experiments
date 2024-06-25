from sisyphus import *
import subprocess as sp
import os

from typing import Dict, Any, Optional

from i6_core.util import create_executable


class SearchJob(Job):
    def __init__(
        self,
        model_id: str,
        model_path: tk.Path,
        dataset_path: tk.Path,
        dataset_name: str,
        cache_dir_name_suffix: Optional[str],
        split: str,
        search_script: tk.Path,
        search_args: Optional[Dict[str, Any]] = None,
        python_exe: Optional[tk.Path] = None,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: int = 4,
        cpu_rqmt: int = 2,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.cache_dir_name_suffix = cache_dir_name_suffix
        self.split = split
        self.search_script = search_script
        self.search_args = search_args if search_args is not None else {}
        self.python_exe = python_exe
        self.device = device
        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

        self.out_search_results = self.output_path("search_results")
        self.out_wer = self.output_var("wer")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def get_cmd(self):
        cmd = [
            self.python_exe.get_path(),
            self.search_script.get_path(),
            "--model_id",
            self.model_id,
            "--model_path",
            self.model_path.get_path(),
            "--dataset_path",
            self.dataset_path.get_path(),
            "--dataset",
            self.dataset_name,
            "--split",
            self.split,
            "--manifest_path",
            self.out_search_results.get_path(),
            "--device",
            "0" if self.device == "gpu" else "-1",
            "--wer_out_path",
            self.out_wer.get_path(),
        ]
        if self.cache_dir_name_suffix:
            cmd.append("--cache_dir_name_suffix")
            cmd.append(self.cache_dir_name_suffix)
        for k, v in self.search_args.items():
            if k == "device":
                continue  # ignored. this is only set via job parameter
            cmd.append(f"--{k}")
            cmd.append(str(v))
        return cmd

    def create_files(self):
        create_executable("run.sh", self.get_cmd())

    def run(self):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
        env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
        sp.check_call(self.get_cmd(), env=env)

    @classmethod
    def hash(cls, kwargs):
        d = {
            "model_id": kwargs["model_id"],
            "model_path": kwargs["model_path"],
            "dataset_path": kwargs["dataset_path"],
            "dataset_name": kwargs["dataset_name"],
            "cache_dir_name_suffix": kwargs["cache_dir_name_suffix"],
            "split": kwargs["split"],
            "search_script": kwargs["search_script"],
            "search_args": kwargs["search_args"],
            "python_exe": kwargs["python_exe"],
        }
        return super().hash(d)
