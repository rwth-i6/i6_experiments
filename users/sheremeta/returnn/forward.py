"""Packed RETURNN forward job: several independent forwards in one allocation, one GPU each."""

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from sisyphus import Job, Task, tk
from sisyphus import global_settings as gs

__all__ = ["PackedReturnnForwardJob", "pack_names", "visible_device"]


def pack_names(names: Sequence[str], max_per_job: int) -> List[List[str]]:
    assert max_per_job > 0, max_per_job
    names = list(names)
    return [names[i : i + max_per_job] for i in range(0, len(names), max_per_job)]


def visible_device(index: int, allocation: Optional[str]) -> str:
    devices = [d for d in (allocation or "").split(",") if d]
    if not devices:
        return str(index)
    assert index < len(devices), (index, devices)
    return devices[index]


class PackedReturnnForwardJob(Job):
    """Run several forward configs side by side, one GPU each, exposing every forward's output files by name."""

    def __init__(
        self,
        *,
        forwards: Dict[str, Dict[str, Any]],
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        log_verbosity: int = 5,
        time_rqmt: float = 24,
        mem_rqmt_per_forward: float = 16,
        cpu_rqmt_per_forward: int = 2,
    ):
        assert forwards, "no forwards to pack"
        self.forwards = forwards
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.log_verbosity = log_verbosity
        self.out_config_files = {name: self.output_path(f"{name}/returnn.config") for name in forwards}
        self.out_files = {
            name: {f: self.output_path(f"{name}/{f}") for f in spec["output_files"]} for name, spec in forwards.items()
        }
        count = len(forwards)
        self.rqmt = {
            "gpu": count,
            "cpu": cpu_rqmt_per_forward * count,
            "mem": mem_rqmt_per_forward * count,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def _returnn_config(cls, spec: Dict[str, Any], log_verbosity: int):
        from i6_core.returnn.forward import ReturnnForwardJobV2

        return ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=spec["model_checkpoint"],
            returnn_config=spec["returnn_config"],
            log_verbosity=log_verbosity,
            device="gpu",
        )

    def create_files(self):
        for name, spec in self.forwards.items():
            path = self.out_config_files[name].get_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._returnn_config(spec, self.log_verbosity).write(path)

    def run(self):
        exe = tk.uncached_path(self.returnn_python_exe)
        rnn = os.path.join(tk.uncached_path(self.returnn_root), "rnn.py")
        allocation = os.environ.get("CUDA_VISIBLE_DEVICES")
        threads = max(1, self.rqmt["cpu"] // len(self.forwards))
        running = {}
        for index, name in enumerate(self.forwards):
            work_dir = tempfile.mkdtemp(prefix=f"{gs.TMP_PREFIX}packed_{name}_")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible_device(index, allocation)
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            log = open(f"returnn_{name}.log", "wt")
            proc = subprocess.Popen(
                [exe, rnn, self.out_config_files[name].get_path()],
                cwd=work_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            print(f"started {name} on device {env['CUDA_VISIBLE_DEVICES']} in {work_dir}")
            running[name] = (proc, work_dir, log)

        failed = []
        for name, (proc, work_dir, log) in running.items():
            code = proc.wait()
            log.close()
            if code != 0:
                crash_dir = f"crash_dir_{name}"
                if os.path.exists(crash_dir):
                    shutil.rmtree(crash_dir)
                shutil.copytree(work_dir, crash_dir, dirs_exist_ok=True)
                failed.append((name, code))
                continue
            for file_name, out in self.out_files[name].items():
                src = os.path.join(work_dir, file_name)
                assert os.path.exists(src), f"{name}: output file {file_name} does not exist"
                os.makedirs(os.path.dirname(out.get_path()), exist_ok=True)
                shutil.move(src, out.get_path())
            shutil.copytree(work_dir, os.path.join("logs", name), dirs_exist_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)
        assert not failed, f"forwards failed with exit codes {failed}, see crash_dir_<name>"

    @classmethod
    def hash(cls, kwargs):
        log_verbosity = kwargs.get("log_verbosity", 5)
        d = {
            "forwards": {name: cls._returnn_config(spec, log_verbosity) for name, spec in kwargs["forwards"].items()},
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)
