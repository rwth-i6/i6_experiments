from glob import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from sisyphus import Job, Task, tk
import os
import signal
import sys
import subprocess
from .common import HF_CACHE_DIR
from contextlib import contextmanager
import shutil
import time
import random
import socket

class ChatterboxInference(Job):
    def __init__(
        self,
        *,
        chatterbox_venv_python_path: tk.Path,
        in_text: tk.Path,
    ):
        self.in_text = in_text
        self.chatterbox_venv_python_path = chatterbox_venv_python_path
        self.out_dir = self.output_path("chatterbox_output", directory=True)
        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 2,
        }
        
    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        #d["__version"] = 5
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        
        this_file_path = Path(__file__).resolve()
        tts_script_path = this_file_path.parent / "chatterbox_inference.py"

        command = [
            self.chatterbox_venv_python_path.get(),
            str(tts_script_path),
            "--in_text", str(self.in_text.get()),
            "--out_dir", str(self.out_dir),
        ]
        env = os.environ.copy()
        env["HF_HOME"] = HF_CACHE_DIR

        print(f"Running Chatterbox inference with command: {' '.join(command)}")
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
        subprocess.run(command, env=env, check=True)