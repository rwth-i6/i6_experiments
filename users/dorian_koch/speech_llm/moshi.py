from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
)
from .common import HF_CACHE_DIR
from pathlib import Path
from sisyphus import Job, Task, tk
import os
import subprocess
import json
import shutil
from i6_experiments.users.dorian_koch.jobs.hf import HfMergeShards
import sys
import moshi_finetune  # needed to get moshi_finetune path for PYTHONPATH below

# None of this is used anywhere i think


def download_moshi():
    # projects/moshi/moshi/moshi/models/loaders.py
    # untested code...

    repo = DownloadHuggingFaceRepoJob(model_id="kyutai/moshiko-pytorch-bf16")
    repo.out_hub_cache_dir = HF_CACHE_DIR
    return repo.out_hub_cache_dir


def moshi_inference_server(model):
    pass


class MergeMoshiAnnotationsViaSymlinks(Job):
    def __init__(self, in_annotations: list[tk.Path]):
        self.in_annotations = in_annotations
        self.out_merged = self.output_path("merged_annotations", directory=True)

    def tasks(self):
        yield Task("merge", mini_task=True)

    def merge(self):
        os.makedirs(self.out_merged.get(), exist_ok=True)
        out_jsonl = os.path.join(self.out_merged.get(), "annotations.jsonl")

        dir_map = {}
        dir_idx = 0

        with open(out_jsonl, "w") as out_f:
            for idx, in_path in enumerate(self.in_annotations):
                # read all .jsonl files\
                for file in os.listdir(in_path.get()):
                    if not file.endswith(".jsonl"):
                        continue
                    src = os.path.join(in_path.get(), file)
                    with open(src) as f:
                        for line in f:
                            data = json.loads(line)
                            path = data["path"]
                            path = os.path.dirname(path) # just get the folder the file is stored in
                            if path not in dir_map:
                                dir_map[path] = f"dir{dir_idx}"
                                dir_idx += 1
                                os.symlink(path, os.path.join(self.out_merged.get(), dir_map[path]), target_is_directory=True)
                            data["path"] = os.path.join(self.out_merged.get(), dir_map[path], os.path.basename(data["path"]))
                            out_f.write(json.dumps(data) + "\n")
                        
                

# runs moshi annotate.py
class MoshiAnnotate(Job):
    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_hf: tk.Path,
        shard: int | None = None,
        num_shards: int | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.in_hf = in_hf
        self.shard = shard
        self.num_shards = num_shards
        self.out_annotations = self.output_path("annotations", directory=True)
        self.rqmt = {
            "gpu": 1,
            "cpu": 6,
            "mem": 8,
            "time": 4,
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs) -> tk.Path:
        assert "in_hf" in kwargs, "Sharding only supported for HF input"
        if num_shards == 1:
            return MoshiAnnotate(**kwargs).out_annotations
        assert num_shards > 1
        shards = []
        for shard in range(num_shards):
            shards.append(
                MoshiAnnotate(shard=shard, num_shards=num_shards, **kwargs)
            )
        return MergeMoshiAnnotationsViaSymlinks(
            in_annotations=[s.out_annotations for s in shards]
        ).out_merged

    def run(self):
        this_file_path = Path(__file__).resolve()
        moshi_annotate_path = this_file_path.parent / "moshi_annotate_inference.py"

        work_dir = os.path.join(os.getcwd(), "annotate_inference_workdir")
        os.makedirs(work_dir, exist_ok=True)

        command = [
            self.venv_python_path.get(),
            str(moshi_annotate_path),
            "--in_hf",
            str(self.in_hf.get()),
        ]
        # if self.out_dir is not None:
        #   command += ["--out_dir", str(self.out_dir.get())]
        # else:
        command += ["--out_dir", str(self.out_annotations.get())]
        if self.shard is not None and self.num_shards is not None:
            command += [
                "--in_hf_shard",
                str(self.shard),
                "--in_hf_num_shards",
                str(self.num_shards),
            ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HOME"] = HF_CACHE_DIR.get()
        top_level_file = sys.modules["moshi_finetune"].__file__
        assert top_level_file is not None, "Could not find moshi_finetune module file"
        package_base_dir = str(Path(top_level_file).parent.parent)
        env["PYTHONPATH"] = (
            f"{package_base_dir}{os.pathsep}{env['PYTHONPATH']}"
            if "PYTHONPATH" in env
            else package_base_dir
        )

        print("Env:")
        for k, v in env.items():
            if k not in ["PYTHONPATH"]:
                continue
            print(f"{k}: {v}")

        print(
            f"Running Moshi annotate with command: {' '.join(command)}",
            flush=True,
        )
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
        subprocess.run(command, env=env, check=True)


class MoshiFinetune(Job):
    def __init__(self, venv_python_path: tk.AbstractPath, train_data: tk.Path):
        self.train_data = train_data
        self.venv_python_path = venv_python_path
        self.out_config = self.output_path("config.yaml")
        self.out_rundir = self.output_path("run_dir", directory=True)
        self.rqmt = {
            "gpu": 1,
            "cpu": 6,
            "mem": 24,
            "time": 23,
        }

    def tasks(self):
        yield Task("write_config", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def write_config(self):
        run_dir = self.out_rundir.get()
        if os.path.exists(run_dir) and os.listdir(run_dir):
            print(f"Warning: run_dir {run_dir} already exists and is not empty.")
            # find dir in cwd to move existing contents to
            new_dir = os.path.join(os.getcwd(), "moshi_finetune_old_runs")
            os.makedirs(new_dir, exist_ok=True)
            cand = os.path.join(new_dir, "0001")
            while os.path.exists(cand):
                cand = os.path.join(new_dir, f"{int(os.path.basename(cand))+1:04d}")
            print(f"Moving existing contents to {cand}")
            os.rename(run_dir, cand)

        txt = f"""
# data
data:
  eval_data: '' # Optional Fill
  shuffle: true
  train_data: '{self.train_data.get()}' # Fill

# model
moshi_paths: 
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

full_finetuning: false # Activate lora.enable if partial finetuning
lora:
  enable: true # Set to False if full_finetuning is True
  rank: 128
  scaling: 2.
  ft_embed: false # Optional, set to True if you want to finetune the embedding layer

first_codebook_weight_multiplier: 100.
text_padding_weight: .5

# optim
duration_sec: 100
batch_size: 16
max_steps: 2000
gradient_checkpointing: true
optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: 0
log_freq: 1
eval_freq: 100
do_eval: false
do_ckpt: true
ckpt_freq: 100
overwrite_run_dir: true

save_adapters: true # Must be False if full_finetuning is True

run_dir: "{run_dir}"  # Fill
"""
        with open(self.out_config, "w") as f:
            f.write(txt)

    def run(self):
        command = [
            self.venv_python_path.get(),
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            str(self.rqmt["gpu"]),
            "-m",
            "moshi_finetune.train",
            self.out_config.get(),
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HOME"] = HF_CACHE_DIR.get()
        top_level_file = sys.modules["moshi_finetune"].__file__
        assert top_level_file is not None, "Could not find moshi_finetune module file"
        package_base_dir = f"{str(Path(top_level_file).parent.parent)}{os.pathsep}{str(Path(top_level_file).parent)}"

        env["PYTHONPATH"] = (
            f"{package_base_dir}{os.pathsep}{env['PYTHONPATH']}"
            if "PYTHONPATH" in env
            else package_base_dir
        )
        print(
            f"Running Moshi training with command: {' '.join(command)}",
            flush=True,
        )
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
        subprocess.run(command, env=env, check=True)
