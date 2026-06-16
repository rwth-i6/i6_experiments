from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
)
from .common import HF_CACHE_DIR, job_progress_fraction
from .finetune import (
    MOSHI_ADAPTER,
    finetune_completed_fraction,
    launch_training,
    write_finetune_config,
)
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
                            path = os.path.dirname(path)  # just get the folder the file is stored in
                            if path not in dir_map:
                                dir_map[path] = f"dir{dir_idx}"
                                dir_idx += 1
                                os.symlink(
                                    path, os.path.join(self.out_merged.get(), dir_map[path]), target_is_directory=True
                                )
                            data["path"] = os.path.join(
                                self.out_merged.get(), dir_map[path], os.path.basename(data["path"])
                            )
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
        self.out_hf = self.output_path("out_hf", directory=True)
        # Keep out_annotations as alias for backward compat with callers that haven't updated
        self.out_annotations = self.out_hf
        self.rqmt = {
            "gpu": 1,
            "cpu": 6,
            "mem": 16,
            "time": 4,
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        # moshi_annotate_inference.py writes progress.json into the job work dir.
        return job_progress_fraction(self)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2  # bumped: arrow-native output format
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs) -> tk.Path:
        assert "in_hf" in kwargs, "Sharding only supported for HF input"
        if num_shards == 1:
            return MoshiAnnotate(**kwargs).out_hf
        assert num_shards > 1
        shards = []
        for shard in range(num_shards):
            shards.append(MoshiAnnotate(shard=shard, num_shards=num_shards, **kwargs))
        return HfMergeShards(shard_paths=[s.out_hf for s in shards]).out_hf

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
        command += ["--out_dir", str(self.out_hf.get())]
        command += ["--mode", "arrow"]
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
            f"{package_base_dir}{os.pathsep}{env['PYTHONPATH']}" if "PYTHONPATH" in env else package_base_dir
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


class SplitAnnotatedDataset(Job):
    """Deterministic train/val split of an annotated HF dataset by hashing a key column.

    Rows whose stable content hash of ``split_key`` falls below ``val_fraction`` go to
    the val split, the rest to train. The hash is content-based (not row order), so the
    split is reproducible and independent of shard layout. Only the key column is read
    and audio columns are copied as encoded bytes (no torchcodec decode), so this is a
    light CPU job. Output features (Audio columns, alignments struct) are preserved, so
    the arrow loader reads both splits unchanged.
    """

    def __init__(
        self,
        dataset: tk.Path,
        val_fraction: float = 0.05,
        split_key: str = "id",
        seed: int = 42,
    ):
        self.dataset = dataset
        self.val_fraction = val_fraction
        self.split_key = split_key
        self.seed = seed
        self.out_train = self.output_path("train", directory=True)
        self.out_val = self.output_path("val", directory=True)
        self.rqmt = {"cpu": 4, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import hashlib
        from datasets import load_from_disk

        ds = load_from_disk(self.dataset.get())
        keys = ds[self.split_key]  # single-column read; no audio decode

        def in_val(k) -> bool:
            h = hashlib.md5(f"{self.seed}:{k}".encode()).hexdigest()
            return (int(h[:8], 16) % 1_000_000) / 1_000_000.0 < self.val_fraction

        val_idx = [i for i, k in enumerate(keys) if in_val(k)]
        train_idx = [i for i, k in enumerate(keys) if not in_val(k)]
        assert val_idx and train_idx, "empty split; check val_fraction/split_key"
        print(
            f"[split] {len(keys)} rows -> train {len(train_idx)} / val {len(val_idx)} "
            f"({100 * len(val_idx) / len(keys):.1f}% val) by hash({self.split_key})",
            flush=True,
        )
        ds.select(train_idx).save_to_disk(self.out_train.get())
        ds.select(val_idx).save_to_disk(self.out_val.get())


class MoshiFinetune(Job):
    # New optional params are dropped from the hash when left at these legacy
    # defaults, so existing finetunes (constructed without them) keep their hash
    # and are not re-run. See sisyphus/job.py Job.hash + __sis_hash_exclude__.
    __sis_hash_exclude__ = {
        "duration_sec": 100,
        "audio_jitter_sec": 0.0,
        "num_epochs": None,
        "max_steps": 2000,
        "eval_data": None,
        "lora_rank": 128,
    }

    def __init__(
        self,
        venv_python_path: tk.AbstractPath,
        train_data: tk.Path,
        seed: int = 0,
        duration_sec: int = 100,
        audio_jitter_sec: float = 0.0,
        num_epochs: int | None = None,
        max_steps: int = 2000,
        eval_data: tk.Path | None = None,
        lora_rank: int = 128,
    ):
        self.train_data = train_data
        self.eval_data = eval_data
        self.venv_python_path = venv_python_path
        self.seed = seed
        self.duration_sec = duration_sec
        self.audio_jitter_sec = audio_jitter_sec
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.lora_rank = lora_rank
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

    def completed_fraction(self):
        return finetune_completed_fraction(self, MOSHI_ADAPTER)

    def write_config(self):
        write_finetune_config(self, MOSHI_ADAPTER)

    def run(self):
        launch_training(self, MOSHI_ADAPTER)
