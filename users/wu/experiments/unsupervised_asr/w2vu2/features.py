"""SAE §1c audio side — sisyphus jobs producing fairseq's `ExtractedFeaturesDataset` layout.

Both jobs are thin wrappers around `dump_w2vu2_data.py` (the `KMeansUnitsJob` pattern: the Job
declares outputs and writes the call; the torch work lives in a sibling worker run under the conda
python). Verified end-to-end: fairseq's own ExtractedFeaturesDataset loads the output with features
and km targets aligned 1:1.
"""

from __future__ import annotations

import os
import subprocess as sp
from typing import Dict, Optional

from sisyphus import Job, Task, gs, tk

_alias_prefix = "sae/1c"

DEFAULT_PYTHON_EXE = tk.Path(
    getattr(gs, "RETURNN_PYTHON_EXE", "/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python"),
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)


def _worker(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _env() -> dict:
    """Recipe roots reconstructed from __file__ so the subprocess resolves them regardless of env."""
    here = os.path.dirname(os.path.abspath(__file__))
    recipe_root = os.path.abspath(os.path.join(here, *([".."] * 5)))  # .../recipe
    pp = [
        recipe_root,
        os.path.join(recipe_root, "i6_models"),
        os.path.join(recipe_root, "2025-10-speech-llm", "src"),
        os.path.join(recipe_root, "returnn"),
    ]
    env = dict(os.environ)
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.setdefault("HF_HOME", "/e/project1/spell/common_hf_home")
    return env


class MergeW2vu2DataJob(Job):
    """Collect per-split dumps into the single directory fairseq's `task.data` expects.

    Symlinks, not copies: the .npy is ~8 GB per split and fairseq only ever mmaps it read-only.
    """

    def __init__(self, *, splits: Dict[str, tk.Path]):
        super().__init__()
        self.splits = dict(splits)  # fairseq split name -> that split's dump dir
        self.out_dir = self.output_path("data", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dst = self.out_dir.get_path()
        for name, d in sorted(self.splits.items()):
            src_dir = d.get_path()
            for ext in ("npy", "lengths", "km", "ids", "stats.txt"):
                src = os.path.join(src_dir, f"{name}.{ext}")
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                link = os.path.join(dst, f"{name}.{ext}")
                if not os.path.exists(link):
                    os.symlink(src, link)
        print("merged:", sorted(os.listdir(dst)), flush=True)


class MfccKmeansJob(Job):
    """64-cluster k-means over Kaldi MFCC+d+dd — the encoder-independent aux target of w2v-U 2.0.

    CPU-only: MFCC comes from the waveform, not the encoder. Hyperparameters copied from HuBERT's
    `learn_kmeans.py`, which fairseq's `prepare_audio_v2.sh` invokes.
    """

    def __init__(
        self,
        *,
        hf_data_dir: tk.Path,
        split: str = "train",
        num_clusters: int = 64,
        mfcc_downsample: int = 4,
        trim_silence: bool = True,
        max_fit_vectors: int = 2_000_000,
        seed: int = 0,
        limit: Optional[int] = None,
        python_exe: tk.Path = DEFAULT_PYTHON_EXE,
    ):
        super().__init__()
        self.hf_data_dir = hf_data_dir
        self.split = split
        self.num_clusters = num_clusters
        self.mfcc_downsample = mfcc_downsample
        self.trim_silence = trim_silence
        self.max_fit_vectors = max_fit_vectors
        self.seed = seed
        self.limit = limit
        self.python_exe = python_exe

        self.out_centroids = self.output_path("mfcc_centroids.npy")
        self.out_stats = self.output_path("mfcc_kmeans.stats.txt")
        self.rqmt = {"cpu": 8, "mem": 48, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        cmd = [
            os.fspath(self.python_exe), _worker("dump_w2vu2_data.py"),
            "--mode", "fit-mfcc",
            "--hf-data-dir", self.hf_data_dir.get_path(),
            "--split", self.split,
            "--num-clusters", str(self.num_clusters),
            "--mfcc-downsample", str(self.mfcc_downsample),
            "--max-fit-vectors", str(self.max_fit_vectors),
            "--seed", str(self.seed),
            "--out-centroids", self.out_centroids.get_path(),
            "--out-stats", self.out_stats.get_path(),
        ]
        if self.trim_silence:
            cmd.append("--trim-silence")
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        print("RUN:", " ".join(cmd), flush=True)
        sp.check_call(cmd, env=_env())


class W2vu2FeatureDumpJob(Job):
    """Frozen BEST-RQ layer-l features -> {name}.npy (fp16) / .lengths / .km, VAD-trimmed.

    `out_dir` is the directory fairseq's `task.data` points at; `name` is the fairseq split name
    (train / valid), which need not equal the HF split.
    """

    def __init__(
        self,
        *,
        hf_data_dir: tk.Path,
        split: str,
        name: str,
        mfcc_centroids: tk.Path,
        encoder_layer: int = 5,
        trim_silence: bool = True,
        mfcc_downsample: int = 4,
        min_length: int = 3,
        limit: Optional[int] = None,
        python_exe: tk.Path = DEFAULT_PYTHON_EXE,
        time_rqmt: float = 8,
    ):
        super().__init__()
        self.hf_data_dir = hf_data_dir
        self.split = split
        self.name = name
        self.mfcc_centroids = mfcc_centroids
        self.encoder_layer = encoder_layer
        self.trim_silence = trim_silence
        self.mfcc_downsample = mfcc_downsample
        self.min_length = min_length
        self.limit = limit
        self.python_exe = python_exe

        self.out_dir = self.output_path("data", directory=True)
        self.out_npy = self.output_path(f"data/{name}.npy")
        self.out_lengths = self.output_path(f"data/{name}.lengths")
        self.out_km = self.output_path(f"data/{name}.km")
        self.out_ids = self.output_path(f"data/{name}.ids")
        self.out_stats = self.output_path(f"data/{name}.stats.txt")
        self.rqmt = {"gpu": 1, "mem": 32, "time": time_rqmt, "cpu": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        cmd = [
            os.fspath(self.python_exe), _worker("dump_w2vu2_data.py"),
            "--mode", "dump",
            "--hf-data-dir", self.hf_data_dir.get_path(),
            "--split", self.split,
            "--encoder-layer", str(self.encoder_layer),
            "--mfcc-downsample", str(self.mfcc_downsample),
            "--min-length", str(self.min_length),
            "--mfcc-centroids", self.mfcc_centroids.get_path(),
            "--out-npy", self.out_npy.get_path(),
            "--out-lengths", self.out_lengths.get_path(),
            "--out-km", self.out_km.get_path(),
            "--out-ids", self.out_ids.get_path(),
            "--out-stats", self.out_stats.get_path(),
        ]
        if self.trim_silence:
            cmd.append("--trim-silence")
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        print("RUN:", " ".join(cmd), flush=True)
        sp.check_call(cmd, env=_env())
