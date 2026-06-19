"""
Sisyphus job for building the FROZEN high-level k-means target codebook offline.

Runs ``pytorch_networks/two_level/build_kmeans.py`` under the conda python (torch + datasets + sklearn)
on a GPU node: loads the frozen lower stack from the BEST-RQ checkpoint, pools layer-9 features over
``round(25/target_rate)`` frames (v1: target_rate=25 => window=1 => frame-level), and fits MiniBatchKMeans.
Output: ``centroids.npy`` [num_clusters, 512], consumed by ``two_level_v1.Model`` as the ``codebook`` net-arg.
"""

from __future__ import annotations

import json
import os
import subprocess as sp
from typing import Any, Dict, Sequence

from sisyphus import Job, Task, tk

from ...data import datasets as ds
from ...default_tools import I6_MODELS_REPO_PATH, RECIPE_ROOT, RETURNN_EXE, RETURNN_ROOT


class BuildKMeansCodebookJob(Job):
    """Build a frozen k-means codebook from fixed-window-pooled layer-9 features of the frozen encoder."""

    def __init__(
        self,
        *,
        base_checkpoint,
        model_config_dict: Dict[str, Any],
        parquet_files: Sequence[str],
        target_rate_hz: float,
        num_clusters: int,
        python_exe: tk.Path = RETURNN_EXE,
        max_utts: int = 4000,
        max_vectors: int = 1_000_000,  # frame-level: ~1M frames -> ample support for K up to ~512
        seed: int = 42,
    ):
        self.base_checkpoint = base_checkpoint  # i6_core Checkpoint (dependency auto-detected via .path)
        self.model_config_dict = model_config_dict
        self.parquet_files = list(parquet_files)
        self.target_rate_hz = target_rate_hz
        self.num_clusters = num_clusters
        self.python_exe = python_exe
        self.max_utts = max_utts
        self.max_vectors = max_vectors
        self.seed = seed
        self.out_centroids = self.output_path("centroids.npy")
        self.rqmt = {"gpu": 1, "gpu_mem": 24, "mem": 32, "time": 4, "cpu": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with open("config.json", "w") as f:
            json.dump(self.model_config_dict, f)
        script = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "pytorch_networks", "two_level", "build_kmeans.py")
        )
        ckpt = self.base_checkpoint
        ckpt_path = os.fspath(ckpt.path if hasattr(ckpt, "path") else ckpt)

        env = dict(os.environ)
        env["HF_HOME"] = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
        # build_kmeans.py imports the model package -> recipe + i6_models + returnn on PYTHONPATH
        pp = [os.fspath(RECIPE_ROOT), os.fspath(I6_MODELS_REPO_PATH), os.fspath(RETURNN_ROOT)]
        if env.get("PYTHONPATH"):
            pp.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pp)

        cmd = [
            os.fspath(self.python_exe), script,
            "--ckpt", ckpt_path,
            "--config", os.path.abspath("config.json"),
            "--out", self.out_centroids.get_path(),
            "--target-rate", repr(self.target_rate_hz),
            "--num-clusters", str(self.num_clusters),
            "--max-utts", str(self.max_utts),
            "--max-vectors", str(self.max_vectors),
            "--seed", str(self.seed),
            "--parquet", *self.parquet_files,
        ]
        print("RUN:", " ".join(cmd[:12]), "... (+%d parquet files)" % len(self.parquet_files), flush=True)
        sp.check_call(cmd, env=env)


def build_kmeans_codebook(
    *,
    prefix: str,
    base_checkpoint,
    model_config_dict: Dict[str, Any],
    target_rate_hz: float,
    num_clusters: int,
    split,
    returnn_exe: tk.Path = RETURNN_EXE,
    returnn_root: tk.Path = RETURNN_ROOT,  # unused (no RETURNN here); accepted for call-site symmetry
) -> tk.Path:
    """Create the codebook-build job and return the centroids ``tk.Path`` (the model's ``codebook`` arg)."""
    job = BuildKMeansCodebookJob(
        base_checkpoint=base_checkpoint,
        model_config_dict=model_config_dict,
        parquet_files=ds.parquet_files(split),
        target_rate_hz=target_rate_hz,
        num_clusters=num_clusters,
        python_exe=returnn_exe,
    )
    job.add_alias(f"{prefix}/kmeans_k{num_clusters}")
    tk.register_output(f"{prefix}/kmeans_k{num_clusters}_centroids.npy", job.out_centroids)
    return job.out_centroids
