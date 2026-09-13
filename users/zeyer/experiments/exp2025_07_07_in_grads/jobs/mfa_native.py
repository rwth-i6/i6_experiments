"""
Native MFA (Montreal Forced Aligner) environment on a host without conda-forge builds of all its deps
(linux-aarch64: no kalpy, baumwelch, ngram, sox packages).

:class:`BuildMfaNativeEnvJob`:
micromamba (static binary),
a conda-forge env (python, kaldi, openfst, pynini, MFA's python deps, compilers),
kalpy compiled from its PyPI sdist against the conda Kaldi (``KALDI_ROOT``),
MFA from PyPI without dependency resolution.
baumwelch and ngram are only used for G2P and LM training, sox only for non-wav input,
none of which ``mfa align`` on wav files needs.
Runs on the login node (mini_task: internet, compilers).
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
import urllib.request
from typing import List, Optional

from sisyphus import Job, Task, tk

__all__ = ["BuildMfaNativeEnvJob", "MfaNativeExeWrapperJob"]

# the conda-forge dependencies of montreal-forced-aligner 3.4.2,
# minus the packages without linux-aarch64 builds (baumwelch, kalpy, ngram, sox),
# plus what the kalpy build needs
_default_conda_packages = [
    "python=3.12",
    "kaldi=5.5.1172",
    "openfst",
    "pynini",
    "pybind11",
    "cmake",
    "ninja",
    "cxx-compiler",
    "c-compiler",
    "pip",
    "setuptools",
    "setuptools_scm",
    "wheel",
    "ffmpeg",
    "hdbscan",
    "huggingface_hub",
    "jinja2",
    "kneed",
    "librosa",
    "numba",
    "numpy",
    "pgvector",
    "pgvector-python",
    "postgresql",
    "praatio>=6.0.0",
    "psycopg2",
    "pysoundfile",
    "pyyaml",
    "requests",
    "rich",
    "rich-click",
    "scikit-learn",
    "sqlalchemy>=2.0",
    "sqlite",
    "tqdm",
]


class BuildMfaNativeEnvJob(Job):
    """conda-forge env + kalpy source build + MFA, see the module docstring. ``out_mfa`` is the ``mfa`` executable."""

    def __init__(
        self,
        *,
        micromamba_url: str = "https://micro.mamba.pm/api/micromamba/linux-aarch64/2.9.0",
        conda_packages: Optional[List[str]] = None,
        kalpy_pip_spec: str = "kalpy-kaldi==0.10.5",
        mfa_pip_spec: str = "montreal-forced-aligner==3.4.2",
    ):
        """
        :param micromamba_url: tar.bz2 with bin/micromamba (pin the version, the URL is hashed)
        :param conda_packages: conda-forge specs, default :data:`_default_conda_packages`
        :param kalpy_pip_spec: sdist to compile (no aarch64 wheel)
        :param mfa_pip_spec: installed with ``--no-deps``
        """
        super().__init__()
        self.micromamba_url = micromamba_url
        self.conda_packages = list(conda_packages) if conda_packages is not None else list(_default_conda_packages)
        self.kalpy_pip_spec = kalpy_pip_spec
        self.mfa_pip_spec = mfa_pip_spec
        self.out_env = self.output_path("env", directory=True)
        self.out_mfa = self.output_path("env/bin/mfa")
        self.rqmt = {"cpu": 8, "mem": 16, "time": 3}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, mini_task=True)

    def run(self):
        root = os.path.join(os.getcwd(), "micromamba")
        os.makedirs(os.path.join(root, "bin"), exist_ok=True)
        with urllib.request.urlopen(self.micromamba_url, timeout=300) as r:
            with tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:bz2") as tar:
                tar.extract("bin/micromamba", root, filter="fully_trusted")
        mm = os.path.join(root, "bin", "micromamba")
        env_dir = self.out_env.get_path()
        # the package cache stays in the job dir (not $HOME)
        env = dict(
            os.environ,
            MAMBA_ROOT_PREFIX=root,
            CONDA_PKGS_DIRS=os.path.join(root, "pkgs"),
            # the env's cmake, ninja, compilers for the kalpy build (pip runs it in a subprocess)
            PATH=os.path.join(env_dir, "bin") + os.pathsep + os.environ.get("PATH", ""),
        )
        subprocess.check_call([mm, "create", "-y", "-p", env_dir, "-c", "conda-forge", *self.conda_packages], env=env)
        py = os.path.join(env_dir, "bin", "python")
        build_env = dict(
            env,
            KALDI_ROOT=env_dir,
            CMAKE_PREFIX_PATH=env_dir,
            CC=os.path.join(env_dir, "bin", "cc"),
            CXX=os.path.join(env_dir, "bin", "c++"),
            CMAKE_BUILD_PARALLEL_LEVEL=str(self.rqmt["cpu"]),
        )
        subprocess.check_call(
            [py, "-m", "pip", "install", "--no-build-isolation", "-v", self.kalpy_pip_spec], env=build_env
        )
        subprocess.check_call([py, "-c", "import _kalpy, kalpy; print('kalpy ok:', kalpy.__file__)"], env=env)
        subprocess.check_call([py, "-m", "pip", "install", "--no-deps", self.mfa_pip_spec], env=env)
        subprocess.check_call([self.out_mfa.get_path(), "version"], env=env)


class MfaNativeExeWrapperJob(Job):
    """``mfa`` wrapper that puts the env's ``bin`` on ``PATH`` first
    (MFA finds the OpenFST / Kaldi binaries via ``PATH``, the plain entry point does not set it)."""

    def __init__(self, *, env_dir: tk.Path):
        """
        :param env_dir: :attr:`BuildMfaNativeEnvJob.out_env`
        """
        super().__init__()
        self.env_dir = env_dir
        self.out_exe = self.output_path("mfa")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        env_bin = os.path.join(self.env_dir.get_path(), "bin")
        with open(self.out_exe.get_path(), "w") as f:
            f.write(
                f'#!/usr/bin/env bash\nset -euo pipefail\nexport PATH="{env_bin}:$PATH"\nexec "{env_bin}/mfa" "$@"\n'
            )
        os.chmod(self.out_exe.get_path(), 0o755)
