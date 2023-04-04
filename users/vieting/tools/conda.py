__all__ = ["InstallMinicondaJob", "CreateCondaEnvJob"]

import os.path
import subprocess as sp
from typing import Optional, List, Dict

from sisyphus import *
from i6_core.util import create_executable


class InstallMinicondaJob(Job):
    """
    Download and install miniconda.
    """

    def __init__(self, version: str = "Miniconda3-latest-Linux-x86_64.sh"):
        self.version = version
        self.out_conda_dir = self.output_path("miniconda", directory=True)
        self.out_conda_exe = self.output_path("conda_exe.sh")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        sp.check_call(["wget", f"https://repo.anaconda.com/miniconda/{self.version}"])
        sp.check_call(["bash", self.version, "-b", "-p", self.out_conda_dir.get(), "-f"])
        create_executable(
            self.out_conda_exe.get(),
            [f'PATH="{os.path.join(self.out_conda_dir.get(), "bin")}:$PATH"', "&&",
             "source", os.path.join(self.out_conda_dir.get(), "bin/activate"), "base", "&&",
             "conda", "$*"],
        )


class CreateCondaEnvJob(Job):
    """
    Create a new environment given an existing conda installation.
    """

    def __init__(self, conda_exe: Path, python_version: Optional[str] = None, channels: Optional[List[str]] = None,
                 packages: Optional[Dict[str, str]] = None):
        self.conda_exe = conda_exe
        self.python_version = python_version
        self.channels = channels or []
        self.packages = packages or {}
        self.env_name = self.hash({
            "conda_exe": conda_exe,
            "python_version": python_version,
            "channels": channels,
            "packages": packages,
        })
        self.out_env_exe = self.output_path("conda_env.sh")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        if self.packages:
            with open("packages", "w") as f:
                f.write("\n".join([f"{package}{version}" for package, version in self.packages.items()]))

        args = [self.conda_exe.get(), "create", "--name", self.env_name, "-q", "-y"]
        if self.python_version is not None:
            args += [f"python={self.python_version}"]
        for channel in self.channels:
            args += ["-c", channel]
        if self.packages:
            args += ["--file", "packages"]
        print(" ".join(["$"] + args))
        sp.run(args, check=True)

        create_executable(
            self.out_env_exe.get(),
            ["source", self.conda_exe.get(), "activate", self.env_name, "&&", "python3", "$*"],
        )
