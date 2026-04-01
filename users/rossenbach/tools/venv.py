import os
import subprocess as sp
from typing import Optional, Union

from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase

from i6_core.util import instanciate_delayed


class CreatePythonVEnvV2Job(Job):
    """
    Creates a python virtualenv and installs pip-packages in it.

    Differences between this job and :class:`CreatePythonVEnvJob`: installs packages in parallel.
    In turn, this allows for better package version management,
    since `pip` now resolves the package versions in the same call.

    In practice, the only difference is that the list of packages is now of the form
    `[["parallel_install_group1==version", ...], ["parallel_install_group2==version", ...]]`.
    Try to use this job whenever possible.
    """

    def __init__(
        self,
        packages: list[list[Union[str, DelayedBase]]],
        python_binary: Union[str, tk.Path] = "/usr/bin/python3",
        venv_extra_args: Optional[list[str]] = None,
    ):
        """
        :param packages: list of packages including version_string (e.g. "==0.5") or extra options
            all elements of each of the inner lists are installed simultaneously
        :param python_binary: python binary to use for venv creation
        :param venv_extra_args: additional arguments passed to the venv cmd-line utility
        """
        self.packages = packages
        self.python_binary = python_binary
        self.venv_extra_args = venv_extra_args or []

        self.out_venv = self.output_path("venv", directory=True)
        self.out_python_bin = self.output_path(
            "venv/bin/python3"
        )  # convenience output for Jobs that take a python binary
        self.out_python_site_pkg = self.output_path(
            "site-packages"
        )  # symlink to site-packages dir (for inclusion in sys.path)

        self.rqmt = {"cpu":2, "mem": 4, "time": 2}

    def tasks(self):
        # no mini-task, to definitely use container
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        sp.check_call([self.python_binary, "-m", "venv", self.out_venv.get_path()] + self.venv_extra_args)

        pip = os.path.join(self.out_venv.get_path(), "bin/pip3")
        sp.check_call([pip, "install", "-U", "pip"])
        sp.check_call([pip, "install", "wheel"])
        for pkgs in self.packages:
            sp.check_call([pip, "install"] + instanciate_delayed(pkgs))

        lib_dir = os.path.join(self.out_venv.get_path(), "lib")
        python_dir = [p for p in os.listdir(lib_dir) if p.startswith("python")][0]
        pkg_dir = os.path.join(lib_dir, python_dir, "site-packages")
        os.symlink(pkg_dir, self.out_python_site_pkg.get_path())