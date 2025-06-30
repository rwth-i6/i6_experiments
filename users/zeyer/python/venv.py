"""
Create a Python virtual environment
"""

from typing import Optional, Union, Tuple, List
from sisyphus import Job, Task, tk


class CreatePythonVirtualEnvJob(Job):
    """
    Create a Python virtual environment
    """

    def __init__(
        self,
        *,
        python: Optional[Union[str, tk.Path]] = None,
        python_version: Tuple[int, int],
        requirements: List[str],
    ):
        """
        :param python: If None (default), will use "pythonX.Y" where X.Y is the version.
            If str, will search in the PATH for the python executable
            (and it will check if it is the right version).
            If tk.Path, will use the path as is
            (and it will check if it is the right version).
        :param python_version: the version of Python to use.
        :param requirements:
        """
        super().__init__()
        if python is None:
            python = f"python{python_version[0]}.{python_version[1]}"
        self.python = python
        self.python_version = python_version
        self.requirements = requirements

        self.out_dir = self.output_path("venv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from subprocess import check_call as call, check_output
        from ast import literal_eval

        py = self.python
        if isinstance(py, tk.Path):
            py = py.get_path()

        # Check Python version.
        py_version = literal_eval(
            check_output([py, "-c", "import sys; print(sys.version_info[:2])"]).decode("utf8").strip()
        )
        assert py_version == self.python_version

        # Create virtual env.
        call([py, "-m", "venv", self.out_dir.get_path()])

        # Check virtual env / check Python binary in it / check version.
        py = self.out_dir.get_path() + "/bin/python"
        py_version = literal_eval(
            check_output([py, "-c", "import sys; print(sys.version_info[:2])"]).decode("utf8").strip()
        )
        assert py_version == self.python_version

        call([py, "-m", "pip", "install"] + self.requirements)
