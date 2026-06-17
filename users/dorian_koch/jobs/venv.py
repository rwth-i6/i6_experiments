import venv
from sisyphus import Job, Task
import os
import subprocess


class CreateVenv(Job):
    def __init__(self, *, packages: list[list[str]] | None = None, hash_overwrite: str | None = None):
        self.out_env_path = self.output_path("venv", directory=True)
        self.out_python_path = self.output_var("venv_python_path")
        self.packages = packages if packages is not None else []
        self.hash_overwrite = hash_overwrite

    @classmethod
    def hash(cls, parsed_args):
        if parsed_args.get("hash_overwrite") is not None:
            return parsed_args["hash_overwrite"]
        d = dict(**parsed_args)
        # d["__version"] = 5
        return super().hash(d)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        venv.create(self.out_env_path, with_pip=True)
        if os.name == "nt":  # Windows
            venv_python = os.path.join(self.out_env_path, "Scripts", "python.exe")
        else:  # Mac/Linux
            venv_python = os.path.join(self.out_env_path, "bin", "python")

        # Install dependencies if specified
        if self.packages:
            for pkgs in self.packages:
                subprocess.check_call([venv_python, "-m", "pip", "install"] + pkgs)

        self.out_python_path.set(venv_python)
