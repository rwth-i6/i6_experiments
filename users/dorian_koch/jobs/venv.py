from sisyphus import Job, Task, gs
import os
import shutil
import subprocess

from i6_experiments.users.dorian_koch.jobs import sqsh_venv


class CreateVenv(Job):
    """A pip venv, shipped as a squashfs image (one file instead of ~60k -- see ``sqsh_venv``).

    ``run`` (login node: pip needs internet) builds the tree at ``output/venv``, through a symlink
    into ``gs.VENV_STAGING_DIR`` when the settings define one, so the transient files never land on
    the job's own volume. ``pack`` (a CPU job) replaces it with image + launcher skeleton at the same
    path and deletes the tree. ``out_python_path`` is ``output/venv/bin/python`` in both layouts, so
    consumers cannot tell them apart; for interactive use see ``output/venv/README``.

    Tasks are not hashed: finished venvs are untouched by this, and were converted in place with
    ``python sqsh_venv.py pack`` (2026-09-14).
    """

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
        yield Task("run", mini_task=True)  # pip needs the login node's internet
        yield Task("pack", rqmt={"cpu": 8, "mem": 8, "time": 2})

    def run(self):
        env = self.out_env_path.get_path()
        staging_root = getattr(gs, "VENV_STAGING_DIR", None)
        staging = os.path.join(staging_root, os.path.basename(self.job_id())) if staging_root else None
        venv_python = sqsh_venv.create_venv(env, staging)
        for pkgs in self.packages:
            subprocess.check_call([venv_python, "-m", "pip", "install"] + pkgs)
        self.out_python_path.set(venv_python)

    def pack(self):
        env = self.out_env_path.get_path()
        if sqsh_venv.is_packed(env):
            return  # a resubmission after a successful pack
        tree = os.path.realpath(env)
        packed = env + ".packed"
        sqsh_venv.pack(tree, packed)
        sqsh_venv.verify(tree, packed)
        if os.path.islink(env):
            os.remove(env)
        else:
            tree = env + ".tree"
            os.rename(env, tree)
        os.rename(packed, env)
        shutil.rmtree(tree)
