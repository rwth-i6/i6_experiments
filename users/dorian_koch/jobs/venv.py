from sisyphus import Job, Task, gs
import os
import shutil
import subprocess
import tempfile

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

    def _sis_setup_directory(self, force=False):
        # A hard kill mid-install (SIGKILL, OOM, a manager restart) skips run()'s finally and leaves output/venv a
        # symlink into a node-local /tmp tree that is gone; Sisyphus's makedirs(output/venv) then raises and takes the
        # manager down. Setup runs only before a (re)submission, never while the build runs.
        env = self.out_env_path.get_path()
        if os.path.islink(env) and not os.path.exists(env):
            os.unlink(env)
        return super()._sis_setup_directory(force=force)

    def tasks(self):
        yield Task("run", mini_task=True)  # pip needs the login node's internet
        yield Task("pack", rqmt={"cpu": 8, "mem": 8, "time": 2})

    def run(self):
        env = self.out_env_path.get_path()
        # gs.VENV_STAGING_LOCAL (a NODE-LOCAL dir on the login node, e.g. "/tmp"): build the tree there and pack it in
        # this same task, then delete it no matter what. The shared VENV_STAGING_DIR on Lustre leaked: a failed install
        # or a cancelled pack left its ~80-150k-file tree behind, and two such trees pushed /hpcwork/tt201262 over its
        # inode quota on 2026-10-01 (1060K of 1024K). A node-local tree costs no quota and cannot outlive the task.
        local_root = getattr(gs, "VENV_STAGING_LOCAL", None)
        base = None
        if local_root:
            base = tempfile.mkdtemp(prefix=f"venv-{os.path.basename(self.job_id())}-", dir=local_root)
            staging = os.path.join(base, "venv")
        else:
            staging_root = getattr(gs, "VENV_STAGING_DIR", None)
            staging = os.path.join(staging_root, os.path.basename(self.job_id())) if staging_root else None
        try:
            venv_python = sqsh_venv.create_venv(env, staging)
            for pkgs in self.packages:
                subprocess.check_call([venv_python, "-m", "pip", "install"] + pkgs)
            if base is not None:
                self.pack()  # the tree exists on THIS node only: image it before the task ends
            self.out_python_path.set(venv_python)
        finally:
            if base is not None:
                shutil.rmtree(base, ignore_errors=True)
                # A failed build leaves output/venv as a symlink into the deleted tree. Sisyphus's job setup then dies
                # in os.makedirs(output/venv) (FileExistsError) and takes the MANAGER down with it (2026-10-01).
                if os.path.islink(env) and not os.path.exists(env):
                    os.unlink(env)

    def pack(self):
        env = self.out_env_path.get_path()
        if sqsh_venv.is_packed(env):
            return  # a resubmission after a successful pack
        tree = os.path.realpath(env)
        packed = env + ".packed"
        if os.path.lexists(packed):
            # env is not packed yet, so this is an earlier attempt that failed after packing (e.g. in verify);
            # sqsh_venv.pack refuses a packed root, which made every rerun die with FileExistsError
            shutil.rmtree(packed)
        sqsh_venv.pack(tree, packed)
        # verify under a throwaway mount base, unmounted after: its squashfuse daemon would otherwise stay mounted on
        # the login node under a key no consumer computes again, pinning the image after a rebuild deletes it
        mount_base = tempfile.mkdtemp(prefix="sqsh-verify-")
        saved = os.environ.get("SQSH_VENV_MOUNT_BASE")
        os.environ["SQSH_VENV_MOUNT_BASE"] = mount_base
        try:
            sqsh_venv.verify(tree, packed)
        finally:
            if saved is None:
                os.environ.pop("SQSH_VENV_MOUNT_BASE", None)
            else:
                os.environ["SQSH_VENV_MOUNT_BASE"] = saved
            for name in os.listdir(mount_base):
                mnt = os.path.join(mount_base, name)
                if os.path.isdir(mnt) and not name.endswith(".lock"):
                    subprocess.run(["fusermount3", "-u", "-z", mnt], check=False)
            shutil.rmtree(mount_base, ignore_errors=True)
        if os.path.islink(env):
            os.remove(env)
        else:
            tree = env + ".tree"
            os.rename(env, tree)
        os.rename(packed, env)
        shutil.rmtree(tree)
