"""Generic Apptainer/Singularity helper jobs: pull a Docker/OCI image into a tracked ``.sif``
artifact, and wrap a containerized command as a plain executable so any job that takes a
configurable executable (RETURNN-style) can transparently use a containerized tool.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

from sisyphus import Job, Task, tk


class PullApptainerImageJob(Job):
    """Pull a Docker/OCI image into a local Apptainer/Singularity ``.sif`` (a tracked, reusable
    Sisyphus artifact).

    Generic -- works for any image. Runs as a ``mini_task`` so it executes on the manager/login
    node (which has internet access + apptainer), not a (possibly network-less) compute node.
    """

    def __init__(self, image_uri: str):
        """
        :param image_uri: e.g. ``"docker://mmcauliffe/montreal-forced-aligner:latest"``.
        """
        super().__init__()
        self.image_uri = image_uri
        self.out_image = self.output_path("image.sif")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # Keep apptainer's build/layer cache inside the job dir so it does not fill $HOME.
        cache = os.path.join(os.getcwd(), "apptainer_cache")
        os.makedirs(cache, exist_ok=True)
        env = dict(os.environ, APPTAINER_CACHEDIR=cache, APPTAINER_TMPDIR=cache)
        subprocess.check_call(["apptainer", "pull", "--force", self.out_image.get_path(), self.image_uri], env=env)


class ApptainerExeWrapperJob(Job):
    """Write an executable wrapper script that runs a command inside an Apptainer image.

    The wrapper forwards its args: ``wrapper <args...>`` ->
    ``apptainer exec [--bind ...] [--env ...] <image> <command> <args...>``.

    This lets any job that accepts a configurable executable path treat a containerized tool exactly
    like a native binary. ``apptainer exec`` auto-binds ``$HOME``, ``$PWD``, ``/tmp``; pass ``bind``
    only for paths outside those (the caller's work dir / outputs are usually already covered).

    Bind/data sanity check: if ``$WRAP_DEBUG_DIR`` is set in the wrapper's environment at run time, it
    first inspects that directory (plus its parents and the container's mount table) FROM INSIDE the
    container before running the real command. This surfaces a bind/visibility problem directly
    ("does the container actually see the data?") instead of as a downstream tool error. It is inert
    unless the var is set, so it never affects normal runs.
    """

    # v1: run.sh gained an inert (env-gated) WRAP_DEBUG_DIR bind/data inspection block. Bump so the
    # already-emitted wrapper scripts are regenerated.
    __sis_version__ = 1

    def __init__(
        self,
        image: tk.Path,
        command: str = "",
        *,
        bind: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        :param image: the ``.sif`` from :class:`PullApptainerImageJob`.
        :param command: the in-container command, e.g. ``"mfa"``. Empty = the image entrypoint.
        :param bind: extra ``--bind`` paths (host[:container]).
        :param env: extra ``--env KEY=VALUE`` pairs.
        """
        super().__init__()
        self.image = image
        self.command = command
        self.bind = bind or []
        self.env = env or {}
        self.out_exe = self.output_path("run.sh")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        binds = "".join(f" --bind {b}" for b in self.bind)
        envs = "".join(f" --env {k}={v}" for k, v in self.env.items())
        exec_prefix = f'apptainer exec{binds}{envs} "{self.image.get_path()}"'
        # $WRAP_DEBUG_DIR (set in the wrapper's env at run time, expanded by THIS host shell and passed
        # as a plain argv to ls inside the container) -> dump the container's own view of that dir, its
        # parent, its grandparent, and the mount table. A dropped bind shows up here as an empty grand-
        # parent / missing mount, instead of as the tool's cryptic "directory does not exist".
        debug = (
            'if [ -n "${WRAP_DEBUG_DIR:-}" ]; then\n'
            '  echo "=== WRAP_DEBUG: container view of $WRAP_DEBUG_DIR ===" >&2\n'
            f'  {exec_prefix} ls -la "$WRAP_DEBUG_DIR" >&2 || echo "WRAP_DEBUG: ls of dir failed" >&2\n'
            f'  {exec_prefix} ls -la "$(dirname "$WRAP_DEBUG_DIR")" >&2 || true\n'
            f'  {exec_prefix} ls -la "$(dirname "$(dirname "$WRAP_DEBUG_DIR")")" >&2 || true\n'
            f"  {exec_prefix} cat /proc/self/mountinfo >&2 || true\n"
            '  echo "=== WRAP_DEBUG: end ===" >&2\n'
            "fi\n"
        )
        body = f'exec {exec_prefix} {self.command} "$@"'
        with open(self.out_exe.get_path(), "w") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n" + debug + body + "\n")
        os.chmod(self.out_exe.get_path(), 0o755)
