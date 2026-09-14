"""Squashfs-packed venvs: one image file instead of ~60k inodes, mounted node-locally on first use.

A job venv is tens of thousands of small files. On Lustre that costs an inode each and makes a cold
``import torch`` take ~3 min (measured 175-219 s on a fresh node vs 16 s from the image, 2026-09-14).
Packed, the venv root holds:

    venv.sqsh         the whole venv tree (mksquashfs, zstd)
    .sqsh-venv        the launcher (LAUNCHER below)
    bin/<x>           -> ../.sqsh-venv, for every executable of the original bin/
    bin/activate*     copied verbatim (sourced, not executed; they put this bin/ on PATH)
    bin/venv-shell    shell with the venv active: `venv-shell` (interactive) or `venv-shell -c CMD`
    bin/venv-mount    mount if needed, print the mounted root (to browse site-packages etc.)
    bin/venv-unpack   `venv-unpack DEST`: writable copy of the tree, for experiments
    README

Consumers keep calling ``<root>/bin/python`` exactly as before. The launcher mounts the image with
``squashfuse_ll`` -- once per SLURM job under ``$TMPDIR`` (SLURM tears the mount down with it at job
end), or once per login node under ``/tmp/sqsh-venv-<uid>`` -- and execs the real interpreter inside,
so ``sys.prefix`` is the mount. The tree's absolute self-references (console-script shebangs,
``pyvenv.cfg``) name the path the tree was BUILT at, ``<that path>/bin/python``; so that path must keep
resolving to this root, where it lands on the launcher. Audited over all 11 venvs: every other
absolute self-reference is a ``.pyc`` source path, which CPython rewrites at import.

The image is read-only: no ``pip install`` into it. Change a venv by editing its ``packages`` and
rebuilding the ``CreateVenv`` job. ``venv_prefix()`` is the one way to get the real root from a
``bin/python`` path -- ``dirname(dirname(python))`` is the skeleton, which has no ``lib/``.

Stdlib only and py3.9-compatible: ``CreateVenv`` imports it, and it is also a CLI
(``pack``/``verify``) for converting finished venvs in place.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

IMAGE = "venv.sqsh"
LAUNCHER_NAME = ".sqsh-venv"
HELPERS = ("venv-shell", "venv-mount", "venv-unpack")

LAUNCHER = r"""#!/bin/bash
# Launcher for a squashfs-packed venv -- see i6_experiments/users/dorian_koch/jobs/sqsh_venv.py.
# Every bin/<x> is a symlink to this file: it mounts ../venv.sqsh node-locally (once per SLURM job,
# or once per login node) and runs <mount>/bin/<x>.
set -euo pipefail
name=$(basename "$0")
root=$(cd "$(dirname "$0")/.." && pwd -P)
img=$root/venv.sqsh
if [[ $name == venv-unpack ]]; then
  [[ $# -eq 1 ]] || { echo "usage: venv-unpack DEST  (writable copy of the venv tree)" >&2; exit 2; }
  exec unsquashfs -q -d "$1" "$img"
fi
# The path hash keeps two venvs apart even when their roots share a name (every new one is
# `output/venv`) and their images happen to share size and mtime.
key=$(basename "$root")-$(printf %s "$img" | md5sum | cut -c1-8)-$(stat -L -c %Y-%s "$img")
if [[ -n ${SQSH_VENV_MOUNT_BASE:-} ]]; then
  base=$SQSH_VENV_MOUNT_BASE
elif [[ -n ${SLURM_JOB_ID:-} ]]; then
  # Per JOB, never shared across jobs: a job's squashfuse daemon dies with the job. Only SLURM's
  # per-job $TMPDIR is cleaned at job end; a mount anywhere else outlives the job as a dead mount.
  [[ ${TMPDIR:-} == *"$SLURM_JOB_ID"* ]] ||
    echo "sqsh-venv: TMPDIR=${TMPDIR:-unset} is not per-job; this mount will outlive job $SLURM_JOB_ID" >&2
  base=${TMPDIR:-/tmp}/sqsh-venv-job$SLURM_JOB_ID
else
  base=/tmp/sqsh-venv-$(id -u)
fi
mnt=$base/$key
# Live = the daemon answers. statfs always reaches the daemon; a plain `-e` does not -- the kernel
# keeps serving cached attributes after the daemon died, and exec then fails with ENOTCONN.
live() { stat -f "$mnt" >/dev/null 2>&1 && [[ -e $mnt/pyvenv.cfg ]]; }
if ! live; then
  mkdir -p "$base"
  exec 9>"$mnt.lock"
  flock 9
  if ! live; then
    fusermount3 -u -z "$mnt" 2>/dev/null || true  # a mount whose daemon died reads as ENOTCONN
    mkdir -p "$mnt"
    squashfuse_ll "$img" "$mnt" 9>&-  # the daemon must not inherit the lock fd, or it holds it forever
  fi
  exec 9>&-
fi
case $name in
  venv-mount) echo "$mnt" ;;
  venv-shell)
    if [[ $# -gt 0 ]]; then
      exec env VIRTUAL_ENV="$mnt" PATH="$mnt/bin:$PATH" bash "$@"
    fi
    exec env VIRTUAL_ENV="$mnt" bash --rcfile <(printf '[ -f ~/.bashrc ] && . ~/.bashrc\nexport PATH=%q:"$PATH"\nPS1="(%s) $PS1"\n' "$mnt/bin" "${root##*/}")
    ;;
  *) exec "$mnt/bin/$name" "$@" ;;
esac
"""

README = """This venv is packed into venv.sqsh (one file instead of tens of thousands) -- see
i6_experiments/users/dorian_koch/jobs/sqsh_venv.py. Everything under bin/ still works as before;
the first call mounts the image node-locally, which takes well under a second.

  bin/python script.py              run the venv's python (all consumers use this path unchanged)
  bin/venv-shell                    interactive shell with the venv active (python, pip, ... on PATH)
  bin/venv-shell -c 'pip list'      one command with the venv active
  bin/venv-mount                    print the mounted root, e.g. to browse lib/python3.12/site-packages
  bin/venv-unpack /tmp/x            writable copy of the whole tree (then use /tmp/x/bin/python)

The image is read-only: pip install into it fails. To change the venv, edit its packages in the
recipe and rebuild the CreateVenv job.
"""


def is_packed(root: str) -> bool:
    return os.path.isfile(os.path.join(root, IMAGE)) and os.path.isfile(os.path.join(root, LAUNCHER_NAME))


def venv_prefix(python: str) -> str:
    """The real root of the venv whose interpreter is ``python`` (for a packed venv: its mount)."""
    root = os.path.dirname(os.path.dirname(str(python)))
    if not is_packed(root):
        return root
    out = subprocess.run([str(python), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"packed venv {root} did not start: {out.stderr.strip()}")
    return out.stdout.strip()


def create_venv(canonical: str, staging: str | None = None) -> str:
    """An empty venv (with pip) whose files all name ``canonical``; returns ``canonical/bin/python``.

    With ``staging``, the tree is stored there and ``canonical`` becomes a symlink to it, so the
    transient files land on the staging volume. ``venv`` refuses to create THROUGH a symlink, so the
    tree is created at ``staging`` without pip, and pip is then bootstrapped by running
    ``canonical/bin/python`` -- everything that writes a path from then on (ensurepip, every later
    ``pip install``) names the canonical path, which is where ``pack`` puts the launcher.
    """
    import venv

    py = os.path.join(canonical, "bin", "python")
    if not staging:
        venv.create(canonical, with_pip=True)
        return py
    if os.path.isdir(staging):
        shutil.rmtree(staging)  # an earlier failed attempt
    os.makedirs(os.path.dirname(staging), exist_ok=True)
    if os.path.islink(canonical):
        os.remove(canonical)
    elif os.path.isdir(canonical):
        os.rmdir(canonical)  # must be empty: an output dir the job framework pre-created
    venv.create(staging, with_pip=False)
    _repoint(staging, staging, canonical)  # venv wrote the dir it was GIVEN into activate/pyvenv.cfg
    os.symlink(staging, canonical)
    subprocess.run([py, "-m", "ensurepip", "--upgrade", "--default-pip"], check=True)
    return py


def _repoint(tree: str, old: str, new: str) -> None:
    """Replace ``old`` with ``new`` in the files ``venv`` writes the env dir into (not in site-packages)."""
    names = ["pyvenv.cfg"] + [os.path.join("bin", n) for n in os.listdir(os.path.join(tree, "bin"))]
    for name in names:
        p = os.path.join(tree, name)
        if name != "pyvenv.cfg" and not os.path.basename(name).lower().startswith("activate"):
            continue
        with open(p, encoding="utf-8") as f:
            text = f.read()
        if old in text:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text.replace(old, new))


def _processors() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)))


def image_inodes(image: str) -> int:
    out = subprocess.run(["unsquashfs", "-s", image], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Number of inodes (\d+)", out)
    if not m:
        raise RuntimeError(f"cannot read the inode count of {image}:\n{out}")
    return int(m.group(1))


def tree_inodes(tree: str) -> int:
    n = 1  # the root itself, as unsquashfs counts it
    for _, dirs, files in os.walk(tree):
        n += len(dirs) + len(files)
    return n


def build_image(tree: str, image: str, processors: int | None = None) -> None:
    """mksquashfs ``tree`` into ``image``, refusing an image that does not hold every entry."""
    partial = image + ".partial"
    if os.path.exists(partial):
        os.remove(partial)
    cmd = ["mksquashfs", tree, partial, "-comp", "zstd", "-no-xattrs", "-noappend", "-quiet", "-no-progress"]
    cmd += ["-processors", str(processors or _processors())]
    subprocess.run(cmd, check=True)
    want, got = tree_inodes(tree), image_inodes(partial)
    if want != got:
        raise RuntimeError(f"{partial} holds {got} inodes but {tree} has {want} -- not using it")
    os.rename(partial, image)
    print(f"[sqsh_venv] {tree} -> {image}: {got} inodes, {os.path.getsize(image) / 2**30:.2f} GiB", flush=True)


def write_launcher(root: str) -> None:
    """(Re)write ``root``'s launcher -- also how an already-packed venv picks up a launcher fix."""
    launcher = os.path.join(root, LAUNCHER_NAME)
    with open(launcher + ".new", "w") as f:
        f.write(LAUNCHER)
    os.chmod(launcher + ".new", 0o755)
    os.replace(launcher + ".new", launcher)  # atomic: a concurrent launch sees the old or the new file


def write_skeleton(tree: str, root: str) -> None:
    """Populate ``root`` (which already holds the image) with the launcher, bin/ and README."""
    bin_dir = os.path.join(root, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    write_launcher(root)
    names = sorted(os.listdir(os.path.join(tree, "bin")))
    clash = sorted(set(names) & set(HELPERS))
    if clash:
        raise RuntimeError(f"{tree}/bin already has {clash}; the helper names would shadow them")
    for name in names + list(HELPERS):
        dst = os.path.join(bin_dir, name)
        if name.lower().startswith("activate"):  # sourced, not executed: must stay a real file
            shutil.copy2(os.path.join(tree, "bin", name), dst)
        else:
            os.symlink("../" + LAUNCHER_NAME, dst)
    with open(os.path.join(root, "README"), "w") as f:
        f.write(README)


def pack(tree: str, root: str, processors: int | None = None) -> None:
    """Build ``root`` = image + skeleton from ``tree``. ``root`` must not exist yet (or be a stale attempt)."""
    tree, root = os.path.abspath(tree), os.path.abspath(root)
    if os.path.realpath(root) == os.path.realpath(tree):
        raise ValueError("pack() writes a NEW root; it never rewrites the tree it reads")
    if os.path.exists(root):
        if is_packed(root):
            raise FileExistsError(f"{root} is already a packed venv")
        shutil.rmtree(root)  # a half-written earlier attempt of ours
    os.makedirs(root)
    build_image(tree, os.path.join(root, IMAGE), processors)
    write_skeleton(tree, root)


def _freeze(python: str) -> list[str]:
    cmd = [python, "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr}")
    return sorted(out.stdout.split())


def verify(tree: str, root: str) -> None:
    """The packed venv must report the tree's exact package set and start from the mount."""
    py = os.path.join(root, "bin", "python")
    prefix = venv_prefix(py)
    if os.path.realpath(prefix) in (os.path.realpath(root), os.path.realpath(tree)):
        raise RuntimeError(f"{py} started from {prefix}, not from a mount")
    want, got = _freeze(os.path.join(tree, "bin", "python")), _freeze(py)
    if not want or want != got:
        diff = sorted(set(want) ^ set(got))
        raise RuntimeError(f"package sets differ ({len(want)} vs {len(got)}): {diff[:20]}")
    if any(p.startswith("torch==") for p in got):
        subprocess.run([py, "-c", "import torch"], check=True)
    print(f"[sqsh_venv] verified {root}: {len(got)} packages, prefix {prefix}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="TREE -> ROOT (image + skeleton), then verify")
    p.add_argument("tree")
    p.add_argument("root")
    p.add_argument("--processors", type=int)
    v = sub.add_parser("verify", help="compare a packed ROOT against its source TREE")
    v.add_argument("tree")
    v.add_argument("root")
    r = sub.add_parser("refresh", help="rewrite the launcher of already-packed ROOTs")
    r.add_argument("roots", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "refresh":
        for root in args.roots:
            if not is_packed(root):
                raise SystemExit(f"{root} is not a packed venv")
            write_launcher(root)
            print(f"[sqsh_venv] launcher refreshed: {root}", flush=True)
        return
    if args.cmd == "pack":
        pack(args.tree, args.root, args.processors)
    verify(args.tree, args.root)


if __name__ == "__main__":
    main()
