"""Guard: squashfs-packed venvs (jobs/sqsh_venv.py) behave like the tree they replaced.

Login node, ~30 s, needs FUSE (squashfuse_ll, fusermount3). Builds a REAL venv the way
``CreateVenv.run`` does -- through a symlink into a staging dir -- packs it with the same functions
``CreateVenv.pack`` and the in-place conversion call, swaps it in, and then asserts:

  * console-script shebangs name the canonical path, not the staging dir (else they dangle once the
    staging tree is deleted);
  * ``bin/python``, a console script via its shebang, ``bin/activate`` and the three helpers
    (venv-shell / venv-mount / venv-unpack) all work, and ``sys.prefix`` is the mount;
  * ``venv_prefix`` finds the real root, and ``common.add_cuda_npp_to_env`` -- the one consumer that
    builds a path into the venv -- finds a library inside the image;
  * ``verify`` rejects a root whose package set differs from its tree (non-vacuous);
  * four concurrent COLD launches share one mount -- and a launcher that lets the squashfuse daemon
    inherit the lock fd (the bug the prototype had) deadlocks that same test, so it cannot pass
    vacuously;
  * a mount whose daemon died is recovered by the next launch;
  * inside SLURM the mount goes to a per-JOB dir under $TMPDIR (a mount shared across jobs would die
    with the first job's daemon).
"""

import concurrent.futures
import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path[:0] = [RECIPE, os.path.join(RECIPE, "sisyphus")]  # jobs/__init__ imports sisyphus

from i6_experiments.users.dorian_koch.jobs import sqsh_venv as sv  # noqa: E402

failures: list[str] = []


def check(cond, msg):
    print(f"[{'ok' if cond else 'FAIL'}] {msg}", flush=True)
    if not cond:
        failures.append(msg)


def run(cmd, env=None, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


def fuse_mounts_under(prefix):
    with open("/proc/mounts") as f:
        return [ln.split()[1] for ln in f if "squashfuse" in ln and ln.split()[1].startswith(prefix)]


def unmount_under(prefix):
    for m in fuse_mounts_under(prefix):
        subprocess.run(["fusermount3", "-u", "-z", m], capture_output=True)


def cold_launches(py, n, timeout):
    def one(_):
        try:
            r = run([py, "-c", "import sys; print(sys.prefix)"], timeout=timeout)
            return r.stdout.strip() if r.returncode == 0 else f"rc={r.returncode} {r.stderr[-200:]}"
        except subprocess.TimeoutExpired:
            return "TIMEOUT"

    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        return list(ex.map(one, range(n)))


tmp = tempfile.mkdtemp(prefix="check_sqsh_venv_", dir="/tmp")
mnt_base = os.path.join(tmp, "mnt")
os.environ["SQSH_VENV_MOUNT_BASE"] = mnt_base  # keep test mounts out of the real /tmp/sqsh-venv-<uid>
os.environ.pop("SLURM_JOB_ID", None)
try:
    # -- build like CreateVenv.run: through a symlink into a staging dir ----------------------------
    staging = os.path.join(tmp, "staging", "CreateVenv.test")
    canon = os.path.join(tmp, "job", "output", "venv")
    os.makedirs(os.path.dirname(canon))
    os.makedirs(canon)  # as if the job framework pre-created the output dir
    sv.create_venv(canon, staging)
    check(os.path.islink(canon) and os.path.realpath(canon) == staging, "the tree lives in staging, reached via canon")
    site = glob.glob(os.path.join(staging, "lib", "python*", "site-packages"))[0]
    os.makedirs(os.path.join(site, "nvidia", "npp", "lib"))  # something add_cuda_npp_to_env must find
    with open(os.path.join(canon, "bin", "pip")) as f:
        head = f.read(400)
    check(canon in head and staging not in head, "console-script shebangs name the canonical path, not staging")
    with open(os.path.join(canon, "bin", "activate")) as f:
        act_src = f.read()
    check(canon in act_src and staging not in act_src, "bin/activate names the canonical path, not staging")

    # -- pack + verify with the real functions ----------------------------------------------------
    packed = canon + ".packed"
    sv.pack(staging, packed, processors=1)
    check(sv.is_packed(packed), "pack() produced image + launcher")
    check(not os.path.islink(os.path.join(packed, "bin", "activate")), "bin/activate is a real file (it is sourced)")
    sv.verify(staging, packed)
    check(True, "verify() accepts the root built from its own tree")

    other = os.path.join(tmp, "other-tree")
    shutil.copytree(staging, other, symlinks=True)
    other_site = glob.glob(os.path.join(other, "lib", "python*", "site-packages"))[0]
    os.makedirs(os.path.join(other_site, "fakepkg-1.0.dist-info"))
    with open(os.path.join(other_site, "fakepkg-1.0.dist-info", "METADATA"), "w") as f:
        f.write("Metadata-Version: 2.1\nName: fakepkg\nVersion: 1.0\n")
    try:
        sv.verify(other, packed)
        check(False, "verify() rejects a root whose package set differs from the tree")
    except RuntimeError:
        check(True, "verify() rejects a root whose package set differs from the tree")

    # -- swap like CreateVenv.pack; the canonical path now reaches the launcher --------------------
    os.remove(canon)
    os.rename(packed, canon)
    shutil.rmtree(staging)  # the tree is gone, as after a real pack
    py = os.path.join(canon, "bin", "python")
    prefix = run([py, "-c", "import sys; print(sys.prefix)"]).stdout.strip()
    check(prefix.startswith(mnt_base + "/"), f"bin/python starts from the mount ({prefix})")
    pip = run([os.path.join(canon, "bin", "pip"), "--version"])
    check(pip.returncode == 0 and prefix in pip.stdout, "a console script runs via its shebang, from the mount")
    check(run([os.path.join(canon, "bin", "venv-mount")]).stdout.strip() == prefix, "venv-mount prints the mount")
    sh = run([os.path.join(canon, "bin", "venv-shell"), "-c", 'echo "$VIRTUAL_ENV"; command -v python'])
    check(sh.stdout.split() == [prefix, prefix + "/bin/python"], "venv-shell -c runs with the venv active")
    act = run(["bash", "-c", f"source {canon}/bin/activate && python -c 'import sys; print(sys.prefix)'"])
    check(act.stdout.strip() == prefix, "source bin/activate, then python, reaches the mount")
    unpacked = os.path.join(tmp, "unpacked")
    run([os.path.join(canon, "bin", "venv-unpack"), unpacked])
    upy = os.path.join(unpacked, "bin", "python")
    got = run([upy, "-c", "import sys; print(sys.prefix)"]).stdout.strip()
    check(got == unpacked and os.access(site.replace(staging, unpacked), os.W_OK), "venv-unpack gives a writable tree")

    # -- venv_prefix and the consumer that needs it -----------------------------------------------
    check(sv.venv_prefix(py) == prefix, "venv_prefix(packed) is the mount")
    check(sv.venv_prefix(upy) == unpacked, "venv_prefix(plain tree) is its root")
    from i6_experiments.users.dorian_koch.speech_llm.common import add_cuda_npp_to_env

    e = {}
    add_cuda_npp_to_env(py, e)
    check(e.get("LD_LIBRARY_PATH", "").startswith(prefix + "/lib/"), "add_cuda_npp_to_env finds NPP inside the image")

    # -- concurrency: 4 cold launches, one mount --------------------------------------------------
    unmount_under(mnt_base)
    res = cold_launches(py, 4, timeout=60)
    check(res == [prefix] * 4, f"4 concurrent cold launches all start ({res})")
    check(fuse_mounts_under(prefix) == [prefix], "... and share ONE mount")

    twin = os.path.join(tmp, "job2", "output", "venv")  # same root name, byte-identical image, same mtime
    shutil.copytree(canon, twin, symlinks=True)
    twin_prefix = run([os.path.join(twin, "bin", "python"), "-c", "import sys; print(sys.prefix)"]).stdout.strip()
    check(twin_prefix.startswith(mnt_base) and twin_prefix != prefix, "two venvs named alike never share a mount")

    bad = os.path.join(tmp, "job-bad", "venv-bad")
    shutil.copytree(canon, bad, symlinks=True)
    with open(os.path.join(bad, sv.LAUNCHER_NAME)) as f:
        src = f.read()
    check(' "$mnt" 9>&-' in src, "the launcher closes the lock fd for the squashfuse daemon")
    with open(os.path.join(bad, sv.LAUNCHER_NAME), "w") as f:
        f.write(src.replace(' "$mnt" 9>&-', ' "$mnt"'))
    res = cold_launches(os.path.join(bad, "bin", "python"), 4, timeout=8)
    check("TIMEOUT" in res, f"non-vacuous: a daemon holding the lock fd deadlocks the same test ({res})")
    unmount_under(mnt_base + "/venv-bad-")

    # -- a mount whose daemon died is recovered ---------------------------------------------------
    alive = run([py, "-c", "import sys; print(sys.prefix)"]).stdout.strip()
    image = os.path.join(os.path.realpath(canon), sv.IMAGE)
    procs = run(["pgrep", "-a", "squashfuse_ll"]).stdout.splitlines()
    pids = [ln.split()[0] for ln in procs if f" {image} {prefix}" in ln]
    for pid in pids:
        os.kill(int(pid), signal.SIGKILL)
    time.sleep(0.5)
    cached = run(["stat", os.path.join(prefix, "pyvenv.cfg")]).returncode == 0
    st = run(["stat", "-f", prefix])
    check(
        alive == prefix and len(pids) == 1 and st.returncode != 0,
        f"killing the daemon leaves a dead mount (alive={alive!r}, pids={pids}, statfs={st.stderr.strip()!r})",
    )
    print(
        f"      (plain stat of a file on the dead mount {'still SUCCEEDS -- cached' if cached else 'fails'}; "
        "the launcher must not trust it)",
        flush=True,
    )
    again = run([py, "-c", "import sys; print(sys.prefix)"])
    check(
        again.stdout.strip() == prefix, f"the next launch remounts it (rc={again.returncode} {again.stderr[-300:]!r})"
    )

    # -- inside SLURM: per-job dir under $TMPDIR --------------------------------------------------
    slurm_tmp = os.path.join(tmp, "slurm_tmp")
    os.makedirs(slurm_tmp)
    env = {k: v for k, v in os.environ.items() if k != "SQSH_VENV_MOUNT_BASE"}
    env.update(SLURM_JOB_ID="4242", TMPDIR=slurm_tmp)
    got = run([py, "-c", "import sys; print(sys.prefix)"], env=env).stdout.strip()
    check(got.startswith(slurm_tmp + "/sqsh-venv-job4242/"), f"under SLURM the mount is per job in $TMPDIR ({got})")
finally:
    unmount_under(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    left = fuse_mounts_under(tmp)
    if left:
        failures.append(f"left mounts behind: {left}")

if failures:
    print(f"\n{len(failures)} FAILED")
    sys.exit(1)
print("\nall passed")
