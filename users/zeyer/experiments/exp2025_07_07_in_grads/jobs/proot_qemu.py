"""
Run an amd64 container image on an aarch64 host fully in userspace: ``proot -q qemu-x86_64``.

No root, no ``binfmt_misc`` registration, no site container group:
proot virtualizes the root directory and, with ``-q``, runs every ``execve`` through qemu-user,
so a tool with a subprocess pipeline (Kaldi binaries, PostgreSQL) works unchanged.
Pieces, each a tracked job:
the static ``qemu-x86_64`` from a Debian arm64 package (:class:`ExtractDebianPackageJob`),
proot built from source against Debian's talloc (:class:`BuildProotJob`,
the packaged proot 5.1 crashes on the 5.14 kernel with glibc 2.34),
the image rootfs pulled as OCI layers from a registry (:class:`PullOciImageRootfsJob`),
and the wrapper script (:class:`ProotQemuExeWrapperJob`), the same interface as
:class:`...apptainer.ApptainerExeWrapperJob`.
Emulation costs roughly an order of magnitude in CPU time; Python start-up of the MFA image is ~45 s.
"""

from __future__ import annotations

import io
import json
import lzma
import os
import shutil
import subprocess
import tarfile
import urllib.request
from typing import Dict, List, Optional

from sisyphus import Job, Task, tk

__all__ = ["ExtractDebianPackageJob", "BuildProotJob", "PullOciImageRootfsJob", "ProotQemuExeWrapperJob"]


def _fetch(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=300) as r:
        return r.read()


def _deb_extract(deb: bytes, dest: str, members: Optional[List[str]] = None):
    """Extract the data.tar of a .deb (an ar archive) into dest, no dpkg needed."""
    assert deb[:8] == b"!<arch>\n", "not a .deb (ar) archive"
    pos = 8
    while pos < len(deb):
        name = deb[pos : pos + 16].decode().strip()
        size = int(deb[pos + 48 : pos + 58].decode().strip())
        pos += 60
        body = deb[pos : pos + size]
        pos += size + (size % 2)
        if not name.startswith("data.tar"):
            continue
        if name.endswith(".xz"):
            body, mode = lzma.decompress(body), "r:"
        else:
            mode = "r:*"
        with tarfile.open(fileobj=io.BytesIO(body), mode=mode) as tar:
            for m in tar.getmembers():
                if members is None or any(m.name.lstrip("./").startswith(x) for x in members):
                    tar.extract(m, dest, filter="fully_trusted")


class ExtractDebianPackageJob(Job):
    """Download one Debian package (an exact pool URL, so the hash pins the version) and unpack its files."""

    def __init__(self, url: str, *, members: Optional[List[str]] = None):
        """
        :param url: e.g. ``https://deb.debian.org/debian/pool/main/q/qemu/qemu-user-static_7.2+dfsg-7+deb12u18+b3_arm64.deb``
        :param members: path prefixes (without leading ``./``) to extract, default all
        """
        super().__init__()
        self.url = url
        self.members = members
        self.out_dir = self.output_path("files", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        _deb_extract(_fetch(self.url), self.out_dir.get_path(), self.members)


class BuildProotJob(Job):
    """Build proot from the release tarball on this host, linked against an unpacked Debian talloc."""

    # v1: fresh source tree per run, feature-check assertion (a build without process_vm / seccomp_filter
    # exited 182 on every exec)
    __sis_version__ = 1

    def __init__(self, *, version: str, talloc_dev_dir: tk.Path, talloc_lib_dir: tk.Path):
        """
        :param version: proot release tag without the ``v``, e.g. ``"5.4.1"``
        :param talloc_dev_dir: files of ``libtalloc-dev`` (:class:`ExtractDebianPackageJob`)
        :param talloc_lib_dir: files of ``libtalloc2`` (same version)
        """
        super().__init__()
        self.version = version
        self.talloc_dev_dir = talloc_dev_dir
        self.talloc_lib_dir = talloc_lib_dir
        self.out_proot = self.output_path("proot")
        self.out_lib_dir = self.output_path("lib", directory=True)  # the talloc shared lib, rpath target
        self.rqmt = {"cpu": 4, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, mini_task=True)

    def run(self):
        # a fresh tree: a rerun after a failed link would keep stale feature-check results
        # (process_vm / seccomp_filter off), and such a proot exits 182 on the first exec
        src = os.path.join(os.getcwd(), "src")
        shutil.rmtree(src, ignore_errors=True)
        with tarfile.open(
            fileobj=io.BytesIO(_fetch(f"https://github.com/proot-me/proot/archive/refs/tags/v{self.version}.tar.gz")),
            mode="r:gz",
        ) as tar:
            tar.extractall(src, filter="fully_trusted")
        srcdir = os.path.join(src, f"proot-{self.version}", "src")
        lib_dir = self.out_lib_dir.get_path()
        for fn in os.listdir(os.path.join(self.talloc_lib_dir.get_path(), "usr/lib/aarch64-linux-gnu")):
            if fn.startswith("libtalloc.so"):
                shutil.copy(os.path.join(self.talloc_lib_dir.get_path(), "usr/lib/aarch64-linux-gnu", fn), lib_dir)
        # the unversioned link for -ltalloc is in the dev package
        os.symlink("libtalloc.so.2", os.path.join(lib_dir, "libtalloc.so"))
        include_dir = os.path.join(self.talloc_dev_dir.get_path(), "usr/include")
        pc_dir = os.path.join(os.getcwd(), "pkgconfig")
        os.makedirs(pc_dir, exist_ok=True)
        with open(os.path.join(pc_dir, "talloc.pc"), "w") as f:
            f.write(
                f"prefix=/\nlibdir={lib_dir}\nincludedir={include_dir}\n\nName: talloc\nDescription: talloc\n"
                "Version: 2.3.1\nLibs: -L${libdir} -ltalloc\nCflags: -I${includedir}\n"
            )
        env = dict(os.environ, PKG_CONFIG_PATH=pc_dir, LDFLAGS=f"-Wl,-rpath,{lib_dir}")
        subprocess.check_call(["make", "-C", srcdir, "proot", "GIT=false", f"-j{self.rqmt['cpu']}"], env=env)
        shutil.copy(os.path.join(srcdir, "proot"), self.out_proot.get_path())
        out = subprocess.check_output([self.out_proot.get_path(), "--version"]).decode()
        print(out)
        assert "process_vm = yes, seccomp_filter = yes" in out, "feature checks failed, see the make output"
        subprocess.check_call([self.out_proot.get_path(), "-r", "/", "/usr/bin/true"])


class PullOciImageRootfsJob(Job):
    """Pull an image from a Docker registry (v2 API, anonymous pull) and unpack its layers into a rootfs dir.

    Device nodes are skipped (bind the host ``/dev``), whiteouts are applied.
    ``out_config`` is the image config (Env, Entrypoint, ...).
    """

    def __init__(
        self, image: str, *, digest: str, registry: str = "registry-1.docker.io", auth: str = "auth.docker.io"
    ):
        """
        :param image: e.g. ``"mmcauliffe/montreal-forced-aligner"``
        :param digest: the platform manifest digest (``sha256:...``), so the pull is pinned
            (a tag such as ``latest`` moves; find the digest in the tag's manifest list)
        :param registry:
        :param auth: token service host
        """
        super().__init__()
        self.image = image
        self.digest = digest
        self.registry = registry
        self.auth = auth
        self.out_rootfs = self.output_path("rootfs", directory=True)
        self.out_config = self.output_path("config.json")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, mini_task=True)

    def run(self):
        token = json.loads(
            _fetch(f"https://{self.auth}/token?service=registry.docker.io&scope=repository:{self.image}:pull")
        )["token"]
        hdr = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
        }
        base = f"https://{self.registry}/v2/{self.image}"
        manifest = json.loads(_fetch(f"{base}/manifests/{self.digest}", hdr))
        assert "layers" in manifest, f"not a platform manifest: {manifest.keys()}"
        config = json.loads(_fetch(f"{base}/blobs/{manifest['config']['digest']}", hdr))
        rootfs = self.out_rootfs.get_path()
        for i, layer in enumerate(manifest["layers"]):
            print(f"layer {i + 1}/{len(manifest['layers'])}: {layer['size'] / 2**20:.0f} MB", flush=True)
            blob = _fetch(f"{base}/blobs/{layer['digest']}", hdr)
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
                for m in tar.getmembers():
                    bn = os.path.basename(m.name)
                    if bn == ".wh..wh..opq":
                        continue
                    if bn.startswith(".wh."):
                        shutil.rmtree(os.path.join(rootfs, os.path.dirname(m.name), bn[4:]), ignore_errors=True)
                        continue
                    if m.isdev():
                        continue
                    tar.extract(m, rootfs, filter="fully_trusted")
        with open(self.out_config.get_path(), "w") as f:
            json.dump(config, f, indent=1)


class ProotQemuExeWrapperJob(Job):
    """Executable wrapper: ``wrapper <args...>`` -> ``proot -q <qemu> -r <rootfs> [-b ...] [-w $PWD] <command> <args...>``.

    The image's ``Env`` (PATH, LANG, ...) is applied, the caller's environment is passed through,
    ``KMP_AFFINITY=disabled`` is set (Intel OpenMP's affinity probe aborts under qemu).
    Host ``/dev``, ``/proc``, ``/sys``, ``/tmp`` and the resolver files are always bound;
    pass ``bind`` for the data and work dirs.
    """

    def __init__(
        self,
        *,
        rootfs: tk.Path,
        image_config: tk.Path,
        proot: tk.Path,
        proot_lib_dir: tk.Path,
        qemu: tk.Path,
        command: str,
        bind: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        :param rootfs: :attr:`PullOciImageRootfsJob.out_rootfs`
        :param image_config: :attr:`PullOciImageRootfsJob.out_config`
        :param proot: :attr:`BuildProotJob.out_proot`
        :param proot_lib_dir: :attr:`BuildProotJob.out_lib_dir`
        :param qemu: the static ``qemu-x86_64`` binary
        :param command: the in-image command, e.g. ``"/env/bin/mfa"``
        :param bind: extra host paths to bind (``host[:guest]``)
        :param env: extra environment variables
        """
        super().__init__()
        self.rootfs = rootfs
        self.image_config = image_config
        self.proot = proot
        self.proot_lib_dir = proot_lib_dir
        self.qemu = qemu
        self.command = command
        self.bind = bind or []
        self.env = env or {}
        self.out_exe = self.output_path("run.sh")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.image_config.get_path()) as f:
            image_env = dict(kv.split("=", 1) for kv in json.load(f)["config"].get("Env", []))
        binds = [
            "/dev",
            "/proc",
            "/sys",
            "/tmp",
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/host.conf",
        ]
        binds += self.bind
        # image env = defaults only, the caller's environment wins (e.g. MFA_ROOT_DIR of the MFA jobs);
        # the image PATH is prepended (the tool's subprocesses find the image binaries by name)
        exports = "".join(f'export {k}="${{{k}:-{v}}}"\n' for k, v in image_env.items() if k != "PATH")
        if "PATH" in image_env:
            exports += f'export PATH="{image_env["PATH"]}:$PATH"\n'
        exports += "".join(f"export {k}={json.dumps(v)}\n" for k, v in {"KMP_AFFINITY": "disabled", **self.env}.items())
        cmd = [self.proot.get_path(), "-q", self.qemu.get_path(), "-r", self.rootfs.get_path()]
        for b in binds:
            cmd += ["-b", b]
        cmd += ["-w", '"$PWD"', self.command]
        with open(self.out_exe.get_path(), "w") as f:
            f.write(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f'export LD_LIBRARY_PATH="{self.proot_lib_dir.get_path()}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n'
                + exports
                + "exec "
                + " ".join(c if c.startswith('"') else json.dumps(c) for c in cmd)
                + ' "$@"\n'
            )
        os.chmod(self.out_exe.get_path(), 0o755)
