"""Guard: ``moshi_family.io_prefetch.prefetch`` reads a checkpoint into the page cache and changes nothing.

Every PersonaPlex / Moshi checkpoint load (self-play, engines, both finetune launchers, overlays,
resumes) now calls ``prefetch`` before its ``safetensors`` mmap load (2026-09-24). It must only READ.

  1. IDENTICAL BYTES. A small safetensors file (several dtypes, a size that is not a chunk multiple)
     loads bit-identically before and after a prefetch.
  2. IT READS. After evicting the file from the page cache, one prefetch leaves every page resident
     (reading at most the file; kernel readahead may cache a chunk before its turn); a second call
     reads nothing ("cached"). With a few chunks already cached it reads at most the rest.
  3. IT BACKS OFF. With too little memory to cache the file, or ``MOSHI_FAMILY_PREFETCH=0``, it reads
     nothing. Ranks on one node start in their own slice of the file (``LOCAL_RANK``).
  4. CAN FAIL. A prefetch whose reads are disabled leaves the file uncached, and check 2 sees it; a
     file with one flipped byte does not pass check 1's comparison.

moshi_family venv, login node, seconds (needs a disk-backed temp dir: ``posix_fadvise`` cannot evict
tmpfs pages):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_io_prefetch.py'
"""

import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import moshi_family.io_prefetch as iop  # noqa: E402

CHUNK = 1 << 20


def evict(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    assert iop.resident_fraction(path) < 0.05, (
        f"cannot evict {path} from the page cache (tmpfs?); run the guard with a disk-backed TMPDIR"
    )


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype
        and a[k].shape == b[k].shape
        and torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8))
        for k in a
    )


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    g = torch.Generator().manual_seed(0)
    tensors = {
        "w_bf16": torch.randn(1000, 1537, generator=g).to(torch.bfloat16),
        "w_fp32": torch.randn(777, 1111, generator=g),
        "codes_i64": torch.randint(0, 2048, (333, 555), generator=g),
        "empty": torch.zeros(0),
    }
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as d:
        path = os.path.join(d, "tiny.safetensors")
        save_file(tensors, path)
        size = os.path.getsize(path)
        n_chunks = -(-size // CHUNK)
        assert size % CHUNK and n_chunks >= 6, (size, n_chunks)
        # CPU load_file tensors are views of its mmap; a live mapping pins the pages, so keep copies.
        ref = {k: v.clone() for k, v in load_file(path).items()}
        assert same(ref, tensors)

        # 1 + 2
        evict(path)
        st = quiet(iop.prefetch, path, chunk=CHUNK, threads=3)
        # Kernel readahead can cache a following chunk before its turn; prefetch then skips it.
        assert st["skipped"] is None and st["missing"] == size and 0 < st["read"] <= size, st
        assert iop.resident_fraction(path) == 1.0, "prefetch left pages uncached"
        assert same(load_file(path), ref), "tensors differ after a prefetch"
        st2 = quiet(iop.prefetch, path, chunk=CHUNK)
        assert st2["skipped"] == "cached" and st2["read"] == 0, st2
        evict(path)
        with open(path, "rb") as f:  # cache chunks 0-2 only
            f.read(3 * CHUNK)
        st3 = quiet(iop.prefetch, path, chunk=CHUNK)
        # (kernel readahead may have cached a little past chunk 2, so "at most the rest")
        assert 0 < st3["read"] <= size - 3 * CHUNK and iop.resident_fraction(path) == 1.0, st3
        print(
            f"[ok] 1+2 {size} B file ({n_chunks} chunks): cold prefetch read {st['read']} B, all pages resident, "
            f"tensors bit-identical; warm call read 0 B; with 3 chunks cached it read {st3['read']} B"
        )

        # 3
        evict(path)
        real_avail = iop.available_bytes
        assert 0 < real_avail() <= os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        iop.available_bytes = lambda: size  # 0.8 x size headroom < size to read
        try:
            st4 = quiet(iop.prefetch, path, chunk=CHUNK)
        finally:
            iop.available_bytes = real_avail
        assert st4["skipped"] == "memory" and st4["read"] == 0 and iop.resident_fraction(path) < 0.05, st4
        os.environ["MOSHI_FAMILY_PREFETCH"] = "0"
        try:
            st5 = quiet(iop.prefetch, path, chunk=CHUNK)
        finally:
            del os.environ["MOSHI_FAMILY_PREFETCH"]
        assert st5["skipped"] == "disabled" and st5["read"] == 0, st5
        os.environ.update(LOCAL_RANK="1", LOCAL_WORLD_SIZE="2")
        try:
            order = iop._chunk_order(10)
        finally:
            del os.environ["LOCAL_RANK"], os.environ["LOCAL_WORLD_SIZE"]
        assert order[0] == 5 and sorted(order) == list(range(10)), order
        assert iop._chunk_order(10) == list(range(10))
        print(
            f"[ok] 3 too little memory -> skipped, 0 B read; MOSHI_FAMILY_PREFETCH=0 -> skipped; rank 1 of 2 starts "
            f"at chunk 5; real headroom {real_avail() / 1e9:.1f} GB"
        )

        # 4
        evict(path)
        real_preadv = os.preadv
        os.preadv = lambda fd, bufs, off: sum(len(b) for b in bufs)  # "reads" without reading
        try:
            quiet(iop.prefetch, path, chunk=CHUNK)
        finally:
            os.preadv = real_preadv
        frac = iop.resident_fraction(path)
        assert frac < 0.05, f"check 2 cannot fail: a prefetch that reads nothing left {frac:.0%} resident"
        bad = os.path.join(d, "flipped.safetensors")
        raw = bytearray(open(path, "rb").read())
        raw[-100] ^= 1  # inside the last tensor's data
        open(bad, "wb").write(bytes(raw))
        assert not same(load_file(bad), ref), "check 1 cannot fail: a flipped byte compared equal"
        print(
            f"[ok] 4 a prefetch with its reads disabled leaves {frac:.0%} resident (check 2 fails); a flipped byte "
            f"fails check 1"
        )
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
