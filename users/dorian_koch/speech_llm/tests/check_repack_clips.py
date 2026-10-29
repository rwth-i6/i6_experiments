"""Guard: ``repack_clips.py`` never repacks a shard dir that a merge dir still symlinks into.

``MergeMoshiOutputsViaSymlinks`` joins shards by symlinking their ``{i}.wav``; repacking such a shard deletes those
wavs and the merge's links dangle silently (the first re-read fails months later, with the audio gone). The module's
"merges first" ordering was not enforced until 2026-10-29. On a fake work tree: a finished shard with a merge linking
into it is kept by ``--kind shard``, an unreferenced finished shard is a candidate, and an explicit ``--dir`` of the
held shard is refused. Can fail: with ``merge_referenced_dirs`` returning nothing, the held shard becomes a candidate.

Run from the setup root: ``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_repack_clips.py``
"""

import os
import sys
import tempfile
from pathlib import Path

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.join(os.getcwd(), "recipe/i6_experiments/users/dorian_koch/speech_llm"))
import repack_clips as rc  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="check_repack_"))
rc.WORK = tmp


def job(rel, name):
    d = tmp / rel / "output" / name
    d.mkdir(parents=True)
    (tmp / rel / "finished").touch()
    return d


held = job("speech_inference/SpeechInference.aaaaaaaaaaaa", "out")
free = job("speech_inference/SpeechInference.bbbbbbbbbbbb", "out")
for d in (held, free):
    for i in range(3):
        (d / f"{i}.wav").write_bytes(b"RIFF")
merge = job("knowledge_benchmark/MergeMoshiOutputsViaSymlinks.cccccccccccc", "out")
for i in range(3):
    os.symlink(held / f"{i}.wav", merge / f"{i}.wav")

got = {d for _, d in rc.candidates(["shard"])}
assert free in got and held not in got, got
real = rc.merge_referenced_dirs
rc.merge_referenced_dirs = lambda: set()
assert held in {d for _, d in rc.candidates(["shard"])}, "the fixture cannot tell a held shard from a free one"
rc.merge_referenced_dirs = real
sys.argv = [sys.argv[0], "--dir", str(held)]
try:
    rc.main()
    raise AssertionError("an explicit --dir of a held shard was not refused")
except SystemExit as e:
    assert "refusing" in str(e), e
print("check_repack_clips: OK (a shard a merge links into is kept; explicit --dir refused; can fail)")
