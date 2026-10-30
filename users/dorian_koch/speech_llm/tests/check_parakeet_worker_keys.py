"""Guard: the Parakeet knowledge worker (``parakeet_benchmark_inference.main``) finds its clips in both clip-store
layouts (2026-10-30).

The audit fix of 2026-10-29 read the clip indices as ``set(clips)``; the lazy arrow store has no ``__iter__``, so Python
iterated it by ``__getitem__(0), (1), ...``, decoding clips into unhashable tuples: every Parakeet transcription of an
arrow (``storage="hf"``) store failed (6 jobs, 2026-10-30). Drives the real ``main()`` on a tiny arrow store and a
``<i>.wav`` dir with a stub model (no NeMo): the output holds exactly the stored indices, in reference order, with the
stub's text. CAN FAIL: the ``set(clips)`` line is caught.

    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup root> && python <this file>'
"""

import json
import os
import sys
import tempfile
import types

import numpy as np
import soundfile as sf
from datasets import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import parakeet_benchmark_inference as PK  # noqa: E402

SRC = PK.__file__
fails = []


def ck(cond, msg, extra=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}  {extra}".rstrip())
    if not cond:
        fails.append(msg)


class _Model:
    def transcribe(self, audios, batch_size, verbose):
        return [types.SimpleNamespace(text=f"len {a.size}") for a in audios]


def run(mod, in_dir, ref_dir, out):
    mod.load_model = lambda name, device: _Model()
    argv = sys.argv
    sys.argv = ["x", "--in_dir", in_dir, "--reference_data", ref_dir, "--out_json", out]
    try:
        mod.main()
    finally:
        sys.argv = argv
    return [json.loads(x) for x in open(out)]


tmp = tempfile.mkdtemp()
ref_dir = os.path.join(tmp, "ref")
Dataset.from_dict({"question": [f"q{i}" for i in range(4)], "answer": ["a"] * 4, "aliases": [["a"]] * 4,
                   "category": ["c"] * 4}).save_to_disk(ref_dir)  # fmt: skip
stored = {0: 16000, 2: 8000, 3: 24000}  # index -> samples at 16 kHz
arrow_dir = os.path.join(tmp, "arrow")
Dataset.from_dict({"index": list(stored), "audio": [np.full(n, 0.1, np.float32).tolist() for n in stored.values()],
                   "sampling_rate": [16000] * 3}).save_to_disk(arrow_dir)  # fmt: skip
wav_dir = os.path.join(tmp, "wav")
os.makedirs(wav_dir)
for i, n in stored.items():
    sf.write(os.path.join(wav_dir, f"{i}.wav"), np.full(n, 0.1, np.float32), 16000)

print("[1] both layouts: exactly the stored indices, in reference order")
for name, d in (("arrow", arrow_dir), ("wav", wav_dir)):
    rows = run(PK, d, ref_dir, os.path.join(tmp, f"{name}.jsonl"))
    ck([r["index"] for r in rows] == [0, 2, 3], f"{name}: indices", str([r["index"] for r in rows]))
    ck([r["transcription"] for r in rows] == ["len 16000", "len 8000", "len 24000"], f"{name}: per-clip text")

print("[2] CAN FAIL: set(clips) on the lazy arrow store")
src = open(SRC).read()
old = "    keys = set(clips.keys())"
assert old in src, "mutation anchor gone"
mut = types.ModuleType("pk_mutant")
mut.__file__ = SRC
exec(compile(src.replace(old, "    keys = set(clips)", 1), SRC, "exec"), mut.__dict__)
try:
    run(mut, arrow_dir, ref_dir, os.path.join(tmp, "mut.jsonl"))
    ck(False, "the mutant is caught")
except TypeError as e:
    ck("unhashable" in str(e), "the mutant is caught", repr(e))

print()
if fails:
    print(f"FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("check_parakeet_worker_keys: all passed")
