"""Guard: ``offline_cli._materialise_arrow_clips`` reads arrow clips in numpy format and writes byte-identical wavs.

Audit item 7c (2026-10-12). Every offline driver unpacks an arrow question store (``storage="hf"``) through this one
function. It iterated the dataset in the default format, which hands each clip to ``soundfile`` as a Python list of
floats -- the pattern ``clip_store.materialise_clips`` had to drop after RSS grew ~0.6 GB/min with the corpus
(2026-09-11). A speed / memory issue, not a correctness one: soundfile turns the list into float64, so reading
``with_format("numpy")`` and passing float64 writes the same bytes.

[1] the real function on a clip store with the clip_store columns (random audio, exact half-LSB int16 boundaries,
    clipping values, two sample rates, a monologue column it must ignore): every wav byte-identical to the legacy
    implementation (default format, list samples); [2] no Python list reaches ``soundfile.write`` (numpy in);
    CAN FAIL: the legacy implementation under the same probe hands lists.

moshi_family venv, CPU, seconds:
    ./hpc-venv.py moshi_family_venv --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_offline_cli_materialise.py'
"""

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import Dataset, Features, Sequence, Value, load_from_disk  # noqa: E402

from moshi_family import offline_cli  # noqa: E402

FAILURES: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def legacy(src: Path, scratch: Path) -> None:
    """The pre-2026-10-12 body of ``_materialise_arrow_clips``."""
    scratch.mkdir(parents=True, exist_ok=True)
    for row in load_from_disk(str(src)):
        sf.write(scratch / f"{int(row['index'])}.wav", row["audio"], int(row["sampling_rate"]))


@contextlib.contextmanager
def probe_write(seen: list):
    real = sf.write

    def spy(path, data, *a, **k):
        seen.append(type(data))
        return real(path, data, *a, **k)

    sf.write = spy
    try:
        yield
    finally:
        sf.write = real


def main():
    tmp = Path(tempfile.mkdtemp(prefix="check_offline_cli_materialise_"))
    rng = np.random.default_rng(0)
    half_lsb = ((np.arange(-32767, 32767, 7) + 0.5) / 32767).astype(np.float32)
    clips = [
        rng.uniform(-1, 1, 48000).astype(np.float32),
        half_lsb,
        np.array([0.0, 1.0, -1.0, 1.5, -2.0, 1e-9, -1e-9, 0.99999], np.float32),
        (0.3 * rng.standard_normal(16000)).astype(np.float32),
    ]
    feats = Features(
        {
            "index": Value("int64"),
            "audio": Sequence(Value("float32")),
            "sampling_rate": Value("int64"),
            "monologue": Value("string"),
        }  # fmt: skip
    )
    src = tmp / "store"
    Dataset.from_dict(
        {
            "index": [3, 0, 7, 1],
            "audio": [c.tolist() for c in clips],
            "sampling_rate": [24000, 24000, 16000, 16000],
            "monologue": ["a", "", None, "b"],
        },  # fmt: skip
        features=feats,
    ).save_to_disk(str(src))

    print("== [1] wav bytes == the legacy implementation")
    new_seen, old_seen = [], []
    with probe_write(new_seen), contextlib.redirect_stdout(io.StringIO()):
        out = offline_cli._materialise_arrow_clips(src, tmp / "new")
    with probe_write(old_seen):
        legacy(src, tmp / "old")
    names = sorted(p.name for p in (tmp / "old").glob("*.wav"))
    check(names == ["0.wav", "1.wav", "3.wav", "7.wav"] and out == tmp / "new", f"clips written: {names}")
    same = [n for n in names if (tmp / "new" / n).read_bytes() == (tmp / "old" / n).read_bytes()]
    check(same == names, f"byte-identical: {len(same)}/{len(names)}")

    print("== [2] numpy, not Python lists, reaches soundfile")
    check(
        new_seen and all(t is np.ndarray for t in new_seen), f"new path hands {sorted({t.__name__ for t in new_seen})}"
    )
    check(
        old_seen and all(t is list for t in old_seen),
        f"CAN FAIL: the legacy path hands {sorted({t.__name__ for t in old_seen})}",
    )

    if FAILURES:
        print(f"\ncheck_offline_cli_materialise: {len(FAILURES)} FAILED")
        sys.exit(1)
    print("\ncheck_offline_cli_materialise: all passed")


if __name__ == "__main__":
    main()
