"""The MFDB output store: a packed tree must give every reader back the SAME files, byte for byte.

Since 2026-10-14 the MFDB (Moshified FDB) inference output is one arrow dataset (``file_tree_store.pack_tree``) instead
of a ``<item>/output.wav, output.json, ...`` tree, and every reader -- the fork's ``evaluate.py`` via
``FullDuplexBenchEval_Evaluation``, ``fdb_turn_taking_v2.py`` via ``FDBTurnTakingRescore`` -- runs on a temporary
unpack (``tree_root``). The safety argument is that the unpacked tree is indistinguishable from the original, so the
benchmark code sees exactly the files it saw before. What would go wrong without this guard:
  * a dropped or renamed file -> an evaluator skips that item (they walk the tree) and the MFDB number moves for a
    storage reason, with nothing failing
  * a NUL-holed ``output.json`` (quota kill, 2026-10-13) packed silently -> the eval dies later, in another job
  * the temp unpack not named ``speech_output`` -> ``evalstats._mfdb_item_of`` stops finding items in the eval log
  * an evaluator's write (``rating.json``) leaking into the stored output, or the temp dir never being deleted

Checks 6-7 drive the REAL ``run()`` of both eval jobs (their scoring subprocess stubbed) on a packed tree.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_file_tree_store.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm import file_tree_store as fts  # noqa: E402


def snapshot(root) -> dict:
    """{relpath: bytes} of every file under root (directories implied)."""
    root = Path(root)
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def make_tree(root: Path) -> None:
    """A small MFDB-shaped tree: nested item dirs, binary wav-like bytes, json, an empty file, a unicode name."""
    for item in ("0", "1", "10", "2"):
        d = root / item
        d.mkdir(parents=True)
        (d / "output.wav").write_bytes(bytes(range(256)) * (int(item) + 3) + b"\x00\x00tail")
        (d / "output.json").write_text(json.dumps({"text": f"item {item}", "chunks": [[0.0, 0.5]]}))
        (d / "pause.json").write_text("[]")
    (root / "2" / "sub").mkdir()
    (root / "2" / "sub" / "nested.txt").write_text("deeper")
    (root / "1" / "empty.txt").write_bytes(b"")
    (root / "10" / "größe ü.txt").write_text("unicode name")


def run_checks(tmp: Path) -> list[str]:
    fails = []

    def check(cond, msg):
        print(("[ok] " if cond else "[FAIL] ") + msg)
        if not cond:
            fails.append(msg)

    src = tmp / "speech_output"
    make_tree(src)
    want = snapshot(src)
    packed = tmp / "packed"

    # 1. round trip, byte for byte, through the reader every eval job uses
    n = fts.pack_tree(src, packed)
    check(n == len(want), f"pack_tree packed {n} of {len(want)} files")
    check(fts.is_tree_dataset(packed) and not fts.is_tree_dataset(src), "is_tree_dataset tells packed from plain")
    check(len(os.listdir(packed)) <= 4, f"packed store is {len(os.listdir(packed))} entries, not one per file")
    with fts.tree_root(packed) as root:
        got = snapshot(root)
        check(got == want, "tree_root unpack is byte-identical to the source tree")
        check(Path(root).name == "speech_output", "temp unpack keeps the speech_output root name (eval-log parsing)")
        (Path(root) / "0" / "rating.json").write_text("{}")  # what eval_user_interruption does
        temp_root = root
    check(not os.path.exists(temp_root), "temp unpack is deleted on exit")
    with fts.tree_root(packed) as root:
        check("0/rating.json" not in snapshot(root), "a reader's write never reaches the stored tree")
    with fts.tree_root(src) as root:
        check(root == str(src), "a plain tree (pre-2026-10-14 output) is passed through unchanged")
    with fts.tree_root(src, writable=True) as root:
        check(root != str(src) and snapshot(root) == want, "writable=True copies a plain tree")
        (Path(root) / "0" / "rating.json").write_text("{}")
    check(not (src / "0" / "rating.json").exists(), "a writer never reaches a stored plain tree (writable=True)")
    check(fts.verify_tree(packed, src) is None, "verify_tree accepts the matching source")

    # 2. verify_tree catches a changed byte, a missing file and an extra file
    (src / "0" / "output.wav").write_bytes(want["0/output.wav"][:-1] + b"X")
    check(fts.verify_tree(packed, src) is not None, "verify_tree catches a one-byte change")
    (src / "0" / "output.wav").write_bytes(want["0/output.wav"])
    (src / "3").mkdir()
    (src / "3" / "output.json").write_text("{}")
    check(fts.verify_tree(packed, src) is not None, "verify_tree catches a file missing from the pack")
    shutil.rmtree(src / "3")

    # 3. a NUL-holed json (valid prefix + \x00 padding, as on 2026-10-13) is refused, naming the file
    holed = tmp / "holed"
    shutil.copytree(src, holed)
    (holed / "1" / "output.json").write_bytes(b'{"text": "ok"}' + b"\x00" * 64)
    try:
        fts.pack_tree(holed, tmp / "holed_packed")
        check(False, "pack_tree refuses a NUL-holed output.json")
    except ValueError as exc:
        check("1/output.json" in str(exc), "pack_tree refuses a NUL-holed output.json, naming it")
    check(not fts.is_tree_dataset(tmp / "holed_packed"), "nothing is written for a refused tree")

    # 4. what the layout cannot represent is refused, not dropped
    for name, setup in (
        ("empty directory", lambda d: (d / "4").mkdir()),
        ("symlink", lambda d: os.symlink(d / "0" / "output.wav", d / "2" / "link.wav")),
    ):
        d = tmp / name.replace(" ", "_")
        shutil.copytree(src, d)
        setup(d)
        try:
            fts.pack_tree(d, tmp / (d.name + "_packed"))
            check(False, f"pack_tree refuses a tree with a {name}")
        except ValueError:
            check(True, f"pack_tree refuses a tree with a {name}")

    # 5. a clip_store dataset is not mistaken for a tree
    from i6_experiments.users.dorian_koch.speech_llm.clip_store import write_clips
    import numpy as np

    clips = tmp / "clips"
    write_clips(clips, [(0, np.zeros(10, dtype=np.float32), 16000)])
    check(not fts.is_tree_dataset(clips), "a clip_store dataset is not a packed tree")

    # 6-7. the REAL run() of both eval jobs hands its scorer a byte-identical tree (scoring itself stubbed)
    from sisyphus import tk

    from i6_experiments.users.dorian_koch.speech_llm import fdb

    seen = {}
    for cls, inner, kwargs in (
        (fdb.FullDuplexBenchEval_Evaluation, "_run", dict(fdb_task="candor_pause_handling")),
        (fdb.FDBTurnTakingRescore, "_score", dict(eval_venv_python=tk.Path("/usr/bin/python3"))),
    ):
        job = cls(in_audios=tk.Path(str(packed)), **kwargs)
        orig = getattr(cls, inner)

        def capture(self, *args, _cls=cls):
            root = args[-1]
            seen[_cls.__name__] = (snapshot(root), os.path.basename(root))

        setattr(cls, inner, capture)
        try:
            job.run()
        finally:
            setattr(cls, inner, orig)
        got, base = seen.get(cls.__name__, ({}, ""))
        check(got == want and base == "speech_output", f"{cls.__name__}.run() scores a byte-identical tree")
    return fails


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="check_file_tree_store."))
    try:
        fails = run_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print("\nFAILED:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("\nfile tree store: a packed MFDB output reads back byte for byte")
