"""Guard: external corpus paths hash by a FROZEN identity, not by where the data currently sits.

A creator-less ``tk.Path`` hashes by its absolute string (``sisyphus/job_path.py::Path._sis_hash``
returns ``sis_hash_helper((creator, path))``). So a raw ``tk.Path("/hpcwork/.../fisher/LDC2004S13")``
ties every downstream job hash to that literal location: relocate the corpus -- or port the setup to
another cluster -- and ``FisherSphToWav`` re-hashes, then ``FisherToMoshiTrainData``, then every
Fisher-trained arm. ~400 GB of ``fisher_prep`` is orphaned and the arms retrain, for a change that
moved no data and no code.

``training.py`` therefore pins those paths with ``hash_overwrite``, and ``track_rl.py`` does the same
for the one absolute ``work/`` reference. This check exists because that pinning is **invisible when
it works** -- nothing fails, nothing warns, and the damage only shows up on a migration nobody is
doing the day the pin is removed.

Two things are checked, and both matter:

  1. The identity ``Path(p, hash_overwrite=p) == Path(p)`` still holds. The pins were applied on the
     strength of it -- ``hash_overwrite``'s setter wraps a bare string to ``(None, value)``, which is
     exactly what a creator-less path already hashes. If a sisyphus upgrade changed that, every pin
     would silently become a *different* hash and cascade the retrain this guard exists to prevent.
  2. The pinned literals still hash to the values the existing job dirs were built with. This is what
     catches somebody "tidying up" ``_FISHER_HASH_ROOT`` to follow ``_fisher_root`` after the corpus
     moves -- the single most natural way to undo the fix.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_path_hash_pinning.py
"""

import hashlib
import os
import sys
from pathlib import Path as _FsPath

# abspath, NOT resolve(): every entry under `recipe/` is a symlink into one of the `projects/`
# repos, so resolve() would land in `projects/` and the source-inspection checks below would look
# for files that only exist under `recipe/`.
RECIPE = _FsPath(os.path.abspath(__file__)).parents[5]  # .../recipe
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(RECIPE / "sisyphus"))

from sisyphus import tk  # noqa: E402

failures = []

#: path -> sha256(_sis_hash())[:16], captured 2026-08-05 from the live graph, whose Fisher job ids
#: (FisherSphToWav.bb5Gjm6qeOg8, PrepareFisherDatasetJob.B3nfAjGOSR4d, FisherToMoshiTrainData
#: .JJi8ACF3GodT / .98IFVP8879x4 / .E6HnLS4wyaba) match the job dirs already on disk. These are the
#: FROZEN identities -- if the corpus moves, the path above them changes and these do NOT.
PINNED = {
    "/hpcwork/tt201262/corpora/fisher/LDC2004S13": "b89d011094588edf",
    "/hpcwork/tt201262/corpora/fisher/LDC2005S13": "fc8e4c4c3c1b4551",
    "/hpcwork/tt201262/corpora/fisher/LDC2004T19/fe_03_p1_tran/data/trans": "e34f72722dd46e8f",
    "/hpcwork/tt201262/corpora/fisher/LDC2005T19/fe_03_p2_tran/data/trans": "49ed3519ebd7120b",
    "/rwthfs/rz/cluster/home/tt201262/setups/2026-01-speech-llm/work/i6_experiments/users/"
    "dorian_koch/speech_llm/rl_finetune/RLFinetune.FESvleKrAwPk/output/run_dir": "34cb2bb472941657",
}


def digest(path):
    return hashlib.sha256(path._sis_hash()).hexdigest()[:16]


# --- 1. pinning is still a no-op for a path that has NOT moved -----------------------------------
for p in PINNED:
    if tk.Path(p)._sis_hash() != tk.Path(p, hash_overwrite=p)._sis_hash():
        failures.append(f"hash_overwrite is no longer identity-preserving for {p!r} -- every pin in "
                        f"the recipe just became a different hash")
print(f"[ok] identity holds      hash_overwrite=p == bare Path(p) for all {len(PINNED)} pinned paths")

# --- 2. the frozen literals still produce the hashes the job dirs were built with -----------------
for p, want in PINNED.items():
    got = digest(tk.Path(p, hash_overwrite=p))
    if got != want:
        failures.append(f"pinned hash changed for {p!r}: {got} != {want} -- downstream jobs will re-run")
print("[ok] literals unchanged  all 5 frozen identities hash as they did on 2026-08-05")

# --- 3. and pinning actually DOES decouple the hash from the location ----------------------------
# The whole point. If this stops being true the pins are decoration.
moved = "/somewhere/else/fisher/LDC2004S13"
original = "/hpcwork/tt201262/corpora/fisher/LDC2004S13"
if tk.Path(moved, hash_overwrite=original)._sis_hash() != tk.Path(original)._sis_hash():
    failures.append("a relocated-but-pinned path no longer hashes as the original -- pinning is broken")
if tk.Path(moved)._sis_hash() == tk.Path(original)._sis_hash():
    failures.append("an UNPINNED relocated path hashes as the original -- then this guard is testing nothing")
print("[ok] decoupling works    a moved-but-pinned path keeps its hash; a moved unpinned one does not")

# --- 4. the recipe really uses the pins ----------------------------------------------------------
# Checking the property without checking the call sites would pass happily on an unpinned recipe.
src = (RECIPE / "speech_llm/full_duplex/sis_recipe/doriank/training.py").read_text()
if "_FISHER_HASH_ROOT" not in src or "hash_overwrite" not in src:
    failures.append("training.py no longer pins the Fisher paths")
if src.count("_fisher_path(") < 4:
    failures.append(f"expected 4+ _fisher_path() call sites in training.py, found {src.count('_fisher_path(')} "
                    f"-- a Fisher path added without pinning?")
if 'tk.Path(f"{_fisher_root}' in src:
    failures.append("training.py builds a Fisher tk.Path directly from _fisher_root -- that path is unpinned")
rl_src = (RECIPE / "speech_llm/full_duplex/sis_recipe/doriank/track_rl.py").read_text()
if "hash_overwrite=_RL_SALVAGE_RUNDIR" not in rl_src:
    failures.append("track_rl.py no longer pins its absolute work/ rundir")
print("[ok] call sites pinned   training.py routes all 4 Fisher paths through _fisher_path(); track_rl pinned")

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nexternal corpus paths hash by frozen identity -- the setup can be relocated without re-running")
