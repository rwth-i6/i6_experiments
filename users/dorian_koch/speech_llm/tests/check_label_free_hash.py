"""Guard: run names never reach a NEW job's hash (user, 2026-10-13: "we should make sure the run name never makes its
way into any hash"), and no job that existed then moved.

Mechanism (``label_free_hash.py``): the ten label-carrying job classes hash their label-free view unless their old,
label-keyed hash is one of the jobs on disk on 2026-10-13 (``label_hash_legacy.py``).

  [1] UNIT, every class, built through sisyphus's own constructor: two jobs that differ ONLY in their labels are one job;
      a different input path is a different job; the label-free hash differs from the label-keyed one; a job whose
      label-keyed hash is listed as legacy keeps it (and keeps hashing its labels). CAN FAIL: the same comparison with
      the label-keyed view (a mutant ``canon`` that keeps the labels) finds the rename. A legacy arm renamed
      later (``RENAMED_RUNS``, the three short-named arms 2026-10-14) keeps its legacy hash, and moves without the entry.
  [2] REAL GRAPH (two child processes, ~3 min each, in parallel): the whole graph as declared, and the graph with every
      arm declared from 2026-10-13 on RENAMED (``RENAME``: the four user-text arms, the constant-LR arm, the T 1 RL arm
      and the eval suffixes) -> the multiset of (class, hash) is identical. The rename must have taken effect (output
      names carry the guard's prefix), the label-carrying classes have label-free jobs in the graph, and no two
      declarations collapsed into one job by differing only in labels (``label_collisions``).
  [3] CAN FAIL: ``--mutant <Class>`` makes that class's label-free view keep its labels in both children; [2] must then
      FAIL (e.g. ``--mutant PersonaSelfPlay``: the renamed arms' self-play moves).

Run from the setup root (~4 min, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_label_free_hash.py
    ... check_label_free_hash.py --mutant PersonaSelfPlay     # must FAIL
"""

import argparse
import collections
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
os.chdir(SETUP)
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

PKG = "i6_experiments.users.dorian_koch.speech_llm"
DORIANK = "speech_llm.full_duplex.sis_recipe.doriank"
PREFIX = "zzguard_renamed"
#: (module, attribute) of every arm name declared from 2026-10-13 on that the guard renames (value: its new name).
RENAME = {
    ("track_persona_usertext", "USERTEXT_KN_ARM"): f"{PREFIX}_kn",
    ("track_persona_usertext", "USERTEXT_MID_ARM"): f"{PREFIX}_mid",
    ("track_persona_usertext", "USERTEXT_CTC_ARM"): f"{PREFIX}_ctc",
    ("track_persona_usertext", "USERTEXT_KN_W005_ARM"): f"{PREFIX}_w005",
    ("track_persona_ab2h", "AB2_CONSTLR_ARM"): f"{PREFIX}_constlr",
    ("track_persona_rl", "RL_T1_ARM"): f"{PREFIX}_rl_t1",
    ("track_persona_depth", "QCTX_SUFFIX"): f"_{PREFIX}_qctx",
    ("track_persona_depth", "ECHO_SUFFIX"): f"_{PREFIX}_echo",
}
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


def apply_mutant(cls_name):
    """The label-free view of ``cls_name`` keeps every label (a label leaking back into the hash)."""
    import importlib

    from i6_experiments.users.dorian_koch.speech_llm import label_hash_legacy

    assert cls_name in label_hash_legacy.LEGACY_LABEL_HASHED, cls_name
    for mod in (
        "persona_selfplay",
        "selfplay_judge",
        "selfplay_bench",
        "selfplay_copy",
        "row_filters",
        "persona_heldout",
        "result_notify",
        "voicebench",
    ):
        m = importlib.import_module(f"{PKG}.{mod}")
        if not hasattr(m, cls_name):
            continue
        orig = m.label_free_hash

        def leaky(cls, parsed_args, canon, base_hash, labels, _orig=orig):
            if cls.__name__ == cls_name:
                canon = dict  # noqa: E731 -- identity view: the labels stay in the hash
            return _orig(cls, parsed_args, canon=canon, base_hash=base_hash, labels=labels)

        m.label_free_hash = leaky
        return
    raise AssertionError(f"no module defines {cls_name}")


def child(mode: str, mutant: str | None, out: str) -> None:
    """Load the real graph (renamed if ``mode == 'rename'``) and dump its (class, hash) multiset."""
    import importlib

    if mutant:
        apply_mutant(mutant)
    if mode == "rename":
        for (mod, attr), new in RENAME.items():
            m = importlib.import_module(f"{DORIANK}.{mod}")
            assert hasattr(m, attr), (mod, attr)
            if attr == "RL_T1_ARM":
                m.RL_T1_SPEC = {new if k == m.RL_T1_ARM else k: v for k, v in m.RL_T1_SPEC.items()}
            setattr(m, attr, new)
    from sisyphus import tk
    from sisyphus.loader import config_manager

    config_manager.load_configs([f"recipe/{DORIANK.replace('.', '/')}/synthetic_train_data.py"])
    for (mod, attr), new in RENAME.items():
        m = sys.modules[f"{DORIANK}.{mod}"]
        assert getattr(m, attr) == (new if mode == "rename" else getattr(m, attr)), (mod, attr)
    from i6_experiments.users.dorian_koch.speech_llm import label_free_hash as lfh

    jobs = list(tk.sis_graph.jobs())
    rows = sorted(f"{type(j).__name__}\t{j._sis_id().rsplit('.', 1)[-1]}" for j in jobs)
    n_free = collections.Counter(type(j).__name__ for j in jobs if lfh.job_is_label_free(j))
    renamed_targets = sorted(t.name for t in tk.sis_graph.targets if PREFIX in t.name)
    json.dump(
        {
            "rows": rows,
            "label_free": n_free,
            "renamed_targets": renamed_targets,
            "collisions": [list(map(str, c)) for c in lfh.label_collisions()],
        },
        open(out, "w"),
    )


def unit() -> None:
    """[1]: every class through its real constructor (sisyphus's JobSingleton -> cls.hash)."""
    from sisyphus import tk

    from i6_experiments.users.dorian_koch.speech_llm import label_free_hash as lfh
    from i6_experiments.users.dorian_koch.speech_llm import label_hash_legacy
    from i6_experiments.users.dorian_koch.speech_llm.persona_heldout import PersonaHeldoutSummary, PersonaHeldoutTurnPos
    from i6_experiments.users.dorian_koch.speech_llm.persona_selfplay import PersonaSelfPlay
    from i6_experiments.users.dorian_koch.speech_llm.result_notify import ResultNotify
    from i6_experiments.users.dorian_koch.speech_llm.row_filters import SelfPlayLexiconRates
    from i6_experiments.users.dorian_koch.speech_llm.selfplay_bench import SelfPlayJudgeV2, SelfPlayJudgeV2Table
    from i6_experiments.users.dorian_koch.speech_llm.selfplay_copy import SelfPlayCopyStats
    from i6_experiments.users.dorian_koch.speech_llm.selfplay_judge import SelfPlayCoherenceJudge
    from i6_experiments.users.dorian_koch.speech_llm.voicebench import VoiceBenchResponses

    P = tk.Path
    x, y, z = P("/guard/x"), P("/guard/y"), P("/guard/z")
    # cls -> builder(labels: (a, b), path) with the labels in every label-carrying slot
    builders = {
        PersonaSelfPlay: lambda L, p: dict(data=p, venv_python_path=x, label_a=L[0], label_b=L[1]),
        SelfPlayCoherenceJudge: lambda L, p: dict(runs={L[0]: p, L[1]: y}),
        SelfPlayJudgeV2: lambda L, p: dict(runs={L[0]: p, L[1]: y}),
        SelfPlayJudgeV2Table: lambda L, p: dict(judges={L[0]: p, L[1]: y}, reference=L[0], order=[L[1], L[0]]),
        SelfPlayCopyStats: lambda L, p: dict(runs={L[0]: p, L[1]: y}, windows=x, corpus=x),
        SelfPlayLexiconRates: lambda L, p: dict(runs={L[0]: p, L[1]: y}),
        PersonaHeldoutSummary: lambda L, p: dict(
            entries=[
                {"run": L[0], "step": 7, "windows_seen": 1, "eval": p},
                {"run": L[1], "step": 7, "windows_seen": 1, "eval": y},
            ]
        ),
        PersonaHeldoutTurnPos: lambda L, p: dict(
            heldout=x, train_config=x, venv_python_path=x, evals={L[0]: (p, 8), L[1]: (y, 8)}
        ),
        ResultNotify: lambda L, p: dict(tag=f"ppx_ft_{L[0]}", results={"knowledge": p}),
        VoiceBenchResponses: lambda L, p: dict(
            venv_python_path=x, in_dir=p, ref_ds=x, model_tag=f"ppx_ft_{L[0]}", subset="sd-qa"
        ),
    }
    check(
        sorted(c.__name__ for c in builders) == sorted(label_hash_legacy.LEGACY_LABEL_HASHED),
        "the guard builds exactly the classes label_hash_legacy lists",
    )
    old, new = (
        ("ab2_old_name", "ab2_old_name_t1.0"),
        ("ab2_a_much_longer_descriptive_name", "ab2_a_much_longer_descriptive_name_t1.0"),
    )

    def hid(cls, kw):
        return cls(**kw)._sis_id().rsplit(".", 1)[-1]

    real_is_legacy = lfh.is_legacy
    for cls, b in builders.items():
        n = cls.__name__
        h_old, h_new, h_path = hid(cls, b(old, x)), hid(cls, b(new, x)), hid(cls, b(old, z))
        check(h_old == h_new, f"{n}: renaming the labels keeps the hash")
        check(h_path != h_old, f"{n}: a different input path is a different job")
        # the label-keyed (pre-2026-10-13) hash of the same arguments, then that hash listed as legacy
        lfh.is_legacy = lambda c, h: True
        try:
            legacy_old, legacy_new = hid(cls, b(old, x)), hid(cls, b(new, x))
        finally:
            lfh.is_legacy = real_is_legacy
        check(legacy_old != h_old, f"{n}: the label-free hash is not the label-keyed one")
        check(legacy_old != legacy_new, f"{n}: CAN FAIL -- the label-keyed hash sees the rename")
        saved = label_hash_legacy.LEGACY_LABEL_HASHED[n]
        label_hash_legacy.LEGACY_LABEL_HASHED[n] = frozenset(saved | {legacy_old})
        try:
            check(hid(cls, b(old, x)) == legacy_old, f"{n}: a legacy job keeps its label-keyed hash")
            check(hid(cls, b(new, x)) == h_new, f"{n}: ... and the same arms renamed are a label-free (new) job")
            # a legacy arm renamed later (RENAMED_RUNS new -> old) keeps its legacy hash; CAN FAIL: without the entry
            saved_map = dict(lfh.RENAMED_RUNS)
            lfh.RENAMED_RUNS.clear()
            lfh.RENAMED_RUNS[new[0]] = old[0]
            try:
                check(hid(cls, b(new, x)) == legacy_old, f"{n}: a legacy arm renamed via RENAMED_RUNS keeps its hash")
                lfh.RENAMED_RUNS.clear()
                check(hid(cls, b(new, x)) != legacy_old, f"{n}: CAN FAIL -- without the RENAMED_RUNS entry it moves")
            finally:
                lfh.RENAMED_RUNS.clear()
                lfh.RENAMED_RUNS.update(saved_map)
        finally:
            label_hash_legacy.LEGACY_LABEL_HASHED[n] = saved

    # label-carrying classes born after 2026-10-13: no legacy hashes, so only the rename / path / can-fail checks
    from i6_experiments.users.dorian_koch.speech_llm.audio_judge import AudioJudgeAgreement

    new_builders = {
        AudioJudgeAgreement: lambda L, p: dict(
            knowledge={
                L[0]: {"replies": p, "transcription": x, "grades": x},
                L[1]: {"replies": y, "transcription": y, "grades": y},
            },  # fmt: skip
            selfplay={L[0]: {"conversations": p, "grades": x}, L[1]: {"conversations": y, "grades": y}},
            question_audio=x,
            venv_python_path=x,
        ),
    }
    for cls, b in new_builders.items():
        n = cls.__name__
        check(n not in label_hash_legacy.LEGACY_LABEL_HASHED, f"{n}: new class, not in label_hash_legacy")
        h_old, h_new, h_path = hid(cls, b(old, x)), hid(cls, b(new, x)), hid(cls, b(old, z))
        check(h_old == h_new, f"{n}: renaming the labels keeps the hash")
        check(h_path != h_old, f"{n}: a different input path is a different job")
        lfh.is_legacy = lambda c, h: True
        try:
            keyed_old, keyed_new = hid(cls, b(old, x)), hid(cls, b(new, x))
        finally:
            lfh.is_legacy = real_is_legacy
        check(keyed_old != keyed_new, f"{n}: CAN FAIL -- the label-keyed hash sees the rename")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", default=None, help="a label-carrying class whose labels leak back in (must FAIL)")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(a.child, a.mutant, a.out)
        return 0

    print("[1] unit: every label-carrying class")
    unit()

    print(
        "[2] real graph: the arms declared from 2026-10-13 on renamed" + (f" (MUTANT {a.mutant})" if a.mutant else "")
    )
    tmp = tempfile.mkdtemp(prefix="check_label_free_hash_", dir="/tmp")
    procs, outs = [], {}
    for mode in ("base", "rename"):
        outs[mode] = os.path.join(tmp, f"{mode}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--child", mode, "--out", outs[mode]]
        if a.mutant:
            cmd += ["--mutant", a.mutant]
        procs.append((mode, subprocess.Popen(cmd)))
    for mode, p in procs:
        check(p.wait() == 0, f"child {mode} loaded the graph")
    if FAILURES and not all(os.path.exists(o) for o in outs.values()):
        print(f"\nFAILED: {len(FAILURES)}")
        return 1
    base, ren = (json.load(open(outs[m])) for m in ("base", "rename"))
    check(
        len(ren["renamed_targets"]) > 0,
        f"the rename took effect ({len(ren['renamed_targets'])} outputs carry {PREFIX})",
    )
    check(not base["renamed_targets"], "the base graph has no guard names")
    cb, cr = collections.Counter(base["rows"]), collections.Counter(ren["rows"])
    moved = sorted((cb - cr).elements())
    check(not moved, f"no hash moved under the rename ({len(base['rows'])} jobs; moved {len(moved)}: {moved[:6]})")
    free = base["label_free"]
    check(
        all(
            free.get(c, 0) > 0 for c in ("PersonaSelfPlay", "SelfPlayCoherenceJudge", "SelfPlayJudgeV2", "ResultNotify")
        ),
        f"label-free jobs in the graph: {dict(free)}",
    )
    check(not base["collisions"] and not ren["collisions"], f"no label-only collisions ({base['collisions'][:3]})")

    print(f"\n{'FAILED: ' + str(len(FAILURES)) if FAILURES else 'all checks passed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
