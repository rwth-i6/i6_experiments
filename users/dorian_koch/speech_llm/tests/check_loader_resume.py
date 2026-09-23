"""Guard: a resumed run must continue the data stream, not restart it.

The bug (found 2026-09-16 on a41, which hit its 24 h wall at step 5,260 of 7,500). ``run_training``
resumes the STEP counter -- ``range(start_step + 1, max_steps + 1)`` -- but the loader was built
fresh and began again at the head of its stream. So the resumed segment re-trained on rows the run
had already consumed and never reached the tail, and **two runs with identical configs differed
purely by whether SLURM preempted them**. Nothing failed; the loss curve was smooth.

Three properties, each a way the fix can rot:

  * **seek** -- a loader built with ``skip_batches=K`` yields exactly what a fresh loader yields from
    its (K+1)-th batch onward. Compared as TENSORS, not row ids.
  * **non-vacuous** -- ``skip_batches=K+1`` must NOT match, so the comparison can fail.
  * **fresh runs are untouched** -- ``skip_batches=0`` is byte-identical to the old call.

Plus a source check that the launcher passes ``start_step * grad_accum`` and not ``start_step``: the
training loop pulls ``grad_accum`` batches per step, so seeking by steps would land a factor of
``grad_accum`` short -- which is a subtler, worse version of the original bug.

Widened 2026-09-23 (code-map §9, bug 4): the fix had landed in the Moshi launcher only; PersonaPlex,
MoshiRAG and Audex restarted their streams, and this check read one file. Now it also drives each of
their REAL loader wrappers (PersonaPlex / MoshiRAG ``build_data_loader``, Audex stage0 / stage1 /
replay-curriculum ``stage1_full_loader``) over a fake table with the decode stubbed, asserting the
same seek / non-vacuous properties, and the source check covers every launcher that calls
``run_training``: it seeks by ``start_step * grad_accum``, after the checkpoint scan, and (outside
base Moshi) only under ``resume_data_position``, whose default is off so old runs resume as trained.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_loader_resume.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.train_data_common import batched_row_loader  # noqa: E402

FAILURES: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


class _Table:
    """The loader only ever reads ``num_rows``; the row itself comes from ``build_item``."""

    def __init__(self, n):
        self.num_rows = n


def _build_item(row_index, rng):
    """A stand-in for the real decode: cheap, but it DRAWS from the per-item rng like the real one.

    That draw is the whole reason the generators had to be split -- with one generator the per-item
    draws advanced the same stream the permutation came from, so the row order after epoch 0 could
    not be reproduced without replaying every decode.
    """
    jitter = float(rng.uniform(0.0, 1.0))
    codes = torch.full((2, 3), float(row_index))
    mask = torch.full((3,), jitter)
    return codes, mask


def batches(n_rows, batch_size, count, *, skip=0, seed=0):
    it = batched_row_loader(_Table(n_rows), _build_item, batch_size=batch_size, seed=seed, skip_batches=skip)
    return [tuple(t.clone() for t in next(it)) for _ in range(count)]


def rows_of(batch):
    """Row ids, recoverable because _build_item stamps them into the codes tensor."""
    return [int(v) for v in batch[0][:, 0, 0].tolist()]


def same(a, b):
    return len(a) == len(b) and all(
        len(x) == len(y) and all(torch.equal(p, q) for p, q in zip(x, y)) for x, y in zip(a, b)
    )


N_ROWS, BS, K, TAIL = 500, 4, 7, 5


def check_seek_matches_uninterrupted():
    fresh = batches(N_ROWS, BS, K + TAIL)
    resumed = batches(N_ROWS, BS, TAIL, skip=K)
    check(same(fresh[K:], resumed), f"skip_batches={K} continues the stream (rows {rows_of(resumed[0])})")
    # The ROWS must match even though the per-item draws are a fresh stream -- state that explicitly,
    # because it is the documented limit of the fix.
    check([rows_of(b) for b in fresh[K:]] == [rows_of(b) for b in resumed], "...and the ROW sequence matches exactly")


def check_non_vacuous():
    fresh = batches(N_ROWS, BS, K + TAIL)
    off_by_one = batches(N_ROWS, BS, TAIL, skip=K + 1)
    check(not same(fresh[K:], off_by_one), "skip_batches=K+1 does NOT match (the comparison is able to fail)")
    # The original bug, re-introduced: no skip at all, compared against the resumed position.
    no_skip = batches(N_ROWS, BS, TAIL, skip=0)
    check(
        not same(fresh[K:], no_skip),
        "skip_batches=0 does NOT match the resumed position (the original bug is detectable)",
    )


def check_fresh_runs_untouched():
    a = batches(N_ROWS, BS, 6, skip=0)
    b = batches(N_ROWS, BS, 6, skip=0)
    check(same(a, b), "skip_batches=0 is deterministic at a fixed seed")
    diff_seed = batches(N_ROWS, BS, 6, skip=0, seed=1)
    check(not same(a, diff_seed), "a different seed really differs (seeding is live)")


def check_seek_across_an_epoch_boundary():
    """The case a41 was actually in: Fisher had wrapped ~1.5 epochs when the wall hit."""
    small = 40  # 10 batches per epoch at BS=4
    per_epoch = small // BS
    k = per_epoch * 2 + 3  # two full epochs plus a bit
    fresh = batches(small, BS, k + 4)
    resumed = batches(small, BS, 4, skip=k)
    check(same(fresh[k:], resumed), f"seek works across epoch boundaries (skip={k}, {per_epoch}/epoch)")


class _RowTable:
    """What the PersonaPlex / MoshiRAG loaders read off a table before decoding: rows and columns."""

    def __init__(self, n):
        self.num_rows = n
        self.column_names = []


def _patched(module, **attrs):
    """Context manager: temporarily replace module attributes (the decode seam), restore after."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        old = {k: getattr(module, k) for k in attrs}
        try:
            for k, v in attrs.items():
                setattr(module, k, v)
            yield
        finally:
            for k, v in old.items():
                setattr(module, k, v)

    return cm()


def _seek_checks(label, make, k=K, tail=TAIL):
    """``make(skip)`` -> a fresh iterator of batches. Seek + non-vacuous, as for batched_row_loader."""

    def take(skip, n):
        it = make(skip)
        return [tuple(t.clone() if torch.is_tensor(t) else t for t in next(it)) for _ in range(n)]

    def eq(a, b):
        return len(a) == len(b) and all(
            len(x) == len(y) and all(torch.equal(p, q) if torch.is_tensor(p) else p == q for p, q in zip(x, y))
            for x, y in zip(a, b)
        )

    fresh = take(0, k + tail)
    check(eq(fresh[k:], take(k, tail)), f"{label}: skip_batches={k} continues the stream")
    check(not eq(fresh[k:], take(k + 1, tail)), f"{label}: skip_batches={k + 1} does NOT match (can fail)")
    check(not eq(fresh[k:], take(0, tail)), f"{label}: no skip does NOT match the resumed position (bug 4 visible)")


def check_personaplex_loader_seeks():
    import moshi_family.personaplex.train_data as ptd

    class Tok:
        n_q = 16

        def build_codes(self, assistant, user, sample_rate, alignments, system_prompt=None, voice_codes=None):
            return torch.full((2, 3), float(assistant)), torch.ones(3, dtype=torch.bool)

    with _patched(
        ptd,
        read_arrow_table=lambda path: _RowTable(N_ROWS),
        is_codes_table=lambda table: False,
        decode_duplex_row=lambda table, i: (i, None, 24000, []),
    ):
        _seek_checks(
            "PersonaPlex build_data_loader",
            lambda skip: ptd.build_data_loader("x", Tok(), batch_size=BS, seed=3, skip_batches=skip),
        )


def check_moshirag_loader_seeks():
    import moshi_family.moshirag_train_data as mrd

    class Tok:
        def build_codes(self, assistant, user, sample_rate, alignments, reference, *, encode_reference, rng):
            return torch.full((2, 3), float(assistant)), torch.ones(3, dtype=torch.bool), torch.zeros(3, 4)

    with _patched(
        mrd, read_arrow_table=lambda path: _RowTable(N_ROWS), decode_duplex_row=lambda t, i: (i, None, 24000, [])
    ):
        _seek_checks(
            "MoshiRAG build_data_loader",
            lambda skip: mrd.build_data_loader("x", Tok(), None, batch_size=BS, seed=3, skip_batches=skip),
        )


def _fake_moshi_build_data_loader(path, tok, *, batch_size, seed=0, infinite=True, skip_batches=0, **_):
    """Stands in for moshi_train_data.build_data_loader (its own seek is exercised by the Moshi runs):
    batch i of the stream is stamped i, so any wrapper that drops ``skip_batches`` shows."""
    i = skip_batches
    while True:
        yield torch.full((batch_size, 17, 3), float(i)), torch.ones(batch_size, 3, dtype=torch.bool)
        i += 1


def check_audex_loaders_seek():
    import moshi_family.audex.train_data as atd
    import moshi_family.moshi_train_data as mtd

    with _patched(mtd, build_data_loader=_fake_moshi_build_data_loader):
        _seek_checks(
            "Audex stage0_mimi_loader",
            lambda skip: atd.stage0_mimi_loader("x", None, text_pad_id=0, batch_size=2, skip_batches=skip),
        )
        _seek_checks(
            "Audex stage1_audex_loader",
            lambda skip: atd.stage1_audex_loader("x", None, batch_size=2, skip_batches=skip),
        )
        _seek_checks(
            "Audex synthetic_stage0_loader",
            lambda skip: atd.synthetic_stage0_loader(num_codebooks=17, card=50, batch_size=2, skip_batches=skip),
        )

        def fake_replay(path, audex_tok, *, text_pad_id=0, batch_size=1, max_tokens=256, seed=0):
            i = 0
            while True:
                yield torch.full((batch_size, 17, 2), -1.0 - i), torch.ones(batch_size, 2, dtype=torch.bool)
                i += 1

        class Tok:
            frame_rate = 12.5
            _audex_tok = None

            def __init__(self):
                self.cfg = type("C", (), {"duration_sec": 20.0})()
                self.num_frames = 250

        def full(skip):
            tok = Tok()
            for kind, codes, mask in atd.stage1_full_loader(
                "x",
                tok,
                text_pad_id=0,
                replay_frac=0.4,
                curriculum=[(0.3, 10.0), (1.0, 20.0)],
                max_steps=30,
                batch_size=2,
                seed=5,
                skip_batches=skip,
            ):
                # the curriculum duration in force for this batch rides along, so a seek that loses the
                # step counter shows too
                yield kind, codes, mask, tok.cfg.duration_sec

        with _patched(atd, text_replay_loader=fake_replay):
            _seek_checks("Audex stage1_full_loader (replay 0.4 + curriculum)", full, k=11, tail=8)


def _launcher_sources():
    root = SETUP / "recipe/speech_llm/full_duplex/moshi_family"
    return {
        str(p.relative_to(root)): p.read_text()
        for p in sorted(root.glob("**/*launcher.py"))
        if "__pycache__" not in p.parts
    }


def seek_problems(src):
    """[] if the launcher builds its loader with skip_batches=<start_step * grad_accum>, after the
    checkpoint scan and before run_training; else what is wrong."""
    import ast

    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def name(f):
        return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")

    skip_names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
            isinstance(b, ast.BinOp)
            and isinstance(b.op, ast.Mult)
            and {getattr(b.left, "id", None), getattr(b.right, "id", None)} == {"start_step", "grad_accum"}
            for b in ast.walk(n.value)
        ):
            skip_names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    seeks = [
        c.lineno
        for c in calls
        for k in c.keywords
        if k.arg == "skip_batches" and isinstance(k.value, ast.Name) and k.value.id in skip_names
    ]
    resume = [c.lineno for c in calls if name(c.func) in ("resume_from_checkpoint", "restore_trainer_state")]
    train = [c.lineno for c in calls if name(c.func) == "run_training"]
    if not seeks:
        return ["no loader call passes skip_batches=<start_step * grad_accum>"]
    out = []
    if not resume or min(seeks) < min(resume):
        out.append("the loader seeks before the checkpoint scan")
    if train and max(seeks) > min(train):
        out.append("the loader seeks after run_training started")
    return out


def check_every_launcher_seeks_by_steps_times_grad_accum():
    srcs = _launcher_sources()
    streamed = {r: s for r, s in srcs.items() if "run_training(" in s}
    check(len(streamed) >= 4, f"{len(streamed)} launchers train through run_training: {sorted(streamed)}")
    for r, src in streamed.items():
        probs = seek_problems(src)
        check(not probs, f"{r}: seeks by start_step * grad_accum after the checkpoint scan {probs or ''}")
        if r != "moshi_finetune_launcher.py":
            # Old runs must resume exactly as they trained: the seek is opt-in, default off.
            check(
                'cfg.get("resume_data_position", False)' in src and "if resume_data_position else 0" in src,
                f"{r}: the seek is gated on resume_data_position (default False)",
            )
    # Can fail: the moshi launcher with the 2026-09-16 fix undone, and one seeking by STEPS.
    moshi = srcs["moshi_finetune_launcher.py"]
    for old, new, what in (
        ("_loader_factory(skip_batches=_skip)", "_loader_factory()", "the loader built without the seek"),
        ("_skip = start_step * grad_accum", "_skip = start_step", "a seek by steps, not batches"),
    ):
        check(old in moshi and seek_problems(moshi.replace(old, new, 1)), f"can fail: {what} is flagged")


def main():
    for fn in (
        check_seek_matches_uninterrupted,
        check_non_vacuous,
        check_fresh_runs_untouched,
        check_seek_across_an_epoch_boundary,
        check_personaplex_loader_seeks,
        check_moshirag_loader_seeks,
        check_audex_loaders_seek,
        check_every_launcher_seeks_by_steps_times_grad_accum,
    ):
        print(f"\n== {fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
