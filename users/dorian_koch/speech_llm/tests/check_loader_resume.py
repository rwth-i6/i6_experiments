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
    it = batched_row_loader(
        _Table(n_rows), _build_item, batch_size=batch_size, seed=seed, skip_batches=skip
    )
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
    check([rows_of(b) for b in fresh[K:]] == [rows_of(b) for b in resumed],
          "...and the ROW sequence matches exactly")


def check_non_vacuous():
    fresh = batches(N_ROWS, BS, K + TAIL)
    off_by_one = batches(N_ROWS, BS, TAIL, skip=K + 1)
    check(not same(fresh[K:], off_by_one),
          "skip_batches=K+1 does NOT match (the comparison is able to fail)")
    # The original bug, re-introduced: no skip at all, compared against the resumed position.
    no_skip = batches(N_ROWS, BS, TAIL, skip=0)
    check(not same(fresh[K:], no_skip),
          "skip_batches=0 does NOT match the resumed position (the original bug is detectable)")


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


def check_launcher_passes_steps_times_grad_accum():
    src = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/moshi_finetune_launcher.py").read_text()
    check("start_step * grad_accum" in src,
          "launcher seeks by start_step * grad_accum (not by steps)")
    check("_loader_factory(skip_batches=" in src,
          "launcher builds the loader AFTER start_step is known")


def main():
    for fn in (
        check_seek_matches_uninterrupted,
        check_non_vacuous,
        check_fresh_runs_untouched,
        check_seek_across_an_epoch_boundary,
        check_launcher_passes_steps_times_grad_accum,
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
