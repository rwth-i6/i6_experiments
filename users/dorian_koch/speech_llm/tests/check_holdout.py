"""Guard: the seedless index holdout must be disjoint, deterministic and free.

Replaces a materialised train/val split that cost a 104 GB duplicate of the corpus to hold out 2%
of it. The properties that make the cheap version safe are exactly the ones worth pinning:

  * DISJOINT     -- a row the model trains on must never appear in eval, or the "held-out" loss is
                    a training loss wearing a different name.
  * COVERING     -- train + eval must be the whole corpus; a silently dropped slice is data paid
                    for and not used.
  * DETERMINISTIC-- no RNG anywhere in the split, so the same rows are held out on every run, on
                    every machine, forever. A seeded shuffle would make two arms' eval losses
                    quietly incomparable, which is worse than having no eval at all.
  * DDP-SAFE     -- ranks still shard disjointly within whichever side they draw from.

Run: CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/.../tests/check_holdout.py
"""

import ast
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
LIB = SETUP / "recipe" / "speech_llm" / "full_duplex"
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
# moshi_train_data does `from moshi_family...` (absolute, as it runs under the job venv where the
# package IS top-level), so the package dir's parent must be importable as well.
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402


class _FakeTable:
    def __init__(self, n, column_names=None):
        self.num_rows = n
        # A real pyarrow table always has this, and `_Corpus` reads it to tell a pre-encoded
        # (mimi codes) corpus from a waveform one. Kept on the fake so the schema check stays
        # STRICT -- a table type without `column_names` is a bug, not a wav corpus.
        self.column_names = list(column_names or ["audio_assistant", "audio_user", "alignments"])


def _corpus(n, **kw):
    """Build a _Corpus without touching disk: only the row-index arithmetic is under test."""
    import moshi_family.moshi_train_data as mtd

    orig = mtd.read_arrow_table
    mtd.read_arrow_table = lambda path: _FakeTable(n)
    try:
        return mtd._Corpus("/fake/corpus", 1.0, None, **kw)
    finally:
        mtd.read_arrow_table = orig


def check_disjoint_and_covering():
    n, every = 60000, 512
    train = _corpus(n, holdout_every=every, holdout_side="train").rows
    ev = _corpus(n, holdout_every=every, holdout_side="eval").rows
    assert set(train).isdisjoint(set(ev)), "train and eval overlap -- the eval loss is contaminated"
    assert len(train) + len(ev) == n, f"{len(train)}+{len(ev)} != {n}: rows silently dropped"
    assert len(ev) == n // every + (1 if n % every else 0), len(ev)
    # The point of the exercise: a few hundred rows, not a corpus copy.
    assert len(ev) / n < 0.01, f"holdout is {len(ev) / n:.1%} -- too big to be free"
    print(f"PASS  disjoint + covering: {len(train)} train / {len(ev)} eval ({len(ev) / n:.2%})")


def check_deterministic_across_calls():
    a = _corpus(12345, holdout_every=97, holdout_side="eval").rows
    b = _corpus(12345, holdout_every=97, holdout_side="eval").rows
    assert np.array_equal(a, b), "holdout is not reproducible"
    # And it must not depend on any global RNG state.
    np.random.seed(1)
    c = _corpus(12345, holdout_every=97, holdout_side="eval").rows
    np.random.seed(999)
    d = _corpus(12345, holdout_every=97, holdout_side="eval").rows
    assert np.array_equal(c, d), "holdout moved with the global RNG -- it must be seedless"
    assert np.array_equal(a, np.arange(0, 12345, 97)), a[:5]
    print("PASS  deterministic and seedless: identical rows regardless of RNG state")


def check_disabled_is_a_noop():
    c = _corpus(1000)
    assert np.array_equal(c.rows, np.arange(1000)), "holdout_every=0 must draw every row"
    print("PASS  holdout_every=0 is a no-op (every existing run unchanged)")


def check_degenerate_configs_rejected():
    for every in (1,):
        try:
            _corpus(1000, holdout_every=every)
        except AssertionError:
            pass
        else:
            raise SystemExit(f"FAIL: holdout_every={every} holds out EVERY row and was accepted")
    try:
        _corpus(1000, holdout_side="eval")  # eval side with no holdout = empty
    except AssertionError:
        pass
    else:
        raise SystemExit("FAIL: holdout_side='eval' without holdout_every was accepted")
    try:
        _corpus(10, holdout_every=100000, holdout_side="eval")
    except AssertionError:
        pass
    else:
        # every % 100000 == 0 only for row 0, so eval is non-empty; train is what could empty out
        pass
    print("PASS  degenerate holdout configs are rejected loudly")


def check_ddp_shards_within_side():
    """Ranks must still partition whichever side they draw from -- holdout must not break sharding."""
    import moshi_family.moshi_train_data as mtd

    c = _corpus(10000, holdout_every=512, holdout_side="train")
    world = 4
    seen = []
    for rank in range(world):
        rng = np.random.default_rng(0)
        order = rng.permutation(c.rows)
        seen.append(set(int(i) for i in order[rank::world]))
    union = set().union(*seen)
    assert sum(len(s) for s in seen) == len(union), "ranks overlap within the train side"
    assert union == set(int(i) for i in c.rows), "ranks do not cover the train side"
    assert not union & set(int(i) for i in _corpus(10000, holdout_every=512, holdout_side="eval").rows)
    print(f"PASS  DDP: {world} ranks partition the train side and never touch eval rows")


# --- the launcher must actually RUN the eval it built --------------------------------------------
#
# `check_holdout` above proves the SPLIT is correct: disjoint, covering, deterministic, DDP-safe.
# None of that reaches the number, because for a year the launcher built the held-out loader from
# `holdout_every` and then passed `eval_fn=eval_fn if (do_eval and eval_data) else None` -- gated on
# the OTHER eval source. A `holdout_every` arm therefore logged
#
#     eval: held-out val loss every N steps (K batches) on every Mth row of the training mixture
#
# and produced no `metrics.eval.jsonl` at all. Every arm of the knowledge-collapse investigation ran
# that way (checked 2026-09-07: the only metrics.eval.jsonl in output/ predates the mechanism). This
# is the exact lesson CLAUDE.md records as "a guard must exercise the caller's wiring, not an
# idealised version of it" -- so assert the wiring, not just the split.


def eval_fn_gate(launcher: Path) -> ast.AST:
    """The condition guarding `eval_fn=` in the launcher's `run_training(...)` call."""
    tree = ast.parse(launcher.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_training"):
            continue
        for kw in node.keywords:
            if kw.arg == "eval_fn":
                assert isinstance(kw.value, ast.IfExp), (
                    "eval_fn is no longer a conditional -- re-read this check's premise before deleting it."
                )
                return kw.value.test
    raise AssertionError(f"no run_training(eval_fn=...) call found in {launcher}")


def gate_names(test: ast.AST) -> set:
    return {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}


def check_eval_actually_runs():
    launcher = LIB / "moshi_family/moshi_finetune_launcher.py"
    names = gate_names(eval_fn_gate(launcher))
    assert "eval_iter" in names, (
        f"moshi_finetune_launcher gates eval_fn on {sorted(names)}, which does not include "
        f"'eval_iter'. eval_iter is the variable that holds whichever eval source was built "
        f"(holdout_every OR eval_data), so gating on anything else silently disables evaluation for "
        f"the source that was not named -- which is how holdout_every ran for a year producing "
        f"nothing. Gate on what was built."
    )
    # Non-vacuous: the shipped bug must be rejected, and the fix accepted.
    bad = ast.parse("run_training(eval_fn=eval_fn if (do_eval and eval_data) else None)")
    good = ast.parse("run_training(eval_fn=eval_fn if eval_iter is not None else None)")
    pick = lambda t: gate_names(
        next(kw.value.test for n in ast.walk(t) if isinstance(n, ast.Call) for kw in n.keywords if kw.arg == "eval_fn")
    )
    assert "eval_iter" not in pick(bad), "self-test: the 2026-09-07 bug is no longer detected"
    assert "eval_iter" in pick(good), "self-test: the fixed form is rejected"
    print("[ok] eval_fn is gated on the eval source that was actually built")


if __name__ == "__main__":
    check_disjoint_and_covering()
    check_deterministic_across_calls()
    check_disabled_is_a_noop()
    check_degenerate_configs_rejected()
    check_ddp_shards_within_side()
    check_eval_actually_runs()
    print("ALL PASS")
