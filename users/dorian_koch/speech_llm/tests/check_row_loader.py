"""Guard: the shared DDP row-loader skeleton shards, reshuffles and skips bad rows correctly.

`batched_row_loader` is the epoch loop PersonaPlex's and MoshiRAG's loaders were collapsed onto. Its
three behaviours all fail *quietly* rather than loudly:

* **Sharding.** Ranks must see disjoint rows. If the per-rank seed offset or the `[rank::world]`
  stride is lost, every rank trains on the same rows -- which converges fine and silently divides the
  effective epoch by `world`. No error, just a worse model than the config claims.
* **Determinism.** One seed must reproduce one epoch order, or a "same config" rerun isn't one.
* **Bad-row skip.** A row that fails to decode is logged and skipped so one corrupt row cannot kill a
  multi-hour run. The cost is that a *systematic* failure looks like slow training, so the skip must
  stay loud in the log and must not swallow the whole epoch silently -- and (2026-09-22) it RAISES
  after ``MAX_CONSECUTIVE_ROW_FAILURES`` consecutive failures or a failure rate above
  ``MAX_ROW_FAILURE_RATE``; the old unlimited skip is re-introduced and shown to spin.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_row_loader.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402

from moshi_family.train_data_common import (  # noqa: E402
    batched_row_loader,
    prefetch_batches,
    shard_rng,
)

NUM_ROWS = 40
table = SimpleNamespace(num_rows=NUM_ROWS)


def item_for(row_index, _rng):
    """A minimal (codes, loss_mask) pair that encodes its own row index in the codes."""
    return torch.full((17, 4), row_index, dtype=torch.long), torch.ones(4, dtype=torch.bool)


def rows_seen(*, rank, world, seed=0, batch_size=2, build=item_for, batches=None):
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
    out = []
    for i, (codes, _mask) in enumerate(
        batched_row_loader(table, build, batch_size=batch_size, seed=seed, infinite=batches is not None)
    ):
        out.extend(int(c[0, 0]) for c in codes)
        if batches is not None and i + 1 >= batches:
            break
    return out


# --- one rank, finite: every row exactly once ----------------------------------------------------
single = rows_seen(rank=0, world=1)
assert sorted(single) == list(range(NUM_ROWS)), f"an epoch must cover every row once, got {len(single)}"
print(f"[ok] a finite epoch covers all {NUM_ROWS} rows exactly once")

# --- two ranks: disjoint, and together they cover the epoch --------------------------------------
r0, r1 = rows_seen(rank=0, world=2), rows_seen(rank=1, world=2)
overlap = set(r0) & set(r1)
assert not overlap, (
    f"ranks share {len(overlap)} rows -- the per-rank seed or the [rank::world] stride was lost, "
    "so every rank would train on the same data and the real epoch would be world-times smaller"
)
assert len(r0) + len(r1) >= NUM_ROWS - 4, f"ranks together saw only {len(r0) + len(r1)}/{NUM_ROWS}"
print(f"[ok] 2 ranks are disjoint          (rank0 {len(r0)}, rank1 {len(r1)}, overlap 0)")

# --- determinism: same seed, same order; different seed, different order --------------------------
assert rows_seen(rank=0, world=1) == single, "same seed gave a different epoch order"
assert rows_seen(rank=0, world=1, seed=7) != single, "different seeds gave the identical order"
print("[ok] one seed reproduces one epoch order")


# --- bad rows are skipped, not fatal, and do not eat the epoch ------------------------------------
def flaky(row_index, rng):
    if row_index % 5 == 0:
        raise ValueError(f"synthetic decode failure on row {row_index}")
    return item_for(row_index, rng)


kept = rows_seen(rank=0, world=1, build=flaky)
assert kept, "every batch was lost -- the skip path swallowed the whole epoch"
assert not any(r % 5 == 0 for r in kept), "a row that raised still reached the batch"
assert len(kept) >= NUM_ROWS - NUM_ROWS // 5 - 2, f"lost more than the failing rows ({len(kept)})"
print(f"[ok] bad rows skipped, rest kept    ({len(kept)}/{NUM_ROWS}, {NUM_ROWS // 5} made to fail)")

# --- infinite mode keeps going past one epoch ------------------------------------------------------
long_run = rows_seen(rank=0, world=1, batch_size=4, batches=15)
assert len(long_run) == 60 > NUM_ROWS, f"infinite loader stopped early ({len(long_run)})"
print(f"[ok] infinite mode reshuffles past the epoch ({len(long_run)} rows from {NUM_ROWS})")

# --- shard_rng reads the env ------------------------------------------------------------------------
os.environ["RANK"], os.environ["WORLD_SIZE"] = "3", "8"
_rng, rank, world = shard_rng(0)
assert (rank, world) == (3, 8), (rank, world)
print("[ok] shard_rng reads RANK/WORLD_SIZE")

# --- prefetch is a no-op on WHAT is trained on ------------------------------------------------------
# The whole claim for the prefetch is "wall clock only". If the batches or their order can differ,
# an A/B on speed is comparing two different experiments and every finished run's meaning is in
# doubt -- so this is asserted on the tensors themselves, not on the row indices.
os.environ["RANK"], os.environ["WORLD_SIZE"] = "0", "1"


def _batches(depth, n=17, seed=0, batch_size=3):
    it = batched_row_loader(table, item_for, batch_size=batch_size, seed=seed)
    it = prefetch_batches(it, depth=depth)
    out = []
    for _ in range(n):
        codes, mask = next(it)
        out.append((codes.clone(), mask.clone()))
    return out


serial = _batches(0)
for _depth in (1, 2, 8):
    pf = _batches(_depth)
    assert len(pf) == len(serial)
    for i, ((c0, m0), (c1, m1)) in enumerate(zip(serial, pf)):
        assert torch.equal(c0, c1), f"depth={_depth}: batch {i} codes differ from the serial loader"
        assert torch.equal(m0, m1), f"depth={_depth}: batch {i} mask differs from the serial loader"
print("[ok] prefetch bit-identical      batches and order unchanged at depth 1, 2, 8")

# Non-vacuous: the comparison above must be capable of failing. A different seed really does
# produce different batches, so equality is a property of the prefetch, not of the fixture.
_other = _batches(0, seed=1)
assert not all(torch.equal(a[0], b[0]) for a, b in zip(serial, _other)), (
    "the fixture yields the same batches for every seed -- the identity check above is vacuous"
)
print("[ok] identity check can fail     a different seed yields different batches")

# depth=0 must be the untouched iterator, so an A/B is a config change and not a code path change.
_raw = batched_row_loader(table, item_for, batch_size=3, seed=0)
assert prefetch_batches(_raw, depth=0) is _raw, "depth=0 wrapped the iterator instead of returning it"
print("[ok] depth=0 is the old path     the iterator is returned untouched")


# A producer that dies must stop the run, not hang it or silently truncate the epoch. Without the
# re-raise the consumer would block on an empty queue forever and the job would burn its walltime
# looking like slow training.
def _explodes():
    yield ("first",)
    raise RuntimeError("synthetic producer failure")


_it = prefetch_batches(_explodes(), depth=2)
assert next(_it) == ("first",)
try:
    next(_it)
except RuntimeError as e:
    assert "synthetic producer failure" in str(e), e
    print("[ok] producer errors re-raised  a dead producer stops the consumer, not hangs it")
else:
    raise SystemExit("FAIL: the prefetch swallowed a producer exception")

# --- a SYSTEMATIC failure stops the run instead of spinning (review 2026-09-22) ---------------------
# The old loop caught every per-row exception forever: with every row failing, an infinite loader
# never yields and never raises -- the job burns its walltime looking like slow training. Driven on
# an arrow table with an `id` column (what the real corpora have), so the log carries the row id.
import logging  # noqa: E402

import pyarrow as pa  # noqa: E402

import moshi_family.train_data_common as tdc  # noqa: E402

id_table = pa.table({"id": [f"ep{i}@a#w0" for i in range(NUM_ROWS)]})


class _Escape(BaseException):
    """Not an Exception: the loader's except cannot swallow it, so a spinning loop can be stopped."""


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def drive(build, *, max_calls=5000, table=id_table):
    """Run an infinite loader; returns ('raised', exc) / ('spun', calls) / ('yielded', n_batches)."""
    calls = [0]

    def counted(i, rng):
        calls[0] += 1
        if calls[0] > max_calls:
            raise _Escape()
        return build(i, rng)

    it = batched_row_loader(table, counted, batch_size=2, seed=0, infinite=True)
    n = 0
    try:
        for _ in it:
            n += 1
            if n >= 1000:
                return "yielded", n
    except RuntimeError as e:
        return "raised", e
    except _Escape:
        return "spun", calls[0]
    return "yielded", n


def always_fails(i, rng):
    raise KeyError(f"variant column missing for row {i}")


cap = _Capture()
tdc.logger.addHandler(cap)
try:
    kind, got = drive(always_fails)
finally:
    tdc.logger.removeHandler(cap)
assert kind == "raised", f"every row failing did not stop the loader ({kind}, {got})"
assert "consecutive" in str(got) and isinstance(got.__cause__, KeyError), got
skips = [m for m in cap.msgs if m.startswith("skip row")]
assert len(skips) == tdc.MAX_CONSECUTIVE_ROW_FAILURES, len(skips)
assert "KeyError" in skips[0] and "@a#w0" in skips[0], f"skip log lacks the exception type / row id: {skips[0]!r}"
print(f"[ok] all rows failing raises       after {tdc.MAX_CONSECUTIVE_ROW_FAILURES} consecutive: {str(got)[:70]}...")

# Frequent but never consecutive: a third of the rows fail. The window rate catches it.
kind, got = drive(lambda i, rng: always_fails(i, rng) if i % 3 == 0 else item_for(i, rng))
assert kind == "raised" and "of the last" in str(got), f"a 1/3 failure rate was tolerated ({kind}, {got})"
print(f"[ok] a 1/3 failure rate raises     {str(got)[:70]}...")

# Rare failures (1 row in 20, 5% < the 10% limit) are still tolerated on an infinite run.
kind, got = drive(lambda i, rng: always_fails(i, rng) if i % 20 == 0 else item_for(i, rng))
assert kind == "yielded", f"a 5% failure rate stopped the run ({kind}, {got})"
print("[ok] 5% bad rows tolerated         1000 batches on an infinite run")

# The log stays bounded: many skips, at most MAX_LOGGED_ROW_SKIPS lines + the cut-off notice.
cap = _Capture()
tdc.logger.addHandler(cap)
try:
    drive(lambda i, rng: always_fails(i, rng) if i % 20 == 0 else item_for(i, rng))
finally:
    tdc.logger.removeHandler(cap)
n_skip_lines = sum(m.startswith("skip row") for m in cap.msgs)
assert n_skip_lines == tdc.MAX_LOGGED_ROW_SKIPS, n_skip_lines
print(f"[ok] skip log bounded              {n_skip_lines} lines for ~100 skips")

# Non-vacuous: with the budget replaced by the OLD unlimited skip, the all-failing loader spins.
_real_budget = tdc.RowSkipBudget


class _OldUnlimitedSkip:
    def ok(self):
        pass

    def failed(self, row_id, exc):
        pass


tdc.RowSkipBudget = _OldUnlimitedSkip
try:
    kind, got = drive(always_fails)
finally:
    tdc.RowSkipBudget = _real_budget
assert kind == "spun", f"guard is vacuous: the old unlimited skip did not spin ({kind})"
print(f"[ok] guard can fail                the old unlimited skip spins ({got} rows, never raises)")

# The base-Moshi / Audex mixed loader has its own epoch loop; it must skip through the same budget (it used to count
# only consecutive failures, so a partly broken corpus in a mix trained on silently with a log line per bad row).
import ast  # noqa: E402

_mtd = os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex", "moshi_family", "moshi_train_data.py")
_fn = next(
    n
    for n in ast.walk(ast.parse(open(_mtd).read()))
    if isinstance(n, ast.FunctionDef) and n.name == "build_mixed_data_loader"
)
_calls = {ast.unparse(n.func) for n in ast.walk(_fn) if isinstance(n, ast.Call)}
assert {"RowSkipBudget", "skips.failed", "skips.ok"} <= _calls, sorted(c for c in _calls if "kip" in c)
_handlers = [h for h in ast.walk(_fn) if isinstance(h, ast.ExceptHandler)]
assert any("skips.failed" in ast.unparse(h) for h in _handlers), "the per-row except no longer goes through the budget"
print("[ok] moshi_train_data mixed loader  skips through RowSkipBudget")

print("\nthe shared row loader shards disjointly, reproduces its order, and survives bad rows")
print("a systematic row failure stops the run with the last exception instead of spinning")
print("prefetch overlaps batch building without changing a single batch")
