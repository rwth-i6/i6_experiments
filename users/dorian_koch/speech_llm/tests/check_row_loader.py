"""Guard: the shared DDP row-loader skeleton shards, reshuffles and skips bad rows correctly.

`batched_row_loader` is the epoch loop PersonaPlex's and MoshiRAG's loaders were collapsed onto. Its
three behaviours all fail *quietly* rather than loudly:

* **Sharding.** Ranks must see disjoint rows. If the per-rank seed offset or the `[rank::world]`
  stride is lost, every rank trains on the same rows -- which converges fine and silently divides the
  effective epoch by `world`. No error, just a worse model than the config claims.
* **Determinism.** One seed must reproduce one epoch order, or a "same config" rerun isn't one.
* **Bad-row skip.** A row that fails to decode is logged and skipped so one corrupt row cannot kill a
  multi-hour run. The cost is that a *systematic* failure looks like slow training, so the skip must
  stay loud in the log and must not swallow the whole epoch silently.

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

print("\nthe shared row loader shards disjointly, reproduces its order, and survives bad rows")
print("prefetch overlaps batch building without changing a single batch")
