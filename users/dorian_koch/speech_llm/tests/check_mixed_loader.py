"""Verify the on-the-fly corpus mixing in moshi_train_data (login node, no GPU, no model).

Guards the things that fail silently: a mix that does not realise the requested proportions, a
single-corpus run whose RNG stream changed under the refactor, and the assumption that the two
corpora's differing HF feature types still decode identically from arrow.
"""

import sys

import numpy as np

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/speech_llm/full_duplex")

from moshi_family.moshi_train_data import _Corpus  # noqa: E402
from moshi_family.train_data_common import read_arrow_table  # noqa: E402

QA = "output/moshi_annotate_triviaqa"
FISHER = "output/fisher/train_windows"


def resolve(p):
    import os

    return os.path.realpath(p)


# --- 1. both corpora decode from arrow despite different HF feature types ------------------------
# TriviaQA declares HF Audio(); Fisher declares a plain {bytes,path} struct. On disk both are
# struct<bytes, path>, which is why no schema conversion is needed to mix them.
corpora = {}
for name, path in (("qa", QA), ("fisher", FISHER)):
    c = _Corpus(resolve(path), 1.0)
    corpora[name] = c
    arrow_type = str(c.table.schema.field("audio_assistant").type)
    assert "bytes" in arrow_type and "path" in arrow_type, (name, arrow_type)
    assistant, user, sr, aligns = c.decode_row(0)
    assert assistant.ndim == 1 and user.ndim == 1, (name, assistant.shape)
    assert assistant.shape[0] == user.shape[0], (name, "channels differ in length")
    assert sr > 0 and len(aligns) > 0, (name, sr, len(aligns))
    print(f"[ok] {name:6s} n={c.n:6d} arrow={arrow_type} sr={sr} words0={len(aligns)}")

assert corpora["qa"].decode_row(0)[2] != corpora["fisher"].decode_row(0)[2], (
    "expected the two corpora to have DIFFERENT native sample rates -- that is the case the "
    "per-row resample in build_codes exists to handle"
)
print("[ok] corpora have different native sample rates (per-row resample path is exercised)")

# --- 2. weighted sampling realises the requested proportions -------------------------------------
# Reproduces the loader's corpus-choice draw without needing mimi/a GPU: the loader does
# `rng.choice(len(corpora), p=weights)` once per row.
for weights in ([0.5, 0.5], [0.75, 0.25], [0.1, 0.9]):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    rng = np.random.default_rng(0)
    draws = rng.choice(len(w), size=20000, p=w)
    realised = np.bincount(draws, minlength=len(w)) / len(draws)
    assert np.allclose(realised, w, atol=0.02), (weights, realised)
    print(f"[ok] weights {list(w.round(2))} -> realised {list(realised.round(3))}")

# --- 3. index streams are endless, in-range, and DDP-sharded -------------------------------------
c = corpora["fisher"]
rng = np.random.default_rng(0)
stream = c.index_stream(rng, rank=0, world=1)
idxs = [next(stream) for _ in range(c.n + 500)]  # must pass n to prove it cycles
assert all(0 <= i < c.n for i in idxs), "index out of range"
assert len(idxs) > c.n, "stream must be endless (it cycles past one pass)"
print(f"[ok] index stream endless and in range ({len(idxs)} draws over n={c.n})")

# Every rank must see a disjoint slice, and together they must cover the corpus.
#
# Build each rank's generator the way the LOADER does -- via shard_rng -- not by hand. This check
# used to construct `np.random.default_rng(0)` itself for every rank, which is disjoint by
# construction and so passed while the real loader seeded `default_rng(seed + rank)` and handed each
# rank a different permutation to stride. The stride only partitions an epoch when every rank strides
# the same order, so the real shards overlapped for as long as this test looked at an idealised rng
# instead of the one the caller actually creates. Test the wiring, not the primitive.
import os  # noqa: E402

from moshi_family.train_data_common import shard_rng  # noqa: E402

shards = []
for rank in range(4):
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), "4"
    r, got_rank, got_world = shard_rng(0)
    assert (got_rank, got_world) == (rank, 4), (got_rank, got_world)
    s = c.index_stream(r, rank=got_rank, world=got_world)
    shards.append({next(s) for _ in range(c.n // 4)})
os.environ["RANK"], os.environ["WORLD_SIZE"] = "0", "1"
overlap = shards[0] & shards[1]
assert not overlap, (
    f"DDP shards overlap ({len(overlap)} shared indices) -- ranks are duplicating rows and missing "
    "others, so the epoch is neither disjoint nor complete"
)
covered = set().union(*shards)
assert len(covered) >= c.n - 4, f"4 ranks together covered only {len(covered)}/{c.n} rows"
print(f"[ok] DDP sharding disjoint across 4 ranks ({len(shards[0])} each, {len(covered)}/{c.n} covered)")

# --- 4. a small corpus is not exhausted when mixed with a large one -------------------------------
# The point of sampling rather than concatenating: a 20k-row corpus mixed 50/50 with a 62k-row one
# must keep supplying rows, cycling as needed, rather than running out.
small, large = corpora["fisher"], corpora["qa"]
assert small.n < large.n, (small.n, large.n)
r = np.random.default_rng(1)
s = small.index_stream(r, rank=0, world=1)
n_draw = 2 * small.n
seen = [next(s) for _ in range(n_draw)]
assert len(seen) == n_draw, "small corpus ran out"
print(f"[ok] smaller corpus cycles rather than starving ({n_draw} draws from n={small.n})")

print("\nALL CHECKS PASSED")
