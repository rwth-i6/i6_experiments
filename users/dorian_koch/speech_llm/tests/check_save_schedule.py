"""The checkpoint ladder must be the SAME for every run, so two runs are comparable step-for-step.

This is a property of the construction, not a convention: ``geometric_save_steps`` takes no
fraction-of-run parameter, so the ladder cannot drift with ``max_steps``. The guard exists because
the obvious "improvement" -- spreading N points across a run, or scaling `start` with length -- is
exactly what would destroy it, silently, and the damage only shows up later as two runs whose curves
cannot be laid over each other without re-running one of them.

Login node, no cluster, <1 s.
"""

import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")

from speech_llm.full_duplex.sis_recipe.doriank.train_config import (  # noqa: E402
    geometric_save_steps,
    projected_checkpoint_mb,
    LORA_R128_ADAPTER_MB,
)

LENGTHS = (600, 1500, 3000, 7500, 24576)
fails = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# The LADDER is the comparable part (tail_n=1 disables the averaging tail); the full schedule adds
# a run-specific tail anchored to max_steps, which cannot be comparable and is checked separately.
ladders = {n: geometric_save_steps(n, tail_n=1) for n in LENGTHS}
scheds = {n: geometric_save_steps(n) for n in LENGTHS}

# 1. Shared prefix: every run's LADDER, minus its own final step, is a prefix of the longest ladder.
longest = ladders[max(LENGTHS)]
for n in LENGTHS:
    body = tuple(s for s in ladders[n] if s != n)
    check(
        f"{n}-step ladder is a prefix of the longest ladder",
        body == longest[: len(body)],
        f"{body} vs {longest[: len(body)]}",
    )

# 1b. The ladder must survive INSIDE the full schedule -- the tail may add points but must never
#     displace a ladder point, or the comparable steps silently stop existing on disk.
for n in LENGTHS:
    missing = [s for s in ladders[n] if s not in scheds[n]]
    check(f"{n}: every ladder step survives in the full schedule", not missing, str(missing))

# 1c. The averaging tail: at least 3 checkpoints near the end, so the standard "average the last N"
#     (ESPnet 5, Transformer 20) has something to average. The window is 25%, not 15%, because tail
#     spacing is ~3% of the run but floored at the 50-step grid -- on a 600-step run that floor makes
#     the tail span 450..600. Tightening this would silently demand sub-grid checkpoints.
for n in LENGTHS:
    tail = [s for s in scheds[n] if s >= n * 0.75]
    check(f"{n}: >=3 checkpoints in the final 25% (averaging tail)", len(tail) >= 3, str(tail))

# 2. Longer runs never have FEWER checkpoints.
counts = [len(scheds[n]) for n in LENGTHS]
check("checkpoint count is non-decreasing in run length", counts == sorted(counts), str(counts))

# 3. Every point is on a round boundary (user request) and strictly increasing.
for n in LENGTHS:
    s = scheds[n]
    on_grid = all(x % 50 == 0 for x in s if x != n)
    check(f"{n}: non-final points land on 50-step boundaries", on_grid, str(s))
    check(f"{n}: strictly increasing, ends at max_steps", list(s) == sorted(set(s)) and s[-1] == n, str(s))

# 4. Dense where the collapse happens. Every collapse measured in this project began under step 500
#    (a17 between 250 and 450; the 600-step arms flat to 150 then falling), so a ladder with fewer
#    than three points there would not be able to see one.
early = [x for x in ladders[7500] if x <= 500]
check("at least 3 checkpoints at or below step 500", len(early) >= 3, str(early))

# 5. NON-VACUOUS: a max_steps-dependent ladder must FAIL check 1, or the guard proves nothing.
def _bad_ladder(n, points=8):
    return tuple(sorted({max(50, int(round(n * (i + 1) / points / 50)) * 50) for i in range(points)}))

bad = {n: _bad_ladder(n) for n in LENGTHS}
bad_longest = bad[max(LENGTHS)]
bad_is_prefix = all(
    tuple(s for s in bad[n] if s != n) == bad_longest[: len(tuple(s for s in bad[n] if s != n))]
    for n in LENGTHS
)
check("a fraction-of-run ladder would NOT be comparable (guard can fail)", not bad_is_prefix)

# 6. Budget is guidance, not a limit: a very long run must still return a schedule.
try:
    long_sched = geometric_save_steps(200_000)
    check("an over-budget run still returns a schedule (guidance, not a limit)", len(long_sched) > 0)
    print(
        f"      (200k steps -> {len(long_sched)} ckpts, "
        f"~{projected_checkpoint_mb(len(long_sched), adapter_mb=LORA_R128_ADAPTER_MB) / 1000:.1f} GB)"
    )
except Exception as e:  # noqa: BLE001
    check("an over-budget run still returns a schedule (guidance, not a limit)", False, repr(e))

print()
print("FAILURES:", ", ".join(fails) if fails else "none")
raise SystemExit(1 if fails else 0)
