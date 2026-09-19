"""Guard: a podcast shard is sized with the cost model of the mode it will actually RUN.

The bug this exists to prevent is silent by construction. `shards_for_hours` sized every corpus with
`audio_hours_per_shard`, which modelled decode + mimi (+ diarization), and had **no branch for the
separating path** -- while defaulting `channel_mode` to the gating mode. So sizing the duplex corpus
gave 276 input-hours per shard instead of the measured 207: each job then runs ~5.3 h against a 4 h
target, which is still inside the 8 h walltime, so nothing errors, nothing is killed, and the only
symptom is that the target the sharding exists to hit is quietly missed.

Every assertion below is paired with its converse where a one-sided test would pass vacuously.

Run: CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_podcast_sharding.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import (  # noqa: E402
    CHANNEL_MODES,
    DUPLEX_RETENTION,
    DUPLEX_SEC_PER_EPISODE_HOUR,
    REJECTED_CHANNEL_MODES,
    audio_hours_per_shard,
    duplex_episode_hours_per_shard,
    shards_for_hours,
)

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)


print("[1] the separating path delegates to the MEASURED model, not the per-stage one")
sep = audio_hours_per_shard("dialogue_sidon", 4.0)
gate = audio_hours_per_shard("diarize_mask", 4.0)
check("dialogue_sidon == duplex_episode_hours_per_shard", sep == duplex_episode_hours_per_shard(4.0))
check("dialogue_sidon ~207 input-h/shard", 200.0 < sep < 215.0, f"got {sep:.1f}")
check("diarize_mask ~276 input-h/shard", 265.0 < gate < 285.0, f"got {gate:.1f}")
# Non-vacuous: if the two models happened to agree, the delegation above would be untestable and
# the original bug would have been harmless. It is not -- the gap is what made it worth fixing.
check(
    "the two models MATERIALLY disagree (else this guard is vacuous)",
    gate / sep > 1.3,
    f"gating is only {100 * (gate / sep - 1):.0f}% more optimistic",
)

print("[2] the real sizing call, at JRE scale")
JRE_H = 5500.0
n_sep = shards_for_hours(JRE_H, channel_mode="dialogue_sidon")
n_gate = shards_for_hours(JRE_H, channel_mode="diarize_mask")
check("JRE separating == 27 shards (matches the docstring)", n_sep == 27, f"got {n_sep}")
check("would have been under-sharded as gating", n_gate < n_sep, f"{n_gate} vs {n_sep}")
# The consequence, stated in the unit that matters: hours per job.
bad_h = (JRE_H / n_gate) * DUPLEX_SEC_PER_EPISODE_HOUR / 3600.0
good_h = (JRE_H / n_sep) * DUPLEX_SEC_PER_EPISODE_HOUR / 3600.0
check("correct sizing lands within 5% of the 4 h target", good_h <= 4.0 * 1.05, f"{good_h:.2f} h")
check("the old sizing overshot the target", bad_h > 4.0 * 1.15, f"{bad_h:.2f} h")
print(f"       -> {n_sep} shards @ {good_h:.2f} h  vs  {n_gate} shards @ {bad_h:.2f} h (the bug)")

print("[3] the mode cannot be omitted or mistyped into the cheapest model")
try:
    shards_for_hours(100.0)  # type: ignore[call-arg]
    check("channel_mode is required", False, "call succeeded")
except TypeError:
    check("channel_mode is required", True)
try:
    audio_hours_per_shard("dialoguesidon", 4.0)  # plausible typo: no underscore
    check("unknown mode raises", False, "typo silently accepted")
except ValueError:
    check("unknown mode raises", True)
# A typo must not fall through to the CHEAPEST model, which would under-shard the worst.
check("every declared mode is priced", all(audio_hours_per_shard(m, 4.0) > 0 for m in CHANNEL_MODES))

print("[4] an explicit cap raises with the arithmetic rather than rounding the problem away")
try:
    shards_for_hours(JRE_H, channel_mode="dialogue_sidon", max_shards=10)
    check("max_shards raises", False, "silently accepted")
except ValueError as e:
    check("max_shards raises", True)
    check("the error carries the numbers", "27" in str(e) and "10" in str(e), str(e)[:90])

print("[5] the rejected mode is declared, so a caller can refuse it")
check("diarize_mask is marked rejected", "diarize_mask" in REJECTED_CHANNEL_MODES)
check("dialogue_sidon is NOT rejected", "dialogue_sidon" not in REJECTED_CHANNEL_MODES)
check("stereo_passthrough is NOT rejected (dual-channel sources)", "stereo_passthrough" not in REJECTED_CHANNEL_MODES)

print("[6] retained hours, so a corpus size is never quoted as input hours")
retained = JRE_H * DUPLEX_RETENTION
check("JRE yields ~2,350 dialogue-hours", 2300.0 < retained < 2400.0, f"got {retained:.0f}")
check("per shard ~88 h retained", 80.0 < sep * DUPLEX_RETENTION < 95.0, f"got {sep * DUPLEX_RETENTION:.1f}")
print(f"       -> {JRE_H:.0f} episode-h in, {retained:.0f} dialogue-h out at {100 * DUPLEX_RETENTION:.1f}% retention")

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
