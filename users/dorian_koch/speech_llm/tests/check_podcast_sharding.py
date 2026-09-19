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
    EPISODE_SEC_PER_EPISODE_HOUR,
    REJECTED_CHANNEL_MODES,
    _stable_id,
    audio_hours_per_shard,
    duplex_episode_hours_per_shard,
    episode_hours_per_shard,
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

print("[5b] the WHOLE-EPISODE mode is priced separately from the dialogue mode")
whole = audio_hours_per_shard("dialogue_sidon_whole", 4.0)
check("dialogue_sidon_whole == episode_hours_per_shard", whole == episode_hours_per_shard(4.0))
check("dialogue_sidon_whole ~127 input-h/shard", 120.0 < whole < 135.0, f"got {whole:.1f}")
# Non-vacuous, and the whole reason the mode exists: separation runs on 100% of the episode here
# rather than the 42.7% that survives filtering, so reusing the dialogue constant under-shards.
check(
    "it is MEANINGFULLY cheaper per shard than the dialogue path",
    sep / whole > 1.5,
    f"dialogue {sep:.0f} vs whole {whole:.0f} input-h per shard",
)
n_whole = shards_for_hours(JRE_H, channel_mode="dialogue_sidon_whole")
check("JRE whole-episode == 44 shards", n_whole == 44, f"got {n_whole}")
whole_h = (JRE_H / n_whole) * EPISODE_SEC_PER_EPISODE_HOUR / 3600.0
check("...landing within 5% of the 4 h target", whole_h <= 4.0 * 1.05, f"{whole_h:.2f} h")
check(
    "sizing it with the DIALOGUE model would overrun",
    (JRE_H / n_sep) * EPISODE_SEC_PER_EPISODE_HOUR / 3600.0 > 4.0 * 1.15,
    "the two models agree too closely for this to be a real trap",
)
print(f"       -> {n_whole} shards @ {whole_h:.2f} h  (dialogue path: {n_sep} @ {good_h:.2f} h)")

print("[6] retained hours, so a corpus size is never quoted as input hours")
retained = JRE_H * DUPLEX_RETENTION
check("JRE yields ~2,350 dialogue-hours", 2300.0 < retained < 2400.0, f"got {retained:.0f}")
check("per shard ~88 h retained", 80.0 < sep * DUPLEX_RETENTION < 95.0, f"got {sep * DUPLEX_RETENTION:.1f}")
print(f"       -> {JRE_H:.0f} episode-h in, {retained:.0f} dialogue-h out at {100 * DUPLEX_RETENTION:.1f}% retention")

print("[7] episode ids must be the same in EVERY process, for ever")
# The bug: `abs(hash(guid)) % 10**16`. Python salts str hashing per process, so every rebuild of the
# work index minted different ids for the same episodes -- silently breaking resume (episodes.json
# stores ids, so a rebuilt index re-does finished shards) and any attempt to grow a corpus. Invisible
# in practice because within ONE process the ids are perfectly consistent.
import subprocess  # noqa: E402

ids_here = (_stable_id("https://example.com/a.mp3"), _stable_id("guid-abc"))
check("deterministic within the process", ids_here == (
    _stable_id("https://example.com/a.mp3"), _stable_id("guid-abc")))
check("distinct inputs give distinct ids", ids_here[0] != ids_here[1])
check("ids are 16 digits", all(len(i) == 16 and i.isdigit() for i in ids_here), str(ids_here))

prog = (
    "import sys;sys.path.insert(0,%r);sys.path.insert(0,%r);"
    "from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import _stable_id;"
    "print(_stable_id('https://example.com/a.mp3'),_stable_id('guid-abc'))"
) % (str(SETUP / "recipe"), str(SETUP / "recipe" / "sisyphus"))
outs = set()
for _ in range(3):
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                       env={**os.environ, "CUDA_HOME": "/usr"})
    outs.add(r.stdout.strip())
check("identical across 3 separate processes", len(outs) == 1, f"got {outs}")
check("...and matches this process", outs and outs.pop() == " ".join(ids_here))

# Non-vacuous: prove the mechanism the fix replaces really is unstable, so this guard cannot pass
# by testing a property that was never at risk.
old = set()
for _ in range(4):
    r = subprocess.run([sys.executable, "-c", "print(abs(hash('guid-abc')) % (10**16))"],
                       capture_output=True, text=True)
    old.add(r.stdout.strip())
check("the OLD builtin-hash id really was unstable (non-vacuity)", len(old) > 1,
      f"builtin hash gave one value {old} -- PYTHONHASHSEED may be pinned in this environment")

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
