"""Guard: the ``*_clarified`` self-play reward rubrics (selfplay_bench.SP_REWARD_RUBRICS).

[1] the base rubrics are untouched (their digests are pinned: a running or finished arm ships them by digest);
[2] each clarified rubric = its base with exactly two edits (the silent-agent sentence -> the rules block, the
    30-word reason cap -> "one short sentence"), so the scales are the base's word for word;
[3] the uptake one still carries ``UPTAKE_MARKER`` (the start-up selftest's uptake check keys on it) and every
    placeholder of its base; [4] the 30-word cap is gone and every rule is present.
Run: .venv/bin/python check_clarified_rubric.py (setup env).
"""

import os
import sys

SETUP = "/home/tt201262/setups/2026-01-speech-llm"
sys.path[:0] = [f"{SETUP}/recipe", f"{SETUP}/recipe/sisyphus", f"{SETUP}/recipe/speech_llm/full_duplex"]
os.chdir(SETUP)

from i6_experiments.users.dorian_koch.speech_llm import selfplay_bench as SB  # noqa: E402
from moshi_family.rl.selfplay_reward import UPTAKE_MARKER  # noqa: E402

fails = []


def ck(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


PINNED = {
    "sp_reward_v2": "6bdd42b431aefbbc",
    "sp_reward_v2_uptake": "b363cd2478f30911",
    "sp_reward_v3_timing": "566e407ef01f5873",
}
for name, dig in PINNED.items():
    ck(SB.judge_rubric_digest(name) == dig, f"[1] {name} digest {dig}")

for base in ("sp_reward_v2", "sp_reward_v2_uptake"):
    b, c = SB.SP_REWARD_RUBRICS[base]["text"], SB.SP_REWARD_RUBRICS[f"{base}_clarified"]["text"]
    undo = c.replace(SB._CLARIFIED_RULES, SB._SILENT).replace(SB._REASON_SHORT, SB._REASON_CAP)
    ck(undo == b, f"[2] {base}_clarified = {base} + the two edits only")
    ck(
        SB.SP_REWARD_RUBRICS[f"{base}_clarified"]["metrics"] == SB.SP_REWARD_RUBRICS[base]["metrics"],
        f"[2] {base} metrics",
    )
    for ph in ("{transcript_source}", "{prompt_a}", "{prompt_b}", "{duration}", "{n_a}", "{n_b}", "{transcript}"):
        ck(c.count(ph) == b.count(ph), f"[3] {base}_clarified keeps {ph}")
    ck("30 words" not in c, f"[4] {base}_clarified has no word cap")
    for rule in ("Recognition errors:", "Names:", "Garbled speech:", "Contradictions:", "overrides every other rule"):
        ck(rule in c, f"[4] {base}_clarified has {rule!r}")
ck(UPTAKE_MARKER in SB.SP_REWARD_RUBRICS["sp_reward_v2_uptake_clarified"]["text"], "[3] uptake marker kept")
ck(UPTAKE_MARKER not in SB.SP_REWARD_RUBRICS["sp_reward_v2_clarified"]["text"], "[3] no uptake marker in the v2 one")

if fails:
    raise SystemExit(f"{len(fails)} FAILED: {fails}")
print("all passed")
