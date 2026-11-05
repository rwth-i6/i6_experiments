"""Guard: the RL2_POLICY_2026_11_02 default (``max_staleness`` 1) reaches new RL V2 arms and no existing one.

``Train.max_staleness`` hashes when non-default, so it is dated (``rl_spec/policy.py``):

  [1] epochs: rl2-2026-10-30 < rl2-2026-11-02 = RL2_POLICY_LATEST, both known.
  [2] ``presets.train`` at rl2-2026-11-02 gives ``max_staleness`` 1, and an explicit ``max_staleness=0`` is kept.
  [3] every earlier epoch gives 0 (the field default), so no declared arm's hash moves (hash_snapshot 2026-11-02: 0).
  [4] CAN FAIL: with ``policy.max_staleness`` mutated to return 0, [2] fails.

Run from the setup root (seconds, no config load, no GPU):
    .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_rl2_2026_11_02.py
"""

import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))

from speech_llm.full_duplex.rl_spec import policy as P  # noqa: E402
from speech_llm.full_duplex.rl_spec import presets  # noqa: E402

fails = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


OLD, NEW = P.RL2_POLICY_2026_10_30, P.RL2_POLICY_2026_11_02
check(
    OLD < NEW <= P.RL2_POLICY_LATEST and {OLD, NEW} <= set(P.KNOWN_POLICIES),
    "[1] rl2-10-30 < rl2-11-02 <= LATEST, known",
)


def train_case():
    t = presets.train(NEW, max_steps=10)
    kept = presets.train(NEW, max_steps=10, max_staleness=0)
    return t.max_staleness == 1 and kept.max_staleness == 0


check(train_case(), "[2] presets.train at rl2-11-02: max_staleness 1; explicit 0 kept")
check(
    all(presets.train(p, max_steps=10).max_staleness == 0 for p in P.KNOWN_POLICIES if p < NEW),
    "[3] every earlier epoch: max_staleness 0 (field default)",
)
real = P.max_staleness
P.max_staleness = lambda policy: 0
try:
    check(not train_case(), "[4] CAN FAIL: policy.max_staleness -> 0 breaks [2]")
finally:
    P.max_staleness = real

print("FAILED: " + "; ".join(fails) if fails else "check_policy_rl2_2026_11_02 OK")
sys.exit(1 if fails else 0)
