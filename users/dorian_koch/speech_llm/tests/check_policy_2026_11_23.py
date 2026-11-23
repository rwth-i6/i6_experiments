"""Guard: the POLICY_2026_11_23 default (``item_rng="per_item"``: a resumed data stream continues its draws) reaches
new Haru instruction-SFT arms and no existing one.

``item_rng`` is a hashed hparam (it changes the draws of a fresh run too), so it is dated (``policies.py``
``RULE_ITEM_RNG_PER_ITEM``, applied in ``track_haru_isft._arm``):

  [1] epochs: 2026-11-05 < 2026-11-23 = POLICY_LATEST, both known.
  [2] ``item_rng_hparams``: per_item at 11-23, {} at every earlier epoch.
  [3] the real ``track_haru_isft._arm``: at 2026-11-05 the full pass builds the running job (``00rxZEWj15xl``, no
      ``item_rng`` in its rendered config); at POLICY_LATEST the same arm is a new job whose rendered config says
      ``item_rng: "per_item"`` (what the launcher reads).
  [4] CAN FAIL: with the track's ``item_rng_hparams`` mutated to ``{}``, [3]'s new arm loses the key.

Run from the setup root (setup venv, ~1 min: builds the corpus jobs, no manager, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_11_23.py
"""

import logging
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
os.chdir(SETUP)
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from speech_llm.full_duplex.sis_recipe.doriank import policies as P  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank import track_haru_isft as T  # noqa: E402

RUNNING = "00rxZEWj15xl"
fails = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


OLD, NEW = P.POLICY_2026_11_05, P.POLICY_2026_11_23
check(OLD < NEW == P.POLICY_LATEST and {OLD, NEW} <= P.KNOWN_POLICIES, "[1] 11-05 < 11-23 = POLICY_LATEST, known")
check(
    P.item_rng_hparams(NEW) == {"item_rng": "per_item"}
    and all(P.item_rng_hparams(p) == {} for p in P.KNOWN_POLICIES if p < NEW),
    "[2] item_rng_hparams: per_item at 11-23, {} before",
)


def full_pass(policy):
    run = T._arm(T.HARU_ISFT_V1_ARM, T.HARU_ISFT_V1_CORPUS, policy=policy, steps=T.HARU_ISFT_V1_STEPS, gpus=4,
                 save_every=250, eval_every=500, heldout_max=512, card=dict(T.COMMON_CARD, change=T.CHANGE, why="guard"))  # fmt: skip
    cfg = run.adapter.render_config(run, run.adapter.batch_size, T.HARU_ISFT_V1_STEPS)
    return run._sis_id().split(".")[-1], cfg


def arms_ok():
    old_id, old_cfg = full_pass(OLD)
    new_id, new_cfg = full_pass(NEW)
    ok = old_id == RUNNING and "item_rng" not in old_cfg and new_id != RUNNING and 'item_rng: "per_item"' in new_cfg
    return ok, f"11-05 -> {old_id}, LATEST -> {new_id}"


ok, why = arms_ok()
check(ok, f"[3] _arm: 11-05 = the running job without item_rng, LATEST = a new job with item_rng per_item ({why})")
real = T.item_rng_hparams
T.item_rng_hparams = lambda policy: {}
try:
    bad, why = arms_ok()
    check(not bad, f"[4] CAN FAIL: item_rng_hparams -> {{}} breaks [3] ({why})")
finally:
    T.item_rng_hparams = real

print("FAILED: " + "; ".join(fails) if fails else "check_policy_2026_11_23 OK")
sys.exit(1 if fails else 0)
