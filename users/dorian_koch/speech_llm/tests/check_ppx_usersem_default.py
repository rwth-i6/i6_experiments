"""Guard: user semantic weight 1.0 is the DEFAULT for new PersonaPlex runs, the 2-hour ablations declared from then on are
one change from ``ab2_usersem``, and no existing arm moved (2026-09-25).

Mechanism: policy epoch ``runs.RULE_PPX_USER_SEMANTIC = POLICY_2026_09_25B`` ("2026-09-25b"; policies compare as
strings, so it is after "2026-09-25"); ``track_persona._ppx_arm`` adds ``user_semantic_weight`` =
``PPX_USER_SEMANTIC_WEIGHT`` unless the hparams name one; ``ablation_2h`` declares new arms at ``AB2H_USERSEM_POLICY``,
keeps ``AB2H_PRE_USERSEM`` at 09-25, and ``ab2h_baseline`` names their baseline. On the REAL graph:

  [1] every ``_ppx_arm``: ``user_semantic_weight`` == 1.0 where the policy is >= the rule; below it only
      ``ab2_usersem`` (which names it itself) carries the key.
  [2] ``ablation_2h``: the arms declared at 09-25 (``AB2H_PRE_USERSEM``) keep 09-25 and no user semantic key, baseline
      lrsweep_x1; every newer arm is at 09-25b, baseline ``ab2_usersem``; ``ab2_turntail03``'s rendered config vs
      ``ab2_usersem``'s differs ONLY in out_dir, the in-loop eval keys (the 09-25 default) and ``turn_tail_*``.
  [3] judge groups: ``ab2_turntail03``'s coherence judge rates ``ab2_usersem``'s self-play, not lrsweep_x1's.
  [4] CAN FAIL: the same arm declared at 09-25 gets no key; at 09-25b gets 1.0; an explicit 0.02 is kept.

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_usersem_default.py
"""

import difflib
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

from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


def mod(suffix):
    (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
    return m


tp, runs, ab2h = mod(".track_persona"), mod(".doriank.runs"), mod(".track_persona_ab2h")
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


print("[1] every _ppx_arm: user_semantic_weight 1.0 iff policy >= RULE_PPX_USER_SEMANTIC")
check(runs.POLICY_2026_09_25 < runs.POLICY_2026_09_25B, "09-25b sorts after 09-25")
check(runs.RULE_PPX_USER_SEMANTIC in runs.KNOWN_POLICIES and runs.PPX_USER_SEMANTIC_WEIGHT == 1.0, "rule + value")
bad = []
n_new = 0
for tag, run in sorted(tp.PPX_ARMS.items()):
    new = runs._policy_applies(tp.PPX_POLICY[tag], runs.RULE_PPX_USER_SEMANTIC)
    n_new += new
    usw = run.hparams.get("user_semantic_weight")
    if (new and usw != 1.0) or (not new and usw is not None and tag != "ab2_usersem"):
        bad.append(tag)
check(
    not bad and n_new > 0,
    f"{n_new} arms at the new epoch carry 1.0, older arms none" + (f"; wrong: {bad}" if bad else ""),
)

print("[2] ablation_2h: epochs and baselines")
for name in sorted(ab2h.AB2H_PRE_USERSEM):
    r = tp.PPX_ARMS.get(name)
    check(
        r is not None
        and tp.PPX_POLICY[name] == runs.POLICY_2026_09_25
        and "user_semantic_weight" not in r.hparams
        and ab2h.ab2h_baseline(name) == ab2h.AB2H_BASELINE,
        f"{name}: pinned at 09-25, no user semantic key, baseline lrsweep_x1",
    )
for name in sorted(ab2h.AB2H_RUNS):
    if name in ab2h.AB2H_PRE_USERSEM or name in ab2h.AB2H_PRE_INLOOP_EVAL:
        continue
    if tp.PPX_POLICY[name] > runs.POLICY_2026_09_25B:
        continue  # a later epoch (09-25c / 09-28 / 09-28b: their own guards); [1] checks its user semantic 1.0
    check(
        tp.PPX_POLICY[name] == runs.POLICY_2026_09_25B and ab2h.ab2h_baseline(name) == ab2h.AB2H_BASELINE_USERSEM,
        f"{name}: 09-25b, baseline ab2_usersem",
    )


def render(run):
    return run.adapter.render_config(run, run.adapter.batch_size, run.max_steps).splitlines()


base, arm = ab2h.AB2H_RUNS["ab2_usersem"], ab2h.AB2H_RUNS.get("ab2_turntail03")
check(arm is not None, "ab2_turntail03 declared")
if arm is not None:
    diff = [
        ln
        for ln in difflib.unified_diff(render(base), render(arm), n=0, lineterm="")
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]
    keys = {ln[1:].split(":")[0] for ln in diff}
    allowed = {
        "out_dir",
        "eval_data",
        "eval_every",
        "eval_subset",
        "eval_full_every",
        "turn_tail_weight",
        "turn_tail_s",
    }
    check(
        keys <= allowed and {"turn_tail_weight", "turn_tail_s"} <= keys, f"turntail vs usersem config: {sorted(keys)}"
    )
    for ln in diff:
        print("     ", ln[:160])

print("[3] judge groups")
from sisyphus import tk  # noqa: E402

judges = [
    j
    for j in tk.sis_graph.jobs()
    if type(j).__name__ == "SelfPlayCoherenceJudge" and "ab2_turntail03_minimal" in getattr(j, "runs", {})
]
check(len(judges) == 1, "one coherence judge rates ab2_turntail03")
if judges:
    rated = set(judges[0].runs)
    check(
        "ab2_usersem_minimal" in rated and "lrsweep_x1_minimal" not in rated,
        "it rates ab2_usersem's self-play, not lrsweep_x1's",
    )
check(ab2h._GROUP_BASE.get("ab2_turntail03") == ab2h.AB2H_BASELINE_USERSEM, "turntail's group baseline is ab2_usersem")

print("[4] CAN FAIL: the gate")
x1 = ab2h.AB2H_RUNS["lrsweep_x1"]
hp = {k: v for k, v in x1.hparams.items() if k != "user_semantic_weight"}
kw = dict(hparams=hp, steps=8, benchmark=False, compute={"rqmt_time_h": 1})
old = tp._ppx_arm("guard_usw_0925", x1.train_data, policy=runs.POLICY_2026_09_25, **kw)
new = tp._ppx_arm("guard_usw_0925b", x1.train_data, policy=runs.POLICY_2026_09_25B, **kw)
own = tp._ppx_arm("guard_usw_own", x1.train_data, {"user_semantic_weight": 0.02}, policy=runs.POLICY_2026_09_25B, **kw)
check("user_semantic_weight" not in old.hparams, "declared at 09-25: no key")
check(new.hparams.get("user_semantic_weight") == 1.0, "declared at 09-25b: 1.0 (non-vacuous)")
check(own.hparams.get("user_semantic_weight") == 0.02, "an explicit value is kept")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("all checks passed")
