"""Guard: the in-loop held-out eval is the DEFAULT for new PersonaPlex runs, and it moved no existing one (user decision
2026-09-25: "make [held-out validation every 100 steps] the default for all future PersonaPlex runs and the 2 h
ablation helper, without moving hashes of running/finished jobs").

The mechanism is a policy epoch (``runs.RULE_PPX_INLOOP_HELDOUT = POLICY_2026_09_25``): ``track_persona._ppx_arm`` gives
a single-GPU arm declared at that epoch or later ``eval_data`` = the shared held-out set and
``runs.PPX_INLOOP_EVAL_HPARAMS``; ``ablation_2h`` declares new arms at ``AB2H_EVAL_POLICY`` and keeps the arms that
existed before (``AB2H_PRE_INLOOP_EVAL``) at 09-23. Each way this could rot is a check, on the REAL graph:

  [1] every ``_ppx_arm`` (``tp.PPX_ARMS`` / ``tp.PPX_POLICY``): in-loop eval iff (policy >= the rule, 1 GPU, not
      opted out) -- then ``eval_data`` IS the shared held-out set and the hparams carry the cadence; else no eval
      unless the arm passed its own ``eval_data``. p6 and ab2_accum4 have it; the LR sweep and the five earlier ab2_*
      do not (their hashes: hash_snapshot, 0 moved).
  [2] ``ablation_2h``: every arm not in ``AB2H_PRE_INLOOP_EVAL`` is at ``AB2H_EVAL_POLICY`` and has the eval; the
      pinned set is exactly the arms that existed on 2026-09-25 (a new arm silently added there would lose it).
  [3] rendered configs (the adapter's real ``render_config``): ``ab2_accum4`` vs the baseline ``lrsweep_x1`` differ
      ONLY in max_steps / grad_accum / warmup_steps / save_every (+ out_dir and the eval keys), read the same rows
      (steps x accum x batch equal), keep the warmup FRACTION; every eval-default config renders
      ``eval_every: 100`` / ``eval_subset: 128`` / ``eval_full_every: 500``.
  [4] CAN FAIL: the same arm declared at 09-24, or on 2 GPUs, gets NO eval (the gate is real, not always-on).

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_inloop_eval_default.py
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

import yaml  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


def mod(suffix):
    (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
    return m


tp, runs, ab2h, held = (
    mod(".track_persona"),
    mod(".doriank.runs"),
    mod(".track_persona_ab2h"),
    mod(".track_persona_heldout"),
)
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


HELD = str(held.heldout_sets()["heldout"])
CAD = runs.PPX_INLOOP_EVAL_HPARAMS


def has_eval(run):
    return run.eval_data is not None


def expected(tag, run):
    pol = tp.PPX_POLICY[tag]
    gpu = int((run.compute or {}).get("gpu", 1))
    return (
        runs._policy_applies(pol, runs.RULE_PPX_INLOOP_HELDOUT) and gpu == 1 and tag not in tp.PPX_INLOOP_EVAL_OPT_OUT
    )


print("[1] every _ppx_arm: in-loop eval iff policy >= RULE_PPX_INLOOP_HELDOUT, 1 GPU, not opted out")
check(set(tp.PPX_ARMS) == set(tp.PPX_POLICY), f"{len(tp.PPX_ARMS)} arms, each with its policy recorded")
bad = []
for tag, run in sorted(tp.PPX_ARMS.items()):
    if expected(tag, run):
        ok = str(run.eval_data) == HELD and all(run.hparams.get(k) == v for k, v in CAD.items())
    else:
        ok = not has_eval(run)
    if not ok:
        bad.append(tag)
check(not bad, f"eval present exactly where the rule says ({sum(expected(t, r) for t, r in tp.PPX_ARMS.items())} arms)"
      + (f"; wrong: {bad}" if bad else ""))  # fmt: skip
for tag in ("p6_alljre_firsthalf", "ab2_accum4"):
    r = tp.PPX_ARMS.get(tag)
    check(r is not None and str(r.eval_data) == HELD, f"{tag} validates in-loop on the shared held-out set")
for tag in sorted(ab2h.AB2H_PRE_INLOOP_EVAL):
    r = tp.PPX_ARMS.get(tag)
    check(r is not None and not has_eval(r) and tp.PPX_POLICY[tag] == runs.POLICY_2026_09_23, f"{tag}: pinned, no eval")

print("[2] ablation_2h: new arms at AB2H_EVAL_POLICY with the eval; the pinned set is the 2026-09-25 arms only")
PINNED_0925 = {f"lrsweep_x{m}" for m in (1, 4, 16, 64)} | {
    "ab2_firsthalf", "ab2_know30", "ab2_know30_concat", "ab2_pad05", "ab2_usersem",
}  # fmt: skip
check(set(ab2h.AB2H_PRE_INLOOP_EVAL) == PINNED_0925, "AB2H_PRE_INLOOP_EVAL == the arms declared before the default")
for name, run in sorted(ab2h.AB2H_RUNS.items()):
    if name in ab2h.AB2H_PRE_INLOOP_EVAL:
        continue
    check(runs._policy_applies(tp.PPX_POLICY[name], ab2h.AB2H_EVAL_POLICY) and has_eval(run), f"{name}: >= 09-25 epoch, in-loop eval")

print("[3] rendered configs")


def render(run):
    return run.adapter.render_config(run, run.adapter.batch_size, run.max_steps)


x1, acc = tp.PPX_ARMS["lrsweep_x1"], tp.PPX_ARMS["ab2_accum4"]
cx1, cacc = yaml.safe_load(render(x1)), yaml.safe_load(render(acc))
diff_keys = {k for k in set(cx1) | set(cacc) if cx1.get(k) != cacc.get(k)}
allowed = {"max_steps", "grad_accum", "warmup_steps", "save_every", "out_dir", *CAD, "eval_data"}
check(diff_keys <= allowed, f"ab2_accum4 vs lrsweep_x1 differ only in {sorted(diff_keys)}")
check(
    cacc["max_steps"] * cacc["grad_accum"] * cacc["per_gpu_batch"] == cx1["max_steps"] * cx1["grad_accum"] * cx1["per_gpu_batch"],
    f"same rows read: {cacc['max_steps']} x {cacc['grad_accum']} x {cacc['per_gpu_batch']} == "
    f"{cx1['max_steps']} x {cx1['grad_accum']} x {cx1['per_gpu_batch']}",
)  # fmt: skip
check(
    abs(cacc["warmup_steps"] / cacc["max_steps"] - cx1["warmup_steps"] / cx1["max_steps"]) < 1e-3
    and (cacc["lr_temporal"], cacc["lr_depformer"]) == (cx1["lr_temporal"], cx1["lr_depformer"]),
    "same warmup fraction and peak LRs",
)
for tag, run in sorted(tp.PPX_ARMS.items()):
    if expected(tag, run):
        c = yaml.safe_load(render(run))
        check(
            c.get("eval_data") == HELD and all(c.get(k) == v for k, v in CAD.items()),
            f"{tag}: config renders eval_data + {CAD}",
        )
print("  --- ab2_accum4 vs lrsweep_x1 (rendered) ---")
for ln in difflib.unified_diff(render(x1).splitlines(), render(acc).splitlines(), "lrsweep_x1", "ab2_accum4", n=0):
    print("   ", ln)

print("[4] CAN FAIL: the gate")
p6 = tp.PPX_ARMS["p6_alljre_firsthalf"]
hp = {k: v for k, v in p6.hparams.items() if k not in CAD}
kw = dict(hparams=hp, steps=8, benchmark=False)
old = tp._ppx_arm("guard_tmp_0924", p6.train_data, policy=runs.POLICY_2026_09_24, compute={"rqmt_time_h": 1}, **kw)
two = tp._ppx_arm("guard_tmp_2gpu", p6.train_data, policy=runs.POLICY_2026_09_25, compute={"gpu": 2}, **kw)
new = tp._ppx_arm("guard_tmp_0925", p6.train_data, policy=runs.POLICY_2026_09_25, compute={"rqmt_time_h": 1}, **kw)
check(not has_eval(old) and "eval_every" not in old.hparams, "declared at 09-24: no eval")
check(not has_eval(two) and "eval_every" not in two.hparams, "on 2 GPUs: no eval (the in-loop eval is single-process)")
check(str(new.eval_data) == HELD and new.hparams.get("eval_subset") == 128, "declared at 09-25: eval (non-vacuous)")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("all checks passed")
