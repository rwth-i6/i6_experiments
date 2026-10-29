"""Guard: the POLICY_2026_09_25C defaults (2026-09-25) -- for NEW PersonaPlex arms AdamW weight decay 1e-4
and the ``ab2_know30_concat`` knowledge mix; for new runs of every family through ``_ppx_arm`` / ``make_finetune_run``
no gradient accumulation -- and no existing arm moved.

Mechanism: ``runs.RULE_PPX_WEIGHT_DECAY`` / ``RULE_PPX_KNOWLEDGE_MIX`` / ``RULE_NO_GRAD_ACCUM`` = "2026-09-25c";
``track_persona._ppx_arm`` fills ``weight_decay``, ``PPX_KNOWLEDGE_DEFAULT`` (registered by
``register_persona_knowledge_mix``) and ``grad_accum`` 1; ``ablation_2h`` declares new arms at ``AB2H_C_POLICY`` with
baseline ``AB2H_BASELINE_C``. On the REAL graph:

  [1] epochs: 09-25b < 09-25c = POLICY_LATEST; no declared arm is at 09-25c yet; ab2_turntail03 pinned at 09-25b with
      baseline ab2_usersem; a new 2-hour name is at 09-25c with baseline ab2_base_c (whose own baseline is ab2_usersem).
  [2] the default knowledge mix == ab2_know30_concat's four knowledge keys (v6 codes, share 0.3, meta, ORIGINAL scheme).
  [3] a probe arm at 09-25c (in memory, never registered for a manager) carries wd 1e-4, grad_accum 1, usersem 1.0, the
      knowledge mix and the in-loop eval; grad_accum 4 / PERSONA_HPARAMS / a clash key without opt-out raise; an
      opted-out arm trains without the mix; ``make_finetune_run`` at 09-25c refuses a missing or > 1 grad_accum and a
      PersonaPlex adapter.
  [4] CAN FAIL: the same probe at 09-25b gets none of it; with the knowledge default unregistered a 09-25c arm raises.

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_09_25c.py
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

from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


def mod(suffix):
    (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
    return m


tp, runs, ab2h, tc = (
    mod(".track_persona"),
    mod(".doriank.runs"),
    mod(".track_persona_ab2h"),
    mod(".train_config"),
)
FAILURES = []
KNOW_KEYS = ("knowledge_data", "knowledge_share", "knowledge_meta", "knowledge_concat")


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


def raises(fn, exc=(ValueError, AssertionError, RuntimeError)):
    try:
        fn()
    except exc as e:
        return str(e)[:120]
    return None


C, B = runs.POLICY_2026_09_25C, runs.POLICY_2026_09_25B
print("[1] epochs, pins, baselines")
check(B < C <= runs.POLICY_LATEST and C in runs.KNOWN_POLICIES, "09-25b < 09-25c <= POLICY_LATEST, known")
check(
    runs.RULE_PPX_WEIGHT_DECAY == runs.RULE_PPX_KNOWLEDGE_MIX == runs.RULE_NO_GRAD_ACCUM == C
    and runs.PPX_WEIGHT_DECAY == 1e-4
    and runs.PPX_KNOWLEDGE_SHARE == 0.3,
    "rules at 09-25c, wd 1e-4, share 0.3",
)
at_c = sorted(t for t, p in tp.PPX_POLICY.items() if p == C)  # later epochs: check_policy_2026_09_28 / _28b
check(not at_c, f"no declared arm at exactly 09-25c ({at_c})")
check(
    tp.PPX_POLICY.get("ab2_turntail03") == B and ab2h.ab2h_baseline("ab2_turntail03") == ab2h.AB2H_BASELINE_USERSEM,
    "ab2_turntail03 pinned at 09-25b, baseline ab2_usersem",
)
check(
    ab2h.ab2h_policy(ab2h.AB2H_BASELINE_C) == C
    and ab2h.ab2h_baseline(ab2h.AB2H_BASELINE_C) == ab2h.AB2H_BASELINE_USERSEM,
    "ab2_base_c: 09-25c, baseline ab2_usersem (new 2-hour arms moved on to 09-28: check_policy_2026_09_28)",
)
wd_arms = sorted(t for t, r in tp.PPX_ARMS.items() if r.hparams.get("weight_decay") and tp.PPX_POLICY.get(t, "") < C)
check(wd_arms == ["ab2_wd"], f"of the arms before 09-25c only ab2_wd trains with weight decay ({wd_arms})")
new_wd = [t for t, p in tp.PPX_POLICY.items() if p >= C and tp.PPX_ARMS[t].hparams.get("weight_decay") != 1e-4]
check(not new_wd, f"every arm at >= 09-25c trains with weight decay 1e-4 ({new_wd})")

print("[2] the default knowledge mix is ab2_know30_concat's")
ref = tp.PPX_ARMS["ab2_know30_concat"].hparams
check(
    set(tp.PPX_KNOWLEDGE_DEFAULT) == set(KNOW_KEYS)
    and all(str(tp.PPX_KNOWLEDGE_DEFAULT[k]) == str(ref[k]) for k in KNOW_KEYS),
    "same codes, share, meta and scheme (barge_prob %s)"
    % tp.PPX_KNOWLEDGE_DEFAULT.get("knowledge_concat", {}).get("barge_prob"),
)

print("[3] a probe arm at 09-25c")
data = ab2h._STATE["train_data"]
HP = ab2h.AB2H_HPARAMS


def arm(tag, extra=None, policy=C, **kw):
    kw.setdefault("hparams", HP)
    return tp._ppx_arm(f"_guard_{tag}", data, extra or {}, steps=100, benchmark=False, policy=policy, **kw)


hp = arm("c").hparams
check(
    hp.get("weight_decay") == 1e-4
    and hp.get("grad_accum") == 1
    and hp.get("user_semantic_weight") == 1.0
    and all(str(hp.get(k)) == str(ref[k]) for k in KNOW_KEYS)
    and hp.get("eval_every") == 100,
    "wd 1e-4, grad_accum 1, usersem 1.0, know30_concat mix, in-loop eval",
)
check(arm("wd0", {"weight_decay": 0.0}).hparams["weight_decay"] == 0.0, "an explicit weight_decay is kept")
check(bool(raises(lambda: arm("ga4", {"grad_accum": 4}))), "grad_accum 4 raises")
check(bool(raises(lambda: arm("ph", hparams=tp.PERSONA_HPARAMS))), "PERSONA_HPARAMS (grad_accum 4) raises")
for k, v in (("turn_tail_weight", 0.3), ("floor_weight", 3.0), ("length_bucket_pool", 64)):
    check(bool(raises(lambda: arm(f"clash_{k}", {k: v}))), f"{k} without opt-out raises")
tp.PPX_KNOWLEDGE_OPT_OUT["_guard_optout"] = "guard"
oo = arm("optout", {"turn_tail_weight": 0.3}).hparams
check("knowledge_data" not in oo and oo["weight_decay"] == 1e-4, "an opted-out arm trains without the mix, with wd")
mf = dict(max_steps=10, register=False, evals=False, train_data=data)
ml = dict(adapter=tc.MOSHI_LIB.adapter, venv_python_path=tc.MOSHI_LIB.venv())
check(
    bool(raises(lambda: runs.make_finetune_run("_g1", policy=C, hparams={"lr": 1e-6}, **ml, **mf))),
    "make_finetune_run: no grad_accum raises",
)
check(
    bool(raises(lambda: runs.make_finetune_run("_g2", policy=C, hparams={"grad_accum": 4}, **ml, **mf))),
    "make_finetune_run: grad_accum 4 raises",
)
check(
    bool(
        raises(
            lambda: runs.make_finetune_run(
                "_g3",
                policy=C,
                hparams={"grad_accum": 1},
                adapter=tc.PERSONAPLEX_LIB.adapter,
                venv_python_path=tc.MOSHI_LIB.venv(),
                **mf,
            )
        )
    ),
    "make_finetune_run: a PersonaPlex adapter raises",
)
check(
    raises(lambda: runs.make_finetune_run("_g4", policy=B, hparams={"grad_accum": 4}, **ml, **mf)) is None,
    "... but not at 09-25b",
)
check(
    bool(raises(lambda: runs.make_finetune_run("_g5", policy=C, hparams={"grad_accum": 1}, **ml, **mf))),
    "make_finetune_run: grad_accum 1 without per_gpu_batch raises (the launcher default batch is 1)",
)
check(
    raises(lambda: runs.make_finetune_run("_g6", policy=C, hparams={"grad_accum": 1, "per_gpu_batch": 16}, **ml, **mf))
    is None,
    "make_finetune_run: grad_accum 1 with per_gpu_batch is accepted",
)

print("[4] CAN FAIL: the gate")
hb = arm("b", policy=B).hparams
check(
    "weight_decay" not in hb and not any(k in hb for k in KNOW_KEYS) and hb.get("grad_accum") == 1,
    "the same arm at 09-25b: no wd, no mix (grad_accum 1 is the standard's own)",
)
saved = dict(tp.PPX_KNOWLEDGE_DEFAULT)
tp.PPX_KNOWLEDGE_DEFAULT.clear()
check(bool(raises(lambda: arm("unreg"))), "knowledge default unregistered -> a 09-25c arm raises")
tp.PPX_KNOWLEDGE_DEFAULT.update(saved)

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
