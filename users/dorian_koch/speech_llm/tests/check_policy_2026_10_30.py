"""Guard: the POLICY_2026_10_30 / RL2_POLICY_2026_10_30 default (``stable_softmax``) reaches new arms and no existing arm.

``stable_softmax`` (exact CE / log-softmax / KL gradients, ``moshi_family.stable_softmax``; guard
``check_stable_softmax.py``) is a launcher config key, hashed, so it is dated:

  [1] epochs: 10-29 < 10-30 = POLICY_LATEST, RULE_STABLE_SOFTMAX dated 10-30; RL2_POLICY_LATEST >= rl2-2026-10-30.
  [2] at 10-30 every launcher-fix family gets the key (``launcher_fix_hparams``), and it reaches the rendered config of
      a PersonaPlex LoRA arm, a 2-GPU PersonaPlex full arm and a base-Moshi LoRA and full arm (``runs.train``); an
      explicit False is kept; an RL V2 PersonaPlex model at rl2-2026-10-30 lowers to ``launcher_fixes`` with it.
  [3] CAN FAIL: the same arms at 10-29 / rl2-2026-10-29 get no key and render none.

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_10_30.py
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
    (m,) = {id(v): v for k, v in sys.modules.items() if k.endswith(suffix) and v is not None}.values()
    return m


tp, runs, ab2h, tc = mod(".track_persona"), mod(".doriank.runs"), mod(".track_persona_ab2h"), mod(".train_config")
rl_policy, rl_presets, rl_lower = mod("rl_spec.policy"), mod("rl_spec.presets"), mod("rl_spec.lower")
from i6_experiments.users.dorian_koch.speech_llm.finetune import LAUNCHER_FIX_PRESETS  # noqa: E402

FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


NEW, OLD = runs.POLICY_2026_10_30, runs.POLICY_2026_10_29
RN, RO = rl_policy.RL2_POLICY_2026_10_30, rl_policy.RL2_POLICY_2026_10_29
print("[1] epochs")
check(OLD < NEW == runs.POLICY_LATEST and NEW in runs.KNOWN_POLICIES, "10-29 < 10-30 = POLICY_LATEST, known")
check(runs.RULE_STABLE_SOFTMAX == NEW, "RULE_STABLE_SOFTMAX dated 10-30")
check(
    RO < RN <= rl_policy.RL2_POLICY_LATEST and RN in rl_policy.KNOWN_POLICIES,
    "rl2-10-29 < rl2-10-30 <= RL2_POLICY_LATEST",
)

data = ab2h._STATE["train_data"]


def ppx_job(tag, policy, gpus=1, adapter=None, extra=None):
    return tp._ppx_arm(
        f"_guard1030_{tag}",
        data,
        extra or {},
        steps=100,
        benchmark=False,
        policy=policy,
        hparams=ab2h.AB2H_HPARAMS,
        adapter=adapter or tp.PERSONAPLEX_LIB_LORA_ADAPTER,
        compute={"gpu": gpus, "rqmt_time_h": 1},
    )


def moshi_job(tag, policy, model=None):
    return runs.train(
        f"_guard1030_{tag}",
        policy=policy,
        model=model or tc.MOSHI_LIB,
        corpus=tc.Corpus(data, duration_sec=40),
        optim=tc.Optim(lr=1e-6),
        compute=tc.Compute(hours=1),
        steps=100,
        extra_hparams={"grad_accum": 1, "per_gpu_batch": 16},  # RULE_NO_GRAD_ACCUM
        register=False,
        evals=False,
    ).job


def rendered(job):
    """The config.yaml text the job's own renderer writes (as SpeechFinetune.run does)."""
    return job.adapter.render_config(job, job.adapter.batch_size, 100)


def arms(policy):
    p = policy.replace("-", "")  # a tag is declared once per graph
    return {
        "PersonaPlex LoRA": ppx_job(f"lora_{p}", policy),
        "PersonaPlex full 2-GPU": ppx_job(f"full2_{p}", policy, gpus=2, adapter=tp.PERSONAPLEX_LIB_FULL_ADAPTER),
        "base Moshi LoRA": moshi_job(f"m_{p}", policy),
        "base Moshi full": moshi_job(f"mf_{p}", policy, model=tc.MOSHI_LIB_FULL),
    }


print("[2] new arms at 10-30")
fams = sorted(f for f in LAUNCHER_FIX_PRESETS if runs.launcher_fix_hparams(NEW, f).get("stable_softmax") is not True)
check(
    not fams,
    f"launcher_fix_hparams: every family ({len(LAUNCHER_FIX_PRESETS)}) gets it" + (f"; missing {fams}" if fams else ""),
)
for name, job in arms(NEW).items():
    check(
        job.hparams.get("stable_softmax") is True and "stable_softmax: true" in rendered(job), f"{name}: key + rendered"
    )
off = ppx_job("lora_off", NEW, extra={"stable_softmax": False})
check(
    off.hparams.get("stable_softmax") is False and "stable_softmax: false" in rendered(off), "an explicit False is kept"
)
check(
    rl_lower._model(rl_presets.personaplex(RN))["launcher_fixes"].get("stable_softmax") is True,
    "RL V2 model at rl2-10-30",
)

print("[3] CAN FAIL: the same arms at 10-29")
check(
    not any("stable_softmax" in runs.launcher_fix_hparams(OLD, f) for f in LAUNCHER_FIX_PRESETS),
    "launcher_fix_hparams: none",
)
for name, job in arms(OLD).items():
    check(
        "stable_softmax" not in job.hparams and "stable_softmax" not in rendered(job), f"{name}: no key, none rendered"
    )
check("stable_softmax" not in rl_lower._model(rl_presets.personaplex(RO))["launcher_fixes"], "RL V2 model at rl2-10-29")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
