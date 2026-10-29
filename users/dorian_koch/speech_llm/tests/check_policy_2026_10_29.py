"""Guard: the POLICY_2026_10_29 defaults reach new arms and no existing arm.

Two run()-side launcher fixes, hashed because they are config keys, so they are gated to new arms:

  [1] epochs: 10-16 < 10-29 = POLICY_LATEST, both rules dated 10-29, no PersonaPlex arm declared at 10-29 yet.
  [2] a single-GPU PersonaPlex FULL finetune at 10-29 (``_ppx_arm``, in memory) gets ``fp32_master_weights`` and the
      renderer writes it; a 2-GPU full arm (FSDP2 keeps fp32 masters) and a LoRA arm do not get the key; an explicit
      False is kept.
  [3] a base-Moshi arm at 10-29 (``runs.train``) gets ``eval_fixed_batches`` and the renderer writes it.
  [4] CAN FAIL: the same two arms at 10-16 get neither key.

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_10_29.py
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


tp, runs, ab2h, tc = mod(".track_persona"), mod(".doriank.runs"), mod(".track_persona_ab2h"), mod(".train_config")
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


N, O = runs.POLICY_2026_10_29, runs.POLICY_2026_10_16
print("[1] epochs")
check(O < N == runs.POLICY_LATEST and N in runs.KNOWN_POLICIES, "10-16 < 10-29 = POLICY_LATEST, known")
check(runs.RULE_EVAL_FIXED_BATCHES == runs.RULE_PPX_FULL_FP32_MASTERS == N, "both rules dated 10-29")
at_n = sorted(t for t, p in tp.PPX_POLICY.items() if p >= N)
check(not at_n, f"no declared PersonaPlex arm at 10-29 yet ({at_n})")

data = ab2h._STATE["train_data"]


def ppx_job(tag, policy, gpus=1, adapter=None, extra=None):
    return tp._ppx_arm(
        f"_guard1029_{tag}",
        data,
        extra or {},
        steps=100,
        benchmark=False,
        policy=policy,
        hparams=ab2h.AB2H_HPARAMS,
        adapter=adapter or tp.PERSONAPLEX_LIB_FULL_ADAPTER,
        compute={"gpu": gpus, "rqmt_time_h": 1},
    )


def ppx(*a, **kw):
    return ppx_job(*a, **kw).hparams


def rendered(job):
    """The config.yaml text the job's own renderer writes (as SpeechFinetune.run does)."""
    return job.adapter.render_config(job, job.adapter.batch_size, 100)


print("[2] PersonaPlex full finetune at 10-29")
j = ppx_job("full1", N)
check(j.hparams.get("fp32_master_weights") is True, "single-GPU full arm: fp32_master_weights True")
check("fp32_master_weights: true" in rendered(j), "... and its config.yaml says so")
check("fp32_master_weights" not in ppx("full2", N, gpus=2), "2-GPU full arm (FSDP2 masters): no key")
check(
    "fp32_master_weights" not in ppx("lora1", N, adapter=tp.PERSONAPLEX_LIB_LORA_ADAPTER),
    "LoRA arm: no key (the launcher keeps masters for LoRA by default)",
)
check(ppx("off", N, extra={"fp32_master_weights": False}).get("fp32_master_weights") is False, "explicit False is kept")

print("[3] base-Moshi arm at 10-29")


def moshi_job(tag, policy):
    return runs.train(
        f"_guard1029_{tag}",
        policy=policy,
        model=tc.MOSHI_LIB,
        corpus=tc.Corpus(data, duration_sec=40),
        optim=tc.Optim(lr=1e-6),
        compute=tc.Compute(hours=1),
        steps=100,
        extra_hparams={"grad_accum": 1, "per_gpu_batch": 16},  # RULE_NO_GRAD_ACCUM
        register=False,
        evals=False,
    ).job


def moshi(tag, policy):
    return moshi_job(tag, policy).hparams


jm = moshi_job("m", N)
check(jm.hparams.get("eval_fixed_batches") is True, "eval_fixed_batches True")
check("eval_fixed_batches: true" in rendered(jm), "... and its config.yaml says so")

print("[4] CAN FAIL: the same arms at 10-16")
check("fp32_master_weights" not in ppx("full_old", O), "PersonaPlex full at 10-16: no key")
check("eval_fixed_batches" not in moshi("m_old", O), "base Moshi at 10-16: no key")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
