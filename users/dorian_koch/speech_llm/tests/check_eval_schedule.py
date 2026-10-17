"""Guard: ``run_training`` owns EVERY eval call, step 0 included, and no launcher calls its own ``eval_fn``.

2026-10-17, merging upstream ``9a87fc8b`` ("train_loop: eval at step 0 before training"): the PersonaPlex
launcher already called ``eval_fn(0, fatal=True)`` itself before ``run_training``, so taking both would have
evaluated step 0 twice (a duplicate step-0 row in ``metrics.eval.jsonl``). Consolidated: the loop calls
``eval_fn(0)`` when ``start_step == 0`` (never on resume), then every ``eval_every`` steps; launchers only
define ``eval_fn`` and hand it over (PersonaPlex's raises at step 0, so a broken eval still stops the run).

Drives the real ``run_training`` over a toy model and reads the launchers' ASTs.
Run: ``CUDA_HOME=/usr .venv/bin/python projects/i6_experiments/users/dorian_koch/speech_llm/tests/check_eval_schedule.py``
"""

import ast
import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402

from speech_llm.full_duplex.moshi_family.train_loop import TrainConfig, run_training  # noqa: E402

FAILS = []


def check(ok, what):
    print(("[ok] " if ok else "[FAIL] ") + what)
    if not ok:
        FAILS.append(what)


def eval_steps(*, max_steps, eval_every, start_step=0, with_eval=True):
    """The steps ``run_training`` calls ``eval_fn`` at, on a toy model."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 4)
    params = list(lin.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-3}])
    calls = []

    def batches():
        while True:
            yield torch.randn(2, 4)

    run_training(
        optimizer=opt,
        base_lrs=[1e-3],
        group_names=("temporal",),
        trainable_params=params,
        batch_iter=batches(),
        loss_step=lambda b: lin(b).pow(2).mean(),
        save_fn=lambda step, final: None,
        cfg=TrainConfig(max_steps=max_steps, warmup_steps=1, save_every=0, log_every=1, eval_every=eval_every),
        out_dir=tempfile.mkdtemp(),
        log=lambda m: None,
        eval_fn=calls.append if with_eval else None,
        start_step=start_step,
    )
    return calls


print("[1] the loop's eval schedule")
got = eval_steps(max_steps=4, eval_every=2)
check(got == [0, 2, 4], f"fresh run: eval at step 0 exactly once, then every eval_every (got {got})")
got = eval_steps(max_steps=4, eval_every=2, start_step=2)
check(0 not in got and got == [4], f"resumed run (start_step 2): no step-0 eval (got {got})")
got = eval_steps(max_steps=4, eval_every=0)
check(got == [], f"eval_every 0: no eval at all, step 0 included (got {got})")

print("[2] launchers never call eval_fn themselves")
for rel in ("moshi_family/personaplex/finetune_launcher.py", "moshi_family/moshi_finetune_launcher.py"):
    tree = ast.parse((SETUP / "recipe/speech_llm/full_duplex" / rel).read_text())
    direct = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "eval_fn"
    ]
    passed = any(
        kw.arg == "eval_fn"
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_training"
        for kw in n.keywords
    )
    check(not direct and passed, f"{rel}: eval_fn only handed to run_training (direct calls at lines {direct})")

ppx = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py").read_text()
check(
    "if step == 0:\n                raise" in ppx,
    "PersonaPlex eval_fn raises at step 0 (a broken eval stops the run before it trains blind)",
)

if FAILS:
    raise SystemExit(f"{len(FAILS)} check(s) failed")
print("all ok")
