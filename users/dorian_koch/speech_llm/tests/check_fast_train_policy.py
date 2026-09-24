"""Guard: the training fast path is the default for NEW arms of every family, and for no existing one.

User, 2026-09-23: "just enable it for all new training runs". The switches (``moshi_family.train_perf``,
finetuning.md "Training performance") are hashed hparams, so turning them on for an existing arm would
re-hash it -- a running one (p3) would restart from zero on different kernels. They are therefore a
dated rule, ``runs.RULE_FAST_TRAIN = POLICY_2026_09_23`` (= ``POLICY_LATEST``):

  [1] the epochs: POLICY_LATEST is the new epoch; ``fast_train_hparams`` is ``{}`` before it and the
      family's preset (``finetune.FAST_TRAIN_PRESETS``) from it, for every family; unknown family raises
  [2] ``train()`` (base Moshi LoRA, full finetune, MoshiRAG): an arm at the new epoch carries and RENDERS
      exactly its preset, an arm one epoch older carries and renders none of the keys, the two hash
      apart, and an arm that states a switch keeps its own value (ablation still possible)
  [3] ``track_persona._ppx_arm``: its default policy (a literal epoch, not POLICY_LATEST) gives the
      PersonaPlex preset AND per-GPU batch 8; ``policy=POLICY_2026_09_22`` (p0..p3) gives neither;
      ``extra_hparams`` can still name the batch
  [4] ``AudexDuplexFinetune`` / ``RLFinetune(train_perf=...)``: nothing rendered without it, the preset
      rendered with it, and a key the launcher cannot act on is refused (fused LoRA add at stage0)
  [5] every preset key is read by that family's launcher (AST of the real launcher)
  [6] the checks can fail: a rule that fires for every epoch, and a renderer that emits unset keys,
      are both caught (mutated real sources)

The hash side of "no existing arm moves" is ``hash_snapshot.py`` over the whole graph (2,664 jobs, no
diff); this guard proves the mechanism that makes it so. Run from the setup root, ~15 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_fast_train_policy.py
"""

import ast
import inspect
import os
import sys
import textwrap
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import sisyphus.job  # noqa: E402
import yaml  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import audex_finetune, finetune, rl_finetune  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank import runs, track_persona  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank.train_config import (  # noqa: E402
    MOSHI_LIB,
    MOSHI_LIB_FULL,
    MOSHIRAG_LIB,
    Corpus,
)

FAILURES: list[str] = []
LIB = SETUP / "recipe" / "speech_llm" / "full_duplex" / "moshi_family"
PERF = set(finetune.TRAIN_PERF_KEYS)


def check(cond, msg, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}{'  ' + extra if extra else ''}")
    if not cond:
        FAILURES.append(msg)


def launcher_keys(rel: str) -> set:
    main = next(
        n for n in ast.walk(ast.parse((LIB / rel).read_text())) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    return {
        n.args[0].value
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", "") == "get"
        and getattr(n.func.value, "id", "") == "cfg"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    }


def mutated(module, name, old, new):
    """Function ``name`` of ``module`` re-compiled from its real source with one line changed."""
    src = textwrap.dedent(inspect.getsource(getattr(module, name)))
    assert old in src, f"mutation anchor not found in {name}: {old!r}"
    g = dict(vars(module))
    exec(compile(src.replace(old, new), f"<mutated {name}>", "exec"), g)
    return g[name]


CORPUS = Corpus(mix=tk.Path("/dev/null/corpus", hash_overwrite="fast_policy_guard_corpus"), duration_sec=60)
FAMILIES = {"moshi_lib": MOSHI_LIB, "moshi_lib_full": MOSHI_LIB_FULL, "moshirag_lib": MOSHIRAG_LIB}
LAUNCHER_OF = {
    "moshi_lib": "moshi_finetune_launcher.py",
    "moshi_lib_full": "moshi_finetune_launcher.py",
    "moshirag_lib": "moshirag_finetune_launcher.py",
    "personaplex_lib_lora": "personaplex/finetune_launcher.py",
    "personaplex_lib_full": "personaplex/finetune_launcher.py",
    "audex_stage0": "audex/finetune_launcher.py",
    "audex_stage1": "audex/finetune_launcher.py",
    "rl_grpo": "rl/launcher.py",
}


def build(model, policy, extra_hparams=None):
    sisyphus.job.created_jobs.clear()  # a fresh singleton cache: two builds must not share one object
    return runs.train(
        "fast_policy_guard",
        model=model,
        corpus=CORPUS,
        policy=policy,
        steps=100,
        extra_hparams=extra_hparams,
        evals=False,
        register=False,
    )


def rendered(job) -> dict:
    return yaml.safe_load(job.adapter.render_config(job, job.adapter.batch_size, 100))


def run_checks(fast_train_hparams, render_perf):
    """Everything below, against the given helper implementations (the real ones, or a mutant)."""
    saved = (runs.fast_train_hparams, finetune.render_train_perf_keys)
    runs.fast_train_hparams, finetune.render_train_perf_keys = fast_train_hparams, render_perf
    n0 = len(FAILURES)
    try:
        print("[1] epochs")
        check(
            runs.RULE_FAST_TRAIN == runs.POLICY_2026_09_23 and runs.POLICY_LATEST >= runs.RULE_FAST_TRAIN,
            "RULE_FAST_TRAIN = 2026-09-23, and POLICY_LATEST includes it",
        )
        for fam, preset in finetune.FAST_TRAIN_PRESETS.items():
            check(
                fast_train_hparams(runs.POLICY_2026_09_22, fam) == {}
                and fast_train_hparams(runs.POLICY_2026_09_23, fam) == preset,
                f"{fam}: {{}} at 2026-09-22, its preset at 2026-09-23",
            )
        try:
            fast_train_hparams(runs.POLICY_2026_09_23, "no_such_family")
            check(False, "an unknown family raises")
        except KeyError:
            check(True, "an unknown family raises")

        print("[2] train(): base Moshi LoRA / full finetune / MoshiRAG")
        for fam, model in FAMILIES.items():
            preset = finetune.FAST_TRAIN_PRESETS[fam]
            new, old = build(model, runs.POLICY_2026_09_23), build(model, runs.POLICY_2026_09_22)
            hp_new, hp_old = new.job.hparams, old.job.hparams
            check({k: hp_new.get(k) for k in preset} == preset, f"{fam}: a new arm carries its preset")
            check(not PERF & set(hp_old), f"{fam}: an arm at 2026-09-22 carries none", str(sorted(PERF & set(hp_old))))
            cfg_new, cfg_old = rendered(new.job), rendered(old.job)
            extra_new = {k: cfg_new[k] for k in PERF & set(cfg_new)}
            # base Moshi renders prefetch_batches for every arm (default 2, since before the switches)
            base_old = {k: cfg_old[k] for k in PERF & set(cfg_old)}
            check(
                {k: v for k, v in extra_new.items() if k not in base_old} == preset,
                f"{fam}: ...and renders exactly it",
                str(extra_new),
            )
            check(
                base_old in ({}, {"prefetch_batches": 2}) and base_old == {k: extra_new[k] for k in base_old},
                f"{fam}: the old arm renders no switch",
                str(base_old),
            )
            check(new.job._sis_id() != old.job._sis_id(), f"{fam}: the two hash apart (a new arm, never an old one)")
            abl = build(model, runs.POLICY_2026_09_23, extra_hparams={"contiguous_step_inputs": False})
            check(abl.job.hparams["contiguous_step_inputs"] is False, f"{fam}: an arm can still ablate one switch")

        print("[3] _ppx_arm (PersonaPlex LoRA)")
        sig = inspect.signature(track_persona._ppx_arm).parameters
        check(sig["policy"].default == runs.POLICY_2026_09_23, "_ppx_arm defaults to the LITERAL new epoch")
        data = tk.Path("/dev/null/ppx", hash_overwrite="fast_policy_guard_ppx")
        sisyphus.job.created_jobs.clear()
        job_new = track_persona._ppx_arm("guard_new", data, hparams=track_persona.P2_HPARAMS, benchmark=False)
        hp_new = job_new.hparams
        sisyphus.job.created_jobs.clear()
        hp_old = track_persona._ppx_arm(
            "guard_old", data, hparams=track_persona.P2_HPARAMS, benchmark=False, policy=runs.POLICY_2026_09_22
        ).hparams
        sisyphus.job.created_jobs.clear()
        hp_b4 = track_persona._ppx_arm(
            "guard_b4", data, {"per_gpu_batch": 4}, hparams=track_persona.P2_HPARAMS, benchmark=False
        ).hparams
        fast = finetune.PERSONAPLEX_FAST_HPARAMS
        check(
            {k: hp_new.get(k) for k in fast} == fast
            and hp_new["per_gpu_batch"] == runs.PERSONAPLEX_FAST_PER_GPU_BATCH == 8,
            "new PersonaPlex arm: the fast preset and per-GPU batch 8",
        )
        check(
            not PERF & set(hp_old) and hp_old["per_gpu_batch"] == track_persona.P2_HPARAMS["per_gpu_batch"] == 4,
            "policy=2026-09-22 (p0..p3): no switch, the arm's own batch",
        )
        check(
            hp_b4["per_gpu_batch"] == 4 and {k: hp_b4.get(k) for k in fast} == fast, "extra_hparams can name the batch"
        )
        cfg = rendered(job_new)
        check({k: cfg.get(k) for k in fast} == fast and cfg["per_gpu_batch"] == 8, "...rendered")

        print("[4] Audex / RL jobs (train_perf=)")
        venv = tk.Path("/dev/null/python", hash_overwrite="fast_policy_guard_venv")
        for stage in ("stage0", "stage1"):
            kw = dict(venv_python_path=venv, stage=stage, synthetic=True, max_steps=10)
            if stage == "stage1":
                kw["init_checkpoint"] = tk.Path("/dev/null/s0", hash_overwrite="fast_policy_guard_s0")
            sisyphus.job.created_jobs.clear()
            plain = audex_finetune.AudexDuplexFinetune(**kw)
            sisyphus.job.created_jobs.clear()
            tp = fast_train_hparams(runs.POLICY_2026_09_23, f"audex_{stage}")
            fastj = audex_finetune.AudexDuplexFinetune(**kw, train_perf=tp)
            c_plain = yaml.safe_load(audex_finetune._render_audex_config(plain))
            c_fast = yaml.safe_load(audex_finetune._render_audex_config(fastj))
            check(not PERF & set(c_plain), f"audex {stage}: no train_perf -> no key rendered")
            check({k: c_fast.get(k) for k in tp} == tp and tp, f"audex {stage}: train_perf rendered", str(tp))
            check(plain._sis_id() != fastj._sis_id(), f"audex {stage}: train_perf is part of the hash when set")
        sisyphus.job.created_jobs.clear()
        bad = audex_finetune.AudexDuplexFinetune(
            venv_python_path=venv, synthetic=True, train_perf={"fused_lora_add": True}
        )
        try:
            audex_finetune._render_audex_config(bad)
            check(False, "audex stage0: fused_lora_add refused at render time")
        except ValueError:
            check(True, "audex stage0: fused_lora_add refused at render time")
        sisyphus.job.created_jobs.clear()
        rl_plain = rl_finetune.RLFinetune(venv_python_path=venv, max_steps=10)
        sisyphus.job.created_jobs.clear()
        tp = fast_train_hparams(runs.POLICY_2026_09_23, "rl_grpo")
        rl_fast = rl_finetune.RLFinetune(venv_python_path=venv, max_steps=10, train_perf=tp)
        c_plain = yaml.safe_load(rl_finetune._render_rl_config(rl_plain))
        c_fast = yaml.safe_load(rl_finetune._render_rl_config(rl_fast))
        check(not PERF & set(c_plain) and {k: c_fast.get(k) for k in tp} == tp, "RL: nothing without, the preset with")

        print("[5] every preset key is read by its family's launcher")
        for fam, preset in finetune.FAST_TRAIN_PRESETS.items():
            keys = launcher_keys(LAUNCHER_OF[fam])
            check(set(preset) <= keys, f"{fam}: {LAUNCHER_OF[fam]} reads {sorted(preset)}", str(set(preset) - keys))
    finally:
        runs.fast_train_hparams, finetune.render_train_perf_keys = saved
    return len(FAILURES) - n0


assert run_checks(runs.fast_train_hparams, finetune.render_train_perf_keys) == 0, FAILURES
print("[6] the checks can fail (mutated real sources)")
real = list(FAILURES)
for label, fth, rpk in (
    (
        "a rule that fires for every epoch",
        mutated(runs, "fast_train_hparams", "if not _policy_applies(policy, RULE_FAST_TRAIN):", "if False:"),
        finetune.render_train_perf_keys,
    ),
    (
        "a renderer that emits keys the arm did not set",
        runs.fast_train_hparams,
        mutated(
            finetune, "render_train_perf_keys", "if k in supported and hp.get(k) is not None:", "if k in supported:"
        ),
    ),
):
    print(f"  -- mutant: {label}")
    try:
        n = run_checks(fth, rpk)
    except Exception as e:  # a mutant may also crash a check outright; that is a rejection too
        n = 1
        print(f"     (raised {type(e).__name__}: {e})")
    del FAILURES[len(real) :]
    check(n > 0, f"rejected: {label} ({n} checks failed)")

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("OK")
