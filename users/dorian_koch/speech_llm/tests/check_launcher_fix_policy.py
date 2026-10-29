"""Guard: the launcher setup fixes (code-map §9) reach NEW arms of every family, and no existing one.

The 2026-09-23 code-sharing review found setup pieces missing from some launchers (RL / MoshiRAG /
Audex without fp32 master weights, PersonaPlex / MoshiRAG / Audex restarting their data stream on
resume, MoshiRAG's fixed loss denominator, Audex's replay branch at the default ``text_pad_weight``,
base Moshi's ``DDP(static_graph=True)``, which never all-reduced). Each fix is a launcher config key
whose default is the OLD behaviour, rendered only when set: ``finetune.LAUNCHER_FIX_PRESETS``, applied
from ``runs.RULE_LAUNCHER_FIXES = POLICY_2026_09_23`` -- p3 (running) and every finished arm keep
their config, hash and behaviour. Mirrors ``check_fast_train_policy.py``:

  [1] ``launcher_fix_hparams`` is ``{}`` before the epoch and the family's preset from it; unknown family raises
  [2] ``train()`` (base Moshi LoRA, full finetune, MoshiRAG): a new arm carries and RENDERS its preset
      and hashes apart; an arm one epoch older is exactly what it is without the rule (hparams, hash,
      rendered config)
  [3] ``_ppx_arm``: the default (new) epoch gets the PersonaPlex preset, ``policy=POLICY_2026_09_22`` does not
  [4] ``AudexDuplexFinetune`` / ``RLFinetune(launcher_fixes=...)``: nothing without, the preset with, part
      of the hash only when set, and a key the launcher does not read is refused
  [5] every preset key is read by its family's launcher, and ABSENT means the old behaviour: the
      ``cfg.get`` default equals the pre-fix value, and the preset value differs from it
  [6] the checks can fail: a rule firing for every epoch, and a renderer emitting unset keys, are caught

The hash side of "no existing arm moves" is ``hash_snapshot.py`` over the whole graph; this guard proves
the mechanism. Run from the setup root, ~15 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_launcher_fix_policy.py
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
    Loss,
)

FAILURES: list[str] = []
LIB = SETUP / "recipe" / "speech_llm" / "full_duplex" / "moshi_family"
FIX = set(finetune.LAUNCHER_FIX_KEYS)
#: What each launcher did before its fix -- the value an ABSENT key must reproduce.
OLD_BEHAVIOUR = {
    "fp32_master_weights": False,
    "resume_data_position": False,
    "loss_norm": "count",
    "text_pad_weight_in_full_stage1": False,
    "ddp_static_graph": True,
}
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


def check(cond, msg, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}{'  ' + extra if extra else ''}")
    if not cond:
        FAILURES.append(msg)


def launcher_defaults(rel: str) -> dict:
    """{key: literal default} of every ``cfg.get("key", <literal>)`` in the launcher's ``main()``."""
    main = next(
        n for n in ast.walk(ast.parse((LIB / rel).read_text())) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    out = {}
    for n in ast.walk(main):
        if (
            isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "get"
            and getattr(n.func.value, "id", "") == "cfg"
            and n.args
            and isinstance(n.args[0], ast.Constant)
        ):
            try:
                out[n.args[0].value] = ast.literal_eval(n.args[1]) if len(n.args) > 1 else None
            except ValueError:  # a non-literal default (a local name): not one of ours
                out[n.args[0].value] = "<expr>"
    return out


def mutated(module, name, old, new):
    """Function ``name`` of ``module`` re-compiled from its real source with one line changed."""
    src = textwrap.dedent(inspect.getsource(getattr(module, name)))
    assert old in src, f"mutation anchor not found in {name}: {old!r}"
    g = dict(vars(module))
    exec(compile(src.replace(old, new), f"<mutated {name}>", "exec"), g)
    return g[name]


CORPUS = Corpus(mix=tk.Path("/dev/null/corpus", hash_overwrite="fix_policy_guard_corpus"), duration_sec=60)
FAMILIES = {"moshi_lib": MOSHI_LIB, "moshi_lib_full": MOSHI_LIB_FULL, "moshirag_lib": MOSHIRAG_LIB}


def build(model, policy):
    sisyphus.job.created_jobs.clear()
    return runs.train(
        "fix_policy_guard", model=model, corpus=CORPUS, policy=policy, steps=100, evals=False, register=False
    )


def rendered(job) -> dict:
    return yaml.safe_load(job.adapter.render_config(job, job.adapter.batch_size, 100))


def run_checks(fix_hparams):
    saved = runs.launcher_fix_hparams
    runs.launcher_fix_hparams = fix_hparams
    n0 = len(FAILURES)
    try:
        print("[1] epochs")
        check(runs.RULE_LAUNCHER_FIXES == runs.POLICY_2026_09_23, "RULE_LAUNCHER_FIXES = POLICY_2026_09_23")
        for fam, preset in finetune.LAUNCHER_FIX_PRESETS.items():
            check(
                fix_hparams(runs.POLICY_2026_09_22, fam) == {} and fix_hparams(runs.POLICY_2026_09_23, fam) == preset,
                f"{fam}: {{}} at 2026-09-22, its preset at 2026-09-23",
                str(preset),
            )
        try:
            fix_hparams(runs.POLICY_2026_09_23, "no_such_family")
            check(False, "an unknown family raises")
        except KeyError:
            check(True, "an unknown family raises")

        print("[2] train(): base Moshi LoRA / full finetune / MoshiRAG")
        for fam, model in FAMILIES.items():
            preset = fix_hparams(runs.POLICY_2026_09_23, fam)
            new, old = build(model, runs.POLICY_2026_09_23), build(model, runs.POLICY_2026_09_22)
            check({k: new.job.hparams.get(k) for k in preset} == preset, f"{fam}: a new arm carries its preset")
            cfg_new = rendered(new.job)
            check({k: cfg_new.get(k) for k in preset} == preset, f"{fam}: ...and renders it", str(preset))
            # An older arm must be exactly what it was without the rule: same hparams, same hash, same
            # config. (Not "carries no fix key": a MoshiRAG arm at 2026-09-14+ already declares loss_norm
            # through its architecture default -- the renderer used to DROP it, a knob that did nothing.)
            real_fn = runs.launcher_fix_hparams
            runs.launcher_fix_hparams = lambda policy, family: {}
            try:
                without = build(model, runs.POLICY_2026_09_22)
            finally:
                runs.launcher_fix_hparams = real_fn
            old = build(model, runs.POLICY_2026_09_22)
            check(
                old.job.hparams == without.job.hparams and old.job._sis_id() == without.job._sis_id(),
                f"{fam}: an arm at 2026-09-22 is unchanged by the rule (hparams + hash)",
            )
            check(rendered(old.job) == rendered(without.job), f"{fam}: ...and renders the same config")
            if preset:
                check(new.job._sis_id() != old.job._sis_id(), f"{fam}: the new arm hashes apart")
        # an explicit typed knob wins over the preset (the preset fills only what the arm left unset)
        sisyphus.job.created_jobs.clear()
        stated = runs.train(
            "fix_policy_guard_norm", model=MOSHIRAG_LIB, corpus=CORPUS, policy=runs.POLICY_2026_09_23, steps=100,
            evals=False, register=False, loss=Loss(audio_other=0.02, text_pad=0.3, norm="count"),
        )  # fmt: skip
        check(stated.job.hparams.get("loss_norm") == "count", "moshirag_lib: an explicit Loss(norm='count') survives")

        print("[3] _ppx_arm (PersonaPlex LoRA)")
        data = tk.Path("/dev/null/ppx", hash_overwrite="fix_policy_guard_ppx")
        preset = fix_hparams(runs.POLICY_2026_09_23, "personaplex_lib_lora")
        sisyphus.job.created_jobs.clear()
        new = track_persona._ppx_arm("fix_guard_new", data, hparams=track_persona.P2_HPARAMS, benchmark=False)
        sisyphus.job.created_jobs.clear()
        old = track_persona._ppx_arm(
            "fix_guard_old", data, hparams=track_persona.P2_HPARAMS, benchmark=False, policy=runs.POLICY_2026_09_22
        )
        check({k: new.hparams.get(k) for k in preset} == preset, "new PersonaPlex arm carries the preset", str(preset))
        check(not set(preset) & set(old.hparams), "policy=2026-09-22 (p0..p3): none of it")
        cfg_new, cfg_old = rendered(new), rendered(old)
        check({k: cfg_new.get(k) for k in preset} == preset, "...rendered")
        check(not set(preset) & set(cfg_old), "...and not rendered for p0..p3")

        print("[4] Audex / RL jobs (launcher_fixes=)")
        venv = tk.Path("/dev/null/python", hash_overwrite="fix_policy_guard_venv")
        jobs = {}
        for stage in ("stage0", "stage1"):
            kw = dict(venv_python_path=venv, stage=stage, synthetic=True, max_steps=10)
            if stage == "stage1":
                kw["init_checkpoint"] = tk.Path("/dev/null/s0", hash_overwrite="fix_policy_guard_s0")
            jobs[f"audex_{stage}"] = (audex_finetune.AudexDuplexFinetune, audex_finetune._render_audex_config, kw)
        jobs["rl_grpo"] = (
            rl_finetune.RLFinetune,
            rl_finetune._render_rl_config,
            dict(venv_python_path=venv, max_steps=10),
        )
        for fam, (cls, render, kw) in jobs.items():
            preset = fix_hparams(runs.POLICY_2026_09_23, fam)
            sisyphus.job.created_jobs.clear()
            plain = cls(**kw)
            sisyphus.job.created_jobs.clear()
            fixed = cls(**kw, launcher_fixes=preset)
            c_plain, c_fixed = yaml.safe_load(render(plain)), yaml.safe_load(render(fixed))
            check(not FIX & set(c_plain), f"{fam}: no launcher_fixes -> no key rendered")
            check({k: c_fixed.get(k) for k in preset} == preset and preset, f"{fam}: the preset rendered", str(preset))
            check(plain._sis_id() != fixed._sis_id(), f"{fam}: launcher_fixes is part of the hash when set")
        sisyphus.job.created_jobs.clear()
        bad = rl_finetune.RLFinetune(venv_python_path=venv, max_steps=10, launcher_fixes={"resume_data_position": True})
        try:
            rl_finetune._render_rl_config(bad)
            check(False, "RL: a key its launcher does not read is refused at render time")
        except ValueError:
            check(True, "RL: a key its launcher does not read is refused at render time")

        print("[5] every preset key is read by its launcher, and absent = the old behaviour")
        for fam, preset in finetune.LAUNCHER_FIX_PRESETS.items():
            defaults = launcher_defaults(LAUNCHER_OF[fam])
            for k, v in preset.items():
                check(
                    k in defaults and defaults[k] == OLD_BEHAVIOUR[k] and v != OLD_BEHAVIOUR[k],
                    f"{fam}: {LAUNCHER_OF[fam]} reads {k!r}, default {defaults.get(k)!r} = old, preset {v!r}",
                )
    finally:
        runs.launcher_fix_hparams = saved
    return len(FAILURES) - n0


assert run_checks(runs.launcher_fix_hparams) == 0, FAILURES
print("[6] the checks can fail (mutated real sources)")
real = list(FAILURES)
saved_render = finetune.render_launcher_fix_keys
for label, fn, render in (
    (
        "a rule that fires for every epoch",
        mutated(runs, "launcher_fix_hparams", "if not _policy_applies(policy, RULE_LAUNCHER_FIXES):", "if False:"),
        saved_render,
    ),
    (
        "a renderer that emits keys the arm did not set",
        runs.launcher_fix_hparams,
        mutated(
            finetune,
            "render_launcher_fix_keys",
            "if k in supported and hp.get(k) is not None:",
            "if k in supported:\n            hp.setdefault(k, OLD_BEHAVIOUR_FOR_MUTANT[k])",
        ),
    ),
):
    print(f"  -- mutant: {label}")
    # the renderer is looked up by name inside each renderer module, so patch it where it is used
    finetune.OLD_BEHAVIOUR_FOR_MUTANT = OLD_BEHAVIOUR
    render.__globals__["OLD_BEHAVIOUR_FOR_MUTANT"] = OLD_BEHAVIOUR
    patched = [m for m in (finetune, audex_finetune, rl_finetune) if hasattr(m, "render_launcher_fix_keys")]
    for m in patched:
        m.render_launcher_fix_keys = render
    try:
        n = run_checks(fn)
    except Exception as e:  # a mutant may crash a check outright; that is a rejection too
        n = 1
        print(f"     (raised {type(e).__name__}: {e})")
    finally:
        for m in patched:
            m.render_launcher_fix_keys = saved_render
    del FAILURES[len(real) :]
    check(n > 0, f"rejected: {label} ({n} checks failed)")

print()
if FAILURES:
    print(f"FAILED {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("OK")
