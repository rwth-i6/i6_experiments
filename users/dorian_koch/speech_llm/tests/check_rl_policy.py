"""Guard: RL arms are dated like SpeechFinetune arms (2026-10-16), with the same policy epochs as
regular speech finetunes. Every ``RLFinetune`` is built by ``runs.make_rl_run(policy=...)``; from
``POLICY_2026_10_16`` (= POLICY_LATEST) an arm gets ``rollout_text_sampler="t1"`` (T 1, no top-k / top-p: no
truncation for all future RL runs) and ``quality_asr_num_beams=1`` (greedy reward Whisper),
under its own keys; earlier arms keep exactly their arguments (hash_snapshot: 0 of 6,154 moved).

On the REAL graph and the real builders:
  [1] epochs: 09-29 < 10-16 <= POLICY_LATEST, known; ``rl_policy_hparams`` is {} at 09-29 and both keys at 10-16;
      an unknown date raises.
  [2] every builder (rl_know_run, rl_instr_run, rl_instr_mt_run, rl_sp_run, rl_mt_run) at POLICY_LATEST yields t1 +
      num_beams 1 + the rl_grpo launcher fixes + ``declared_policy``; at 09-29 with a declared arm's spec it returns THAT
      graph job (same object: the pin is exact); an explicit key beats the rule.
  [2b] echo CER (2026-10-16: echo RL is rewarded with CER, not WER): an arm with
      an echo aux task gets ``aux_echo_metric="cer"`` at POLICY_LATEST and "wer" at 09-29; a non-echo arm at LATEST
      gets no key (job default "wer"); an explicit "wer" wins; every declared echo arm in the graph is "wer".
  [3] a builder called without ``policy`` raises (TypeError): an arm cannot be declared without choosing an epoch.
  [4] every RLFinetune in the graph was built by make_rl_run (has a known ``declared_policy``); at 09-29 or earlier
      none samples t1 unless its spec says so (the T1 ablations).
  [5] CAN FAIL: with ``rl_policy_hparams`` mutated to return {}, [2]'s POLICY_LATEST checks fail; with
      ``RULE_RL_ECHO_CER`` moved past 10-16, [2b]'s LATEST echo check fails; an RLFinetune built directly (not through
      make_rl_run) fails [4].

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_rl_policy.py
"""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
os.chdir(SETUP)
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


def mod(suffix):
    (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
    return m


runs, tpr, tsp = mod(".doriank.runs"), mod(".track_persona_rl"), mod(".track_persona_rl_selfplay")
from i6_experiments.users.dorian_koch.speech_llm.finetune import LAUNCHER_FIX_PRESETS  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.rl_finetune import RLFinetune  # noqa: E402

FAILURES = []
OLD, NEW = runs.POLICY_2026_09_29, runs.POLICY_2026_10_16


def check(ok, what, sink=None):
    """``sink`` = a mutant run: a failure is collected there (and expected), never in FAILURES."""
    tag = ("  ok   " if ok else "  FAIL ") if sink is None else ("  mutant passed  " if ok else "  mutant caught  ")
    print(tag + what, flush=True)
    if not ok:
        (FAILURES if sink is None else sink).append(what)


GRAPH_RL = [j for j in tk.sis_graph.jobs() if isinstance(j, RLFinetune)]
graph_ids = {id(j) for j in GRAPH_RL}
# snapshot BEFORE any builder runs: Sisyphus jobs are singletons per hash, so building an arm at POLICY_LATEST that
# equals a declared one (e.g. asr_b04 + t1 + greedy = the 09-29 T1 ablation) returns that object and re-stamps it
DECLARED = [(j, getattr(j, "declared_policy", None)) for j in GRAPH_RL]
# the real TrainingArtifacts' two fields the builders read, recovered from a declared arm's TTS job
asr_b04 = dict(tpr.RL_KNOW_ARMS_SPEC["ppx_rl_know_asr_b04"])
tts = next(
    j for j in GRAPH_RL if j.reward_transcript == "asr" and j.knowledge_audio is not None
).knowledge_audio.creator
ART = SimpleNamespace(
    speakers=SimpleNamespace(out_dir=tts.speaker_dir), ffmpeg=SimpleNamespace(out_path=tts.ffmpeg_path)
)


def builders(policy_kw):
    """name -> (job at the given policy, the declared arm's spec it was built from)."""
    know = {k: v for k, v in asr_b04.items() if k != "policy"}
    tr = know.pop("transcript")
    instr = {k: v for k, v in tpr.RL_INSTR_SPEC["ppx_rl_instr_asr_b04"].items() if k != "policy"}
    imt = {k: v for k, v in tpr.RL_INSTR_MT_SPEC["ppx_rl_instr_mt_asr_b04"].items() if k not in ("policy", "card")}
    sp = {k: v for k, v in tsp.RL_SP_ARMS_SPEC["ppx_rl_sp_mono"].items() if k != "policy"}
    mt = {k: v for k, v in tsp.RL_MT_SPEC["ppx_rl_mt_know_sp_b04"].items() if k != "policy"}
    return {
        "rl_know_run": tpr.rl_know_run(ART, tr, **policy_kw, **know),
        "rl_instr_run": tpr.rl_instr_run(ART, **policy_kw, **instr),
        "rl_instr_mt_run": tpr.rl_instr_mt_run(ART, **policy_kw, **{**imt, **tpr.RL_INSTR_MT_SMOKE_OVERRIDE}),
        "rl_sp_run": tsp.rl_sp_run(**policy_kw, **sp),
        "rl_mt_run": tsp.rl_mt_run(ART, **policy_kw, **mt),
    }


def latest_checks(sink):
    for name, job in builders({"policy": runs.POLICY_LATEST}).items():
        check(
            job.rollout_text_sampler == "t1"
            and job.quality_asr_num_beams == 1
            and job.launcher_fixes == LAUNCHER_FIX_PRESETS["rl_grpo"]
            and job.declared_policy == runs.POLICY_LATEST,
            f"{name} at POLICY_LATEST: sampler {job.rollout_text_sampler}, num_beams {job.quality_asr_num_beams}, "
            f"launcher fixes {'preset' if job.launcher_fixes == LAUNCHER_FIX_PRESETS['rl_grpo'] else job.launcher_fixes}",
            sink,
        )


def graph_checks(declared, sink):
    """``declared``: [(job, its declared_policy or None)]."""
    bad = [j for j, p in declared if p not in runs.KNOWN_POLICIES]
    check(not bad, f"every RLFinetune ({len(declared)}) built by make_rl_run with a known policy; bad: "
          f"{[j._sis_path() for j in bad][:3]}", sink)  # fmt: skip
    return sorted(
        os.path.basename(j._sis_path()) for j, p in declared if p and p < NEW and j.rollout_text_sampler == "t1"
    )


print("[1] epochs and rules")
check(OLD < NEW <= runs.POLICY_LATEST and NEW in runs.KNOWN_POLICIES, "09-29 < 10-16 <= POLICY_LATEST, known")
check(runs.RULE_RL_TEXT_SAMPLER_T1 == NEW == runs.RULE_RL_ASR_GREEDY, "both RL rules dated 2026-10-16")
check(runs.rl_policy_hparams(OLD) == {}, "rl_policy_hparams(09-29) == {}")
check(
    runs.rl_policy_hparams(NEW) == {"rollout_text_sampler": "t1", "quality_asr_num_beams": 1},
    "rl_policy_hparams(10-16) = t1 + num_beams 1",
)
try:
    runs.rl_policy_hparams("2026-10-17")
    check(False, "an unknown policy date raises")
except AssertionError:
    check(True, "an unknown policy date raises")

print("[2] builders")
latest_checks(None)
for name, job in builders({"policy": OLD}).items():
    check(
        id(job) in graph_ids and job.rollout_text_sampler == "released" and job.quality_asr_num_beams == 5,
        f"{name} at 09-29 with the declared spec = the graph job {os.path.basename(job._sis_path())} (released, beams 5)",
    )
know = {k: v for k, v in asr_b04.items() if k not in ("policy", "transcript")}
j = tpr.rl_know_run(ART, "asr", policy=runs.POLICY_LATEST, rollout_text_sampler="released", **know)
check(j.rollout_text_sampler == "released" and j.quality_asr_num_beams == 1, "an explicit key beats the rule")


def echo_checks(sink):
    """[2b]: the echo aux arm's task mix on _asr_b04's spec."""
    aux = tpr.RL_AUX_SPEC["ppx_rl_knowledge_continue_from_greedy_b04_plus_echo_and_question_in_system_prompt_tasks"]
    mix = dict(aux_tasks=aux["aux_tasks"], aux_prompt_set=aux["aux_prompt_set"])
    assert mix["aux_tasks"].get("echo", 0) > 0
    new = tpr.rl_know_run(ART, "asr", policy=runs.POLICY_LATEST, **know, **mix)
    old = tpr.rl_know_run(ART, "asr", policy=OLD, **know, **mix)
    plain = tpr.rl_know_run(ART, "asr", policy=runs.POLICY_LATEST, **know)
    pinned = tpr.rl_know_run(ART, "asr", policy=runs.POLICY_LATEST, aux_echo_metric="wer", **know, **mix)
    check(new.aux_echo_metric == "cer", f"echo aux arm at POLICY_LATEST: aux_echo_metric {new.aux_echo_metric}", sink)
    check(old.aux_echo_metric == "wer", f"the same arm at 09-29: aux_echo_metric {old.aux_echo_metric}", sink)
    check(
        plain.aux_echo_metric == "wer",
        f"a non-echo arm at POLICY_LATEST keeps the default ({plain.aux_echo_metric})",
        sink,
    )
    check(pinned.aux_echo_metric == "wer", "an explicit aux_echo_metric='wer' beats the rule", sink)


print("[2b] echo CER")
echo_checks(None)
echo_graph = [(j, p) for j, p in DECLARED if float((j.aux_tasks or {}).get("echo", 0)) > 0]
check(
    echo_graph and all(j.aux_echo_metric == "wer" for j, _ in echo_graph),
    f"the {len(echo_graph)} declared echo arm(s) in the graph keep 'wer' (all at {sorted({p for _, p in echo_graph})})",
)

print("[3] no policy, no arm")
for name, fn in {
    "rl_know_run": lambda: tpr.rl_know_run(ART, "asr", **know),
    "rl_instr_run": lambda: tpr.rl_instr_run(ART),
    "rl_instr_mt_run": lambda: tpr.rl_instr_mt_run(ART),
    "rl_sp_run": lambda: tsp.rl_sp_run(),
    "rl_mt_run": lambda: tsp.rl_mt_run(ART),
    "make_rl_run": lambda: runs.make_rl_run(venv_python_path=None),
}.items():
    try:
        fn()
        check(False, f"{name} without policy raises")
    except TypeError:
        check(True, f"{name} without policy raises TypeError")

print("[4] the graph")
old_t1 = graph_checks(DECLARED, None)
by = {}
for _, p in DECLARED:
    by[p] = by.get(p, 0) + 1
print(f"       RLFinetune by declared policy: {dict(sorted(by.items()))}")
#: the pre-10-16 arms that state t1 themselves: the T1 ablations (continuation + from released) and the continuation's smoke
EXPECTED_OLD_T1 = 3
#: arms declared at 10-16 (first: the question-in-prompt distillation smokes + the teacher grading pass, 2026-10-16) get
#: t1 + greedy reward Whisper from the policy; only the grading pass states the released sampler (gap-ladder rung B's)
new_jobs = [j for j, p in DECLARED if p == NEW]
check(
    bool(new_jobs) and all(j.quality_asr_num_beams == 1 for j in new_jobs),
    f"{len(new_jobs)} arm(s) at 10-16, greedy Whisper",
)
check(
    all((j.rollout_text_sampler == "t1") != bool(getattr(j, "teacher_grade_questions", 0)) for j in new_jobs),
    "10-16 arms sample t1; the teacher grading pass alone samples released",
)
check(len(old_t1) == EXPECTED_OLD_T1, f"pre-10-16 arms sampling t1 = the {EXPECTED_OLD_T1} T1 ablation jobs ({old_t1})")

print("[5] CAN FAIL")
real = runs.rl_policy_hparams
runs.rl_policy_hparams = lambda policy: {}
sink = []
try:
    latest_checks(sink)
finally:
    runs.rl_policy_hparams = real
check(len(sink) == 5, f"rl_policy_hparams -> {{}}: all 5 POLICY_LATEST builder checks fail ({len(sink)})")
real_rule = runs.RULE_RL_ECHO_CER
runs.RULE_RL_ECHO_CER = "2099-01-01"
sink = []
try:
    echo_checks(sink)
finally:
    runs.RULE_RL_ECHO_CER = real_rule
check(
    sink == [sink[0]] and "POLICY_LATEST" in sink[0] if sink else False,
    f"RULE_RL_ECHO_CER moved past 10-16: the LATEST echo check fails ({len(sink)})",
)
rogue = RLFinetune(venv_python_path=GRAPH_RL[0].venv_python_path, max_steps=3)
sink = []
graph_checks(DECLARED + [(rogue, getattr(rogue, "declared_policy", None))], sink)
check(len(sink) == 1, "an RLFinetune built directly (not via make_rl_run) fails the graph check")

print(f"\n{'PASS' if not FAILURES else 'FAIL'}: {len(FAILURES)} failure(s)")
sys.exit(1 if FAILURES else 0)
