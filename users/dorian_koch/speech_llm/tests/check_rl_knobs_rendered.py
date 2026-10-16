"""Guard: every hash-relevant ``RLFinetune`` knob set away from its default reaches the launcher config.

A hashed knob that is never rendered changes the job's identity and nothing else: the run trains exactly like its
control under a new name (2026-10-02: init_lora missing for self-play; 2026-10-12 audit: self-play never rendered
advantage_split / scale_std / brevity_weight / garbled_weight, base-Moshi interactivity runs never rendered
lr_schedule / loss_norm / log_every / skip_zero_signal / probe_every). ``write_config`` now calls
``rl_finetune.check_knobs_rendered`` (RuntimeError), which compares every ``__init__`` default against the rendered
text, with an explicit ``KNOBS_CONSUMED_ELSEWHERE`` allowlist (rqmt, manifests, judge rubric, run()-side judge).

Can fail: a render line dropped (mutant below, driven through the real ``write_config``); a knob set on a run kind
whose launcher branch does not read it (brevity_weight on base-Moshi interactivity); a default-valued knob renders a
line (old configs would no longer be byte-identical).

``--graph`` additionally loads the whole recipe graph and, for every registered RLFinetune: the guard on today's
render (must pass), the guard on the job's existing ``output/config.yaml`` (knobs that were silently ignored when it
ran -- reported, not fatal) and whether today's render differs from that file.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_rl_knobs_rendered.py [--graph]
"""

import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import sisyphus.job  # noqa: E402
from sisyphus import tk  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.rl_finetune as rf  # noqa: E402

VENV = tk.Path("/dev/null/venv/python", hash_overwrite="knobs_guard_venv")
SCEN = tk.Path("/dev/null/scenarios", hash_overwrite="knobs_guard_scenarios")


def rl(**kw):
    sisyphus.job.created_jobs.clear()
    return rf.RLFinetune(venv_python_path=VENV, max_steps=100, **kw)


def selfplay(**kw):
    return rl(family="personaplex", segment_source="selfplay", selfplay_scenarios=SCEN, quality_judge="judge", **kw)


def write_config(job, tmp):
    """The real write_config, its outputs redirected to ``tmp``; returns the rendered config text."""
    for attr in ("out_rundir", "out_config", "out_selfplay_manifest", "out_judge_rubric", "out_knowledge_manifest"):
        if hasattr(job, attr):
            setattr(job, attr, tk.Path(os.path.join(tmp, attr)))
    job.write_config()
    return Path(tmp, "out_config").read_text()


def raises(fn, needle):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"guard did not fire for {needle}")


def unit():
    # at the defaults: none of the new lines (earlier configs stay byte-identical)
    for job, keys in ((rl(), rf._INTERACTIVITY_EXTRA_KEYS), (selfplay(), rf._SELFPLAY_EXTRA_KEYS)):
        text = rf._render_rl_config(job)
        assert not [k for k in keys if f"\n{k}: " in text], (job.segment_source, text)
        assert not rf.unrendered_knobs(job, text), rf.unrendered_knobs(job, text)
    inter = dict(lr_schedule="constant", loss_norm="constant", log_every=5, skip_zero_signal=True, probe_every=50)
    sp = dict(advantage_split=True, scale_std=False, brevity_weight=0.0, garbled_weight=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        text = write_config(rl(**inter), tmp)
        for line in ('lr_schedule: "constant"', 'loss_norm: "constant"', "log_every: 5", "skip_zero_signal: true",
                     "probe_every: 50"):  # fmt: skip
            assert f"\n{line}\n" in text, line
    with tempfile.TemporaryDirectory() as tmp:
        text = write_config(selfplay(**sp), tmp)
        for line in ("advantage_split: true", "scale_std: false", "brevity_weight: 0.0", "garbled_weight: 0.5"):
            assert f"\n{line}\n" in text, line
    # the rollout text sampler (every run kind): hash-neutral at "released", re-hashes at "t1"; absent at "released",
    # rendered at "t1"
    sid = rl()._sis_id()
    assert rl(rollout_text_sampler="released")._sis_id() == sid != rl(rollout_text_sampler="t1")._sis_id()
    for job_fn in (rl, selfplay):
        assert "rollout_text_sampler" not in rf._render_rl_config(job_fn())
        with tempfile.TemporaryDirectory() as tmp:
            assert '\nrollout_text_sampler: "t1"\n' in write_config(job_fn(rollout_text_sampler="t1"), tmp)
    # a knob the run kind's launcher branch does not read: base-Moshi interactivity ignores the knowledge reward weights
    with tempfile.TemporaryDirectory() as tmp:
        raises(lambda: write_config(rl(brevity_weight=0.0), tmp), "brevity_weight")
    with tempfile.TemporaryDirectory() as tmp:  # a manifest knob on a run without that manifest
        raises(lambda: write_config(rl(n_probe=8), tmp), "n_probe")
    # mutants: one render line dropped -> the real write_config refuses
    for attr, drop, job_fn in (("_SELFPLAY_EXTRA_KEYS", "garbled_weight", lambda: selfplay(**sp)),
                               ("_INTERACTIVITY_EXTRA_KEYS", "lr_schedule", lambda: rl(**inter)),
                               ("_COMMON_EXTRA_KEYS", "rollout_text_sampler",
                                lambda: rl(rollout_text_sampler="t1"))):  # fmt: skip
        keep = getattr(rf, attr)
        setattr(rf, attr, tuple(k for k in keep if k != drop))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                raises(lambda: write_config(job_fn(), tmp), drop)
        finally:
            setattr(rf, attr, keep)
    qctx_unit()
    print("check_rl_knobs_rendered: unit ok (defaults byte-stable, knobs rendered, 5 mutants + 2 misuses caught)")


KQ = tk.Path("/dev/null/questions", hash_overwrite="knobs_guard_questions")
GATE = tk.Path("/dev/null/teacher_grades.jsonl", hash_overwrite="knobs_guard_gate")


def know(**kw):
    return rl(family="personaplex", segment_source="knowledge", knowledge_questions=KQ, knowledge_audio=KQ,
              quality_judge="judge", reward_transcript="asr", skip_zero_signal=True, **kw)  # fmt: skip


def qctx_unit():
    """Question-in-prompt OPD (2026-10-16): the teacher gate, opd_only and the grading pass are hash-neutral at their
    defaults, absent from the render then, rendered when set; mutants: a dropped render line -> write_config refuses."""
    qctx = dict(text_teacher="qctx_speech", text_teacher_weight=0.1)
    sid = know(**qctx)._sis_id()
    assert know(**qctx, text_teacher_gate=None, opd_only=False, teacher_grade_questions=0)._sis_id() == sid
    assert know(**qctx, text_teacher_gate=GATE)._sis_id() != sid != know(**qctx, opd_only=True)._sis_id()
    plain = rf._render_rl_config(know(**qctx))
    assert not [k for k in ("text_teacher_gate", "opd_only", "teacher_grade_questions") if k in plain], plain
    full = dict(**qctx, text_teacher_gate=GATE, opd_only=True)
    with tempfile.TemporaryDirectory() as tmp:
        text = write_config(know(**full), tmp)
        for line in ('text_teacher: "qctx_speech"', f'text_teacher_gate: "{GATE.get()}"', "opd_only: true"):
            assert f"\n{line}\n" in text, line
    with tempfile.TemporaryDirectory() as tmp:
        assert "\nteacher_grade_questions: 8000\n" in write_config(know(teacher_grade_questions=8000), tmp)
    keep = rf._render_knowledge_keys
    for drop, kw in (("text_teacher_gate", full), ("opd_only", full)):
        rf._render_knowledge_keys = lambda job, d=drop: "".join(
            x for x in keep(job).splitlines(keepends=True) if not x.startswith(d + ":")
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                raises(lambda: write_config(know(**kw), tmp), drop)
        finally:
            rf._render_knowledge_keys = keep


def graph():
    import logging

    logging.disable(logging.WARNING)
    os.chdir(SETUP)
    from sisyphus.loader import config_manager

    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
    jobs = sorted((j for j in tk.sis_graph.jobs() if isinstance(j, rf.RLFinetune)), key=lambda j: j._sis_id())
    trip_now, n_disk = [], 0
    for job in jobs:
        sid = job._sis_id()
        text = rf._render_rl_config(job)
        bad = rf.unrendered_knobs(job, text)
        if bad:
            trip_now.append((sid, bad))
        disk = Path(job._sis_path(), "output", "config.yaml")
        if not disk.exists():  # (a finished job keeps output/ next to finished.tar.gz)
            print(f"{sid}: write_config not run yet ({job.segment_source}){'  TRIPS ' + str(bad) if bad else ''}")
            continue
        n_disk += 1
        old = disk.read_text()
        bad_old = rf.unrendered_knobs(job, old)
        diff = sorted(set(text.splitlines()) ^ set(old.splitlines()))
        if bad or bad_old or diff:
            print(f"{sid} ({job.segment_source}): now={bad} on_disk_ignored={bad_old} render_vs_disk_changed={diff}")
    print(f"# {len(jobs)} RLFinetune jobs, {n_disk} with a written config; {len(trip_now)} trip the guard today")
    if trip_now:
        raise RuntimeError(f"RLFinetune jobs trip check_knobs_rendered: {trip_now}")


if __name__ == "__main__":
    unit()
    if "--graph" in sys.argv:
        graph()
