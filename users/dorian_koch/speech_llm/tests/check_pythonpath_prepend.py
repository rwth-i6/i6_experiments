"""Guard: every job that PREPENDS to the inherited PYTHONPATH spells it through ``common.prepend_pythonpath``.

Why (duplication audit item 8, 2026-10-12): the idiom ``prefix + (os.pathsep + env["PYTHONPATH"] if ... else "")``
was hand-rolled in six places (``worker_env``, ``VoicePromptCodes``, ``EvalVoiceCodes``, ``TurnBenchScore``,
``mtr_duplexbench._pythonpath_env``, ``MoshiAnnotate``). They agreed; the helper keeps them agreeing. Jobs that
pass ``extra_env={"PYTHONPATH": ...}`` to ``run_worker_script`` REPLACE the inherited value on purpose (standalone
worker scripts in their own venvs) and are not touched.

Checks:
  [1] ``prepend_pythonpath`` semantics: prefix first, the old value kept after one ``os.pathsep``, no stray separator
      when there is no old value;
  [2] the REAL callers, driven with a PYTHONPATH set and unset: ``worker_env`` (all three ``pythonpath=`` modes),
      ``mtr_duplexbench._pythonpath_env``, and the ``run()`` of ``VoicePromptCodes`` / ``EvalVoiceCodes`` /
      ``TurnBenchScore`` (``subprocess.run`` intercepted; the env it would have received is inspected);
  [3] no module under ``speech_llm/`` or ``full_duplex/sis_recipe/`` hand-rolls the idiom again.

CAN FAIL: the self-test re-runs [2] with a ``prepend_pythonpath`` that REPLACES (drops the inherited value) and one
that appends, and [3] on an in-memory copy of ``selfplay_bench.py`` with the old hand-rolled line put back; each must
be flagged.

Run from the setup root, no GPU, ~2 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_pythonpath_prepend.py
"""

import os
import re
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import i6_experiments.users.dorian_koch.speech_llm.common as common  # noqa: E402
import i6_experiments.users.dorian_koch.speech_llm.mtr_duplexbench as mtr  # noqa: E402
import i6_experiments.users.dorian_koch.speech_llm.persona_prompts as pp  # noqa: E402
import i6_experiments.users.dorian_koch.speech_llm.selfplay_bench as sb  # noqa: E402
import i6_experiments.users.dorian_koch.speech_llm.turnbench as tb  # noqa: E402

SPEECH_LLM = SETUP / "recipe" / "i6_experiments" / "users" / "dorian_koch" / "speech_llm"
SIS_RECIPE = SETUP / "recipe" / "speech_llm" / "full_duplex" / "sis_recipe"
#: modules that bind ``prepend_pythonpath`` at import (patched there for the self-test)
BINDERS = (common, pp, sb, tb)  # mtr_duplexbench imports it from common at call time
#: the hand-rolled spellings that existed before the helper
HANDROLLED = re.compile(
    r"os\.pathsep\s*\+\s*env\[\s*[\"']PYTHONPATH[\"']\s*\]\)?\s*if\s+env\.get\(\s*[\"']PYTHONPATH|"
    r"\{os\.pathsep\}\{existing\}"
)
OLD = "/old/a" + os.pathsep + "/old/b"


class _Captured(Exception):
    def __init__(self, env):
        self.env = env


class _Stub:
    """Any attribute a job's run() reads: paths answer ``get()``/``get_path()``, lists iterate empty."""

    def __init__(self, path="/stub"):
        self._p = path

    def get(self):
        return self._p

    def get_path(self):
        return self._p

    def __iter__(self):
        return iter(())

    def __str__(self):
        return self._p


def _fake_job(cls, **attrs):
    job = object.__new__(cls)
    for k, v in attrs.items():
        object.__setattr__(job, k, v)
    return job


def _env_of_run(module, job) -> dict:
    """The env ``job.run()`` hands to ``subprocess.run`` (intercepted before anything is spawned)."""
    real = module.subprocess.run

    def fake(cmd, *a, env=None, **kw):
        raise _Captured(env)

    module.subprocess.run = fake
    try:
        job.run()
    except _Captured as c:
        return c.env
    finally:
        module.subprocess.run = real
    raise AssertionError(f"{type(job).__name__}.run() never called subprocess.run")


def _with_pythonpath(value, fn):
    saved = os.environ.get("PYTHONPATH")
    try:
        if value is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = value
        return fn()
    finally:
        if saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved


def callers(tmp: str) -> dict:
    """{caller name: (expected prefix, fn() -> env)} -- the REAL code paths."""
    pred = os.path.join(tmp, "pred.jsonl")
    Path(pred).write_text("")
    voice_kw = dict(voice_sec=4.0, venv_python_path=_Stub("/venv/python"), out_dir=_Stub(os.path.join(tmp, "o")))
    return {
        "worker_env(pythonpath='moshi')": (
            common.moshi_pythonpath(),
            lambda: common.worker_env(offline=False, hf_cache=False),
        ),
        "worker_env(pythonpath='lib')": (
            common.moshi_family_lib_parent(),
            lambda: common.worker_env(offline=False, pythonpath="lib", hf_cache=False),
        ),
        "worker_env(pythonpath='/x/y')": (
            "/x/y",
            lambda: common.worker_env(offline=False, pythonpath="/x/y", hf_cache=False),
        ),
        "mtr_duplexbench._pythonpath_env": (common.moshi_family_lib_parent(), mtr._pythonpath_env),
        "VoicePromptCodes.run": (
            common.moshi_pythonpath(),
            lambda: _env_of_run(pp, _fake_job(pp.VoicePromptCodes, episode_shard_dirs=[], **voice_kw)),
        ),
        "EvalVoiceCodes.run": (
            common.moshi_pythonpath(),
            lambda: _env_of_run(
                sb,
                _fake_job(
                    sb.EvalVoiceCodes,
                    n=1,
                    seed=0,
                    subset="s",
                    repo="r",
                    revision="v",
                    min_active=0.5,
                    **voice_kw,
                ),
            ),
        ),
        "TurnBenchScore.run": (
            tmp,
            lambda: _env_of_run(
                tb,
                _fake_job(
                    tb.TurnBenchScore,
                    repo=_Stub(tmp),
                    predictions_rel="pred.jsonl",
                    predictions_path=None,
                    venv_python=_Stub("/venv/python"),
                    dataset=None,
                ),
            ),
        ),
    }


def check_callers(tmp: str) -> list[str]:
    bad = []
    for name, (prefix, fn) in callers(tmp).items():
        got_old = _with_pythonpath(OLD, fn).get("PYTHONPATH")
        got_none = _with_pythonpath(None, fn).get("PYTHONPATH")
        if got_old != prefix + os.pathsep + OLD:
            bad.append(f"{name}: with PYTHONPATH={OLD!r} got {got_old!r}, want {prefix + os.pathsep + OLD!r}")
        if got_none != prefix:
            bad.append(f"{name}: with no PYTHONPATH got {got_none!r}, want {prefix!r}")
    return bad


def check_source(texts: dict[str, str]) -> list[str]:
    bad = []
    for name, src in texts.items():
        for ln, line in enumerate(src.splitlines(), 1):
            if HANDROLLED.search(line) and "def prepend_pythonpath" not in line:
                bad.append(f"{name}:{ln}: hand-rolled PYTHONPATH prepend -- use common.prepend_pythonpath")
    return bad


def _sources() -> dict[str, str]:
    files = [p for d in (SPEECH_LLM, SIS_RECIPE) for p in d.rglob("*.py") if "__pycache__" not in p.parts]
    return {str(p.relative_to(SETUP / "recipe")): p.read_text() for p in files if p.name != Path(__file__).name}


def _patched(fn, impl):
    saved = [(m, m.prepend_pythonpath) for m in BINDERS]
    try:
        for m in BINDERS:
            m.prepend_pythonpath = impl
        return fn()
    finally:
        for m, f in saved:
            m.prepend_pythonpath = f


def main():
    failed = False
    p = common.prepend_pythonpath
    sem = {
        "no old value": (p("/a", None), "/a"),
        "empty old value": (p("/a", ""), "/a"),
        "old value kept after the prefix": (p("/a", "/b:/c"), "/a" + os.pathsep + "/b:/c"),
    }
    for what, (got, want) in sem.items():
        ok = got == want
        failed |= not ok
        print(f"[{'ok' if ok else 'FAIL'}] [1] prepend_pythonpath: {what}: {got!r}")

    with tempfile.TemporaryDirectory() as tmp:
        bad = check_callers(tmp)
        n = len(callers(tmp))
        failed |= bool(bad)
        print(
            f"[{'ok' if not bad else 'FAIL'}] [2] {n} real callers prepend exactly once"
            + "".join(f"\n   - {b}" for b in bad)
        )

        src = _sources()
        bad3 = check_source(src)
        failed |= bool(bad3)
        print(
            f"[{'ok' if not bad3 else 'FAIL'}] [3] {len(src)} recipe modules, no hand-rolled prepend"
            + "".join(f"\n   - {b}" for b in bad3)
        )

        # self-test: each mutation must be flagged
        replace = lambda prefix, existing: prefix  # noqa: E731
        append = lambda prefix, existing: (existing + os.pathsep if existing else "") + prefix  # noqa: E731
        for what, impl in (
            ("a prepend that REPLACES the inherited value", replace),
            ("a prepend that APPENDS", append),
        ):
            got = _patched(lambda: check_callers(tmp), impl)
            # every caller must be caught (two lines each: with and without an inherited PYTHONPATH)
            hit = {g.split(":")[0] for g in got}
            ok = hit == set(callers(tmp))
            failed |= not ok
            print(f"[{'ok' if ok else 'FAIL'}] can fail: {what:46s} -> {len(hit)}/{len(callers(tmp))} callers flagged")
        key = "i6_experiments/users/dorian_koch/speech_llm/selfplay_bench.py"
        new = 'env["PYTHONPATH"] = prepend_pythonpath(_moshi_pythonpath(), env.get("PYTHONPATH"))'
        old = 'env["PYTHONPATH"] = _moshi_pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")'
        assert new in src[key], "self-test: selfplay_bench no longer calls prepend_pythonpath -- update the mutation"
        got = check_source({key: src[key].replace(new, old, 1)})
        ok = bool(got)
        failed |= not ok
        print(
            f"[{'ok' if ok else 'FAIL'}] can fail: {'the old hand-rolled line put back':46s} -> {(got or ['NOT flagged'])[0][:100]}"
        )
    if failed:
        sys.exit(1)
    print("\nall PYTHONPATH prepends go through common.prepend_pythonpath")


if __name__ == "__main__":
    main()
