"""Guard: our MTR-DuplexBench ASR and judge calls equal the benchmark's own, read from the vendored upstream code.

Why (duplication audit item 8, 2026-10-12): the Conversational Features runner (``mtr.eval_runner``) called
``whisper.transcribe(model, ch, language="en", vad=False, verbose=None)`` -- greedy, no temperature fallback --
while the benchmark's ASR (``Eval/asr_incremental_save.whisper_ts_inference``, the one its README names for CF) uses
``beam_size=5, best_of=5, temperature=(0.0 ... 1.0), ...``. And our Dialogue Quality judge got the joined transcript
lines, while upstream passes ``merge_stereo_segments``' ``(lines, texts)`` tuple into the f-string, so GPT-4o reads
the tuple's repr. Both went unnoticed because the comments said "verbatim". Fix (2026-10-12):
align with upstream and use its config.

Both protocols stay runnable (``mtr_duplexbench.MTR_PROTOCOL``, per-job ``code_version``): v2 = upstream's config,
v1 = our original calls (every published CF/DQ number; the v2 re-run is HELD since 2026-10-12). Reads the
vendored ``Eval/`` (``output/mtr/code``, the ``MTRCodeDownload`` clone) and drives our real code:
  [1] the keywords of each upstream ``whisper.transcribe`` call (``asr_incremental_save`` for CF v2/IF/Safety,
      ``gpt4o_mark_in_turn_GT_condor`` for DQ) equal what ``eval_runner._asr_channel``, ``asr_runner._asr_text`` and
      ``dq_asr_runner._segments`` pass to a recording fake ``whisper`` (no extra keyword, same positional count);
      CF v1 passes exactly the pinned original call; every upstream ``load_model`` names our whisper model;
  [2] judge requests: upstream's ``evaluate_instruction_following_with_gpt4o`` / ``evaluate_safety_with_gpt4o`` /
      ``evaluate_turn_segment`` (exec'd from the vendored source with a recording fake client) send the same messages
      and sampling arguments as ``mtr_duplexbench._judge_row`` at v1 and v2 -- all but the ``model`` name (our judge
      swap). DQ v2's reference is fed upstream's own ``merge_stereo_segments`` tuple, DQ v1's our joined lines
      (``dq_asr_runner.merge_segments``);
  [3] the parse: on the same replies, upstream's parse and ours give the same score.

CAN FAIL: the self-test re-runs [1]-[3] on mutants -- CF v2 = the old call, CF v1 silently on the v2 settings,
``beam_size`` 1 in the shared settings, DQ ``vad`` False, DQ v2 fed the joined lines, DQ v1 fed the v2 tuple,
IF temperature 0.0 -- each must be flagged.

Run from the setup root, no GPU, ~5 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mtr_upstream.py
"""

import ast
import importlib
import inspect
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path[:0] = [str(SETUP / "recipe"), str(SETUP / "recipe" / "sisyphus"), str(SETUP / "recipe/speech_llm/full_duplex")]
UP = (SETUP / "output" / "mtr" / "code" / "Eval").resolve()
FD_MTR = SETUP / "recipe" / "speech_llm" / "full_duplex" / "mtr"
JOB = SETUP / "recipe" / "i6_experiments" / "users" / "dorian_koch" / "speech_llm" / "mtr_duplexbench.py"


# ---------------------------------------------------------------- upstream (parsed / exec'd from the vendored source)
def _up_src(name: str) -> str:
    return (UP / name).read_text()


def _up_transcribe_kw(name: str) -> tuple[int, dict]:
    """(number of positional args, keywords) of the one ``whisper.transcribe`` call in ``whisper_ts_inference``."""
    fn = next(
        n
        for n in ast.walk(ast.parse(_up_src(name)))
        if isinstance(n, ast.FunctionDef) and n.name == "whisper_ts_inference"
    )
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "transcribe"]
    assert len(calls) == 1, f"{name}: expected one whisper.transcribe call, found {len(calls)}"
    return len(calls[0].args), {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}


def _up_load_models() -> set:
    out = set()
    for f in UP.glob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "load_model" and n.args:
                out.add((f.name, ast.literal_eval(n.args[0])))
    return out


def _up_funcs(name: str, names: list[str], **glob) -> dict:
    """The named top-level functions (and the module-level ``speaker_map`` / ``context``) of an upstream file,
    exec'd alone -- its top level loads whisper and opens an OpenAI client."""
    tree = ast.parse(_up_src(name))
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in ("speaker_map", "context") for t in n.targets))
    ]
    ns = {"re": re, **glob}
    exec(compile(ast.Module(body=keep, type_ignores=[]), f"upstream/{name}", "exec"), ns)
    return ns


# ---------------------------------------------------------------- fakes
class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, *args, **kw):
        self.calls.append((args, kw))
        return {"segments": []}


class FakeClient:
    """``client.chat.completions.create(**kw)`` -> a reply with ``content``; records every request."""

    def __init__(self, reply: str):
        self.reply, self.requests = reply, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        msg = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


# ---------------------------------------------------------------- ours (loaded from source, optionally mutated)
def _load(path: Path, modname: str, package: str, mutate: tuple[str, str] | None = None) -> types.ModuleType:
    src = path.read_text()
    if mutate:
        assert src.count(mutate[0]) == 1, f"self-test: {path.name} no longer contains {mutate[0]!r} once"
        src = src.replace(*mutate)
    mod = types.ModuleType(modname)
    mod.__file__, mod.__package__ = str(path), package
    sys.modules[modname] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def _ours(mut: dict) -> dict:
    return {
        "eval_runner": _load(FD_MTR / "eval_runner.py", "mtr._guard_eval_runner", "mtr", mut.get("eval_runner.py")),
        "asr_runner": _load(FD_MTR / "asr_runner.py", "mtr._guard_asr_runner", "mtr", mut.get("asr_runner.py")),
        "dq_asr_runner": _load(
            FD_MTR / "dq_asr_runner.py", "mtr._guard_dq_asr_runner", "mtr", mut.get("dq_asr_runner.py")
        ),
        "job": _load(
            JOB,
            "i6_experiments.users.dorian_koch.speech_llm._guard_mtr_duplexbench",
            "i6_experiments.users.dorian_koch.speech_llm",
            mut.get("mtr_duplexbench.py"),
        ),
    }


# ---------------------------------------------------------------- checks
# The CF call every published (protocol 1) number used, pinned from full_duplex a94f2957 (2026-06-29).
V1_CF_KW = dict(language="en", vad=False, verbose=None)

SEG_L = [  # left = model (DQ layout)
    {"timestamp": (0.5, 2.25), "text": " Hello, how are you?"},
    {"timestamp": (7.0, 9.875), "text": " I think it's \"fine\" -- isn't it?"},
]
SEG_R = [  # right = user
    {"timestamp": (2.4, 6.9), "text": " Pretty good. What's up with the weather"},
    {"timestamp": (10.0, 11.0), "text": " Yeah."},
]


def check(ours: dict) -> list[str]:
    import soundfile as sf

    bad = []
    # [1] ASR settings
    n_cf, kw_cf = _up_transcribe_kw("asr_incremental_save.py")
    n_dq, kw_dq = _up_transcribe_kw("gpt4o_mark_in_turn_GT_condor.py")
    with tempfile.TemporaryDirectory() as td:
        wav = f"{td}/x.wav"
        sf.write(wav, np.zeros((16000, 2), np.float32), 16000)
        drives = []
        fw = FakeWhisper()
        ours["eval_runner"]._asr_channel(fw, "M", wav, 1, 2)
        drives.append(("eval_runner._asr_channel (CF v2)", fw, n_cf, kw_cf, "upstream"))
        fw = FakeWhisper()
        ours["eval_runner"]._asr_channel(fw, "M", wav, 1, 1)
        drives.append(("eval_runner._asr_channel (CF v1)", fw, 2, V1_CF_KW, "the v1 call"))
        fw = FakeWhisper()
        ours["asr_runner"]._asr_text(fw, "M", wav, 1, 0.0)
        drives.append(("asr_runner._asr_text (IF/Safety)", fw, n_cf, kw_cf, "upstream"))
        fw = FakeWhisper()
        ours["dq_asr_runner"]._segments(fw, "M", np.zeros(16000, np.float32))
        drives.append(("dq_asr_runner._segments (DQ)", fw, n_dq, kw_dq, "upstream"))
    for what, fw, n_up, kw_up, ref in drives:
        (args, kw), *rest = fw.calls
        if rest:
            bad.append(f"{what}: {len(fw.calls)} transcribe calls for one channel")
        if len(args) != n_up:
            bad.append(f"{what}: {len(args)} positional args, {ref} {n_up}")
        if kw != kw_up:
            diff = {
                k: (kw.get(k, "<absent>"), kw_up.get(k, "<absent>"))
                for k in set(kw) | set(kw_up)
                if kw.get(k, "<a>") != kw_up.get(k, "<a>")
            }
            bad.append(f"{what}: transcribe keywords differ from {ref} (ours, {ref}): {diff}")
    models = _up_load_models()
    from mtr import upstream_asr

    wrong = {m for m in models if m[1] != upstream_asr.WHISPER_MODEL}
    if wrong or not models:
        bad.append(f"upstream load_model {sorted(models)} vs ours {upstream_asr.WHISPER_MODEL!r}")
    ev_default = inspect.signature(ours["job"].MTRDuplexEval.__init__).parameters["whisper_model"].default
    if ev_default != upstream_asr.WHISPER_MODEL:
        bad.append(f"MTRDuplexEval whisper_model default {ev_default!r} != {upstream_asr.WHISPER_MODEL!r}")

    # [2] + [3] judge requests and parse
    job = ours["job"]
    up_if = _up_funcs(
        "instruction_following_evaluation.py",
        ["evaluate_instruction_following_with_gpt4o", "parse_instruction_following_evaluation"],
    )
    up_sf = _up_funcs("safety_evaluation.py", ["evaluate_safety_with_gpt4o"])
    up_dq = _up_funcs(
        "gpt4o_mark_in_turn_GT_condor.py",
        ["merge_stereo_segments", "evaluate_turn_segment", "extract_score_from_response"],
    )
    user, model_text = (
        "  Tell me a joke about cats. ",
        " Why did the cat sit on the computer? To keep an eye on the mouse. ",
    )
    dq_up_transcript = up_dq["merge_stereo_segments"](SEG_L, SEG_R)  # upstream returns (lines, texts)
    dq_ours_transcript = ours["dq_asr_runner"].merge_segments({0: SEG_L, 1: SEG_R})
    seg_start, seg_end = 2.4, 9.875
    cases = [
        (
            "instruction_following",
            ["1", "0", "Answer: 1", "I'd say 0.", "yes"],
            lambda c, v: (
                up_if.__setitem__("client", c) or up_if["evaluate_instruction_following_with_gpt4o"](user, model_text)
            ),
            lambda r: r.get("instruction_following_score"),
            {"user_text": user, "model_text": model_text},
        ),
        (
            "safety",
            ["0", "1", " 1\n", "Score: 0", "none"],
            lambda c, v: up_sf.__setitem__("client", c) or up_sf["evaluate_safety_with_gpt4o"](model_text),
            lambda r: None if r == -1 else r,
            {"user_text": user, "model_text": model_text},
        ),
        (
            "dialogue_quality",
            ["3.5", "4", "Score: 2.5", "5.0", "n/a"],
            # v2: upstream's own (lines, texts) tuple; v1: upstream's prompt around our joined lines
            lambda c, v: (
                up_dq.__setitem__("client", c)
                or up_dq["evaluate_turn_segment"](
                    dq_up_transcript if v == 2 else dq_ours_transcript, seg_start, seg_end
                )
            ),
            lambda r: r,
            {"transcript": dq_ours_transcript, "seg_start": seg_start, "seg_end": seg_end},
        ),
    ]
    for (dim, replies, run_up, up_score, row), v in [(c, v) for c in cases for v in (1, 2)]:
        for reply in replies:
            cu, co = FakeClient(reply), FakeClient(reply)
            s_up = up_score(run_up(cu, v))
            s_ours = job._judge_row(co, "JUDGE", dim, row, v)
            (ku,), (ko,) = cu.requests, co.requests
            ku = {k: v for k, v in ku.items() if k != "model"}
            ko = {k: v for k, v in ko.items() if k != "model"}
            if ku != ko:
                diff = {
                    k: (ko.get(k, "<absent>"), ku.get(k, "<absent>"))
                    for k in set(ku) | set(ko)
                    if ko.get(k, "<a>") != ku.get(k, "<a>")
                }
                bad.append(f"judge {dim} v{v}: request differs from reference (ours, ref): {str(diff)[:300]}")
                break
            if (None if s_up is None else float(s_up)) != s_ours:
                bad.append(f"judge {dim} v{v}: reply {reply!r} -> upstream {s_up!r}, ours {s_ours!r}")
    return bad


MUTATIONS = [
    (
        "eval_runner.py",
        ("    2: TRANSCRIBE_KW,", '    2: dict(language="en", vad=False, verbose=None),'),
        "CF v2 = the old greedy call",
        "eval_runner._asr_channel (CF v2)",
    ),
    (
        "eval_runner.py",
        ('    1: dict(language="en", vad=False, verbose=None),', "    1: TRANSCRIBE_KW,"),
        "CF v1 silently runs the v2 settings",
        "eval_runner._asr_channel (CF v1)",
    ),
    (None, ("TRANSCRIBE_KW", "beam_size", 1), "shared settings beam_size 1", "asr_runner._asr_text"),
    (None, ("DQ_TRANSCRIBE_KW", "vad", False), "DQ vad False", "dq_asr_runner._segments"),
    (
        "mtr_duplexbench.py",
        ("2: _dq_upstream_transcript}", "2: lambda t: t}"),
        "DQ judge v2 sees the joined lines only",
        "judge dialogue_quality v2",
    ),
    (
        "mtr_duplexbench.py",
        ("{1: lambda t: t,", "{1: _dq_upstream_transcript,"),
        "DQ judge v1 silently gets the v2 input",
        "judge dialogue_quality v1",
    ),
    (
        "mtr_duplexbench.py",
        ("temperature=0.1,", "temperature=0.0,"),
        "IF judge temperature 0.0",
        "judge instruction_following",
    ),
]


def main():
    assert UP.is_dir(), f"vendored upstream Eval/ not found at {UP} (output/mtr/code -> MTRCodeDownload)"
    upstream_asr = importlib.import_module("mtr.upstream_asr")
    bad = check(_ours({}))
    failed = bool(bad)
    print(f"[{'ok' if not bad else 'FAIL'}] v2 ASR/judge = upstream, v1 = the original calls ({UP})")
    for b in bad:
        print(f"   - {b}")
    for f, mut, what, expect in MUTATIONS:
        if f is None:  # mutate a shared settings dict in place, then restore
            d = getattr(upstream_asr, mut[0])
            old = d[mut[1]]
            d[mut[1]] = mut[2]
            try:
                got = [g for g in check(_ours({})) if expect in g]
            finally:
                d[mut[1]] = old
        else:
            got = [g for g in check(_ours({f: mut})) if expect in g]
        failed |= not got
        print(f"[{'ok' if got else 'FAIL'}] can fail: {what:46s} -> {(got or ['NOT flagged'])[0][:110]}")
    if failed:
        sys.exit(1)
    print("\nMTR protocol 2 equals the benchmark's calls; protocol 1 reproduces the original ones")


if __name__ == "__main__":
    main()
