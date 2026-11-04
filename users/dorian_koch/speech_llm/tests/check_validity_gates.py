"""Guard: a job whose measurement broke fails instead of publishing a plausible number (review 2026-11-04).

* ``LLMGrading`` scores a judge failure as a wrong answer: above ``JUDGE_MAX_ERROR_FRAC`` the job must fail.
* ``MTRJudge`` drops unparseable verdicts: zero scored rows used to publish ``safe_rate`` 1.0.
* ``annotate_hf_to_hf`` dropped any number of failed rows and still saved the corpus.
* The MFDB interruption / turn-taking scorers crashed on a model that never takes the turn, and a judge-parse
  retry counted the clip's TOR again.
* ``RLTrainer`` stepped the optimizer on a non-finite gradient norm (``run_training`` refuses it).
* ``restore_trainer_state`` resumed a run under a different DDP world size (different effective batch).

Each gate is driven through its real function and its call site is checked in the real caller's source; every
source check is also shown to fail on a mutated copy.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_validity_gates.py
"""

import ast
import json
import os
import sys
import tempfile
import types
from pathlib import Path

SETUP = next((p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir()), Path.cwd())
REC = SETUP / "recipe"
sys.path.insert(0, str(REC))
sys.path.insert(0, str(REC / "sisyphus"))
sys.path.insert(0, str(REC / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

SL = REC / "i6_experiments" / "users" / "dorian_koch" / "speech_llm"
FDB_EVAL = REC / "moshified_fdb_v1_v15" / "evaluation"
FAILURES: list[str] = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


def raises(fn, *args):
    try:
        fn(*args)
    except RuntimeError:
        return True
    return False


def calls_before(src: str, func: str, gate: str, before: str) -> bool:
    """True iff ``func``'s body in ``src`` calls ``gate`` on a line before the first line containing ``before``."""
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func)
    body = ast.get_source_segment(src, node).splitlines()
    gate_line = next((i for i, ln in enumerate(body) if f"{gate}(" in ln and "def " not in ln), None)
    before_line = next((i for i, ln in enumerate(body) if before in ln), None)
    return gate_line is not None and before_line is not None and gate_line < before_line


def gate_case(name, src, func, gate, before):
    check(calls_before(src, func, gate, before), f"{name}: {func}() calls {gate}() before `{before}`")
    mutated = src.replace(f"{gate}(", "_removed_gate(")
    check(not calls_before(mutated, func, gate, before), f"{name}: the call-site check fails without the call")


def class_src(path: Path, name: str) -> str:
    src = path.read_text()
    return ast.get_source_segment(
        src, next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == name)
    )


def load_defs(path: Path, names: set) -> dict:
    """Execute only the named top-level assignments/functions of a module whose imports this env lacks."""
    tree = ast.parse(path.read_text())
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


def check_knowledge_judge_gate():
    from i6_experiments.users.dorian_koch.speech_llm import knowledge_benchmark as kb

    check(not raises(kb.check_judge_error_rate, 1, 1000), "knowledge: 1/1000 judge errors (the measured max) pass")
    check(raises(kb.check_judge_error_rate, 11, 1000), "knowledge: 11/1000 judge errors fail the job")
    check(raises(kb.check_judge_error_rate, 0, 0), "knowledge: zero graded rows fail the job")
    gate_case(
        "knowledge",
        class_src(SL / "knowledge_benchmark.py", "LLMGrading"),
        "run",
        "check_judge_error_rate",
        "out_summary",
    )


def check_mtr_parse_gate():
    from i6_experiments.users.dorian_koch.speech_llm import mtr_duplexbench as mtr

    check(not raises(mtr.check_judge_parse_rate, 998, 1000), "MTR: 2/1000 unparseable (the measured max) pass")
    check(raises(mtr.check_judge_parse_rate, 0, 3), "MTR: zero parseable verdicts fail (was safe_rate 1.0)")
    check(raises(mtr.check_judge_parse_rate, 980, 1000), "MTR: 20/1000 unparseable fail")
    gate_case("MTR", class_src(SL / "mtr_duplexbench.py", "MTRJudge"), "run", "check_judge_parse_rate", "summary = {")


def check_annotate_gate():
    path = SL / "moshi_annotate_inference.py"
    ns = load_defs(path, {"ANNOTATE_MAX_DROP_FRAC", "check_annotate_drop_rate"})
    gate = ns["check_annotate_drop_rate"]
    check(not raises(gate, 2500, 2500), "annotation: no dropped row passes")
    check(raises(gate, 1000, 980), "annotation: 20/1000 dropped rows fail the part")
    check(raises(gate, 5, 0), "annotation: an empty output fails the part")
    src = path.read_text()
    gate_case("annotation", src, "annotate_hf_to_hf", "check_annotate_drop_rate", "save_to_disk(out_dir)")
    check('_counters["empty_alignments"] +=' in src, "annotation: kept rows with no words are counted")


def _run_interruption(root: str) -> dict:
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda rows, **_: rows))
    ns: dict = {}
    exec(compile((FDB_EVAL / "eval_user_interruption.py").read_text(), "eval_user_interruption.py", "exec"), ns)
    return ns["eval_user_interruption"](root, None)


def check_fdb_all_silent():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            d = Path(tmp) / str(i)
            d.mkdir()
            (d / "interrupt.json").write_text(json.dumps([{"interrupt": "q", "context": "c", "timestamp": [0, 5]}]))
            (d / "output.json").write_text(json.dumps({"text": "", "chunks": []}))
        try:
            res = _run_interruption(tmp)
        except ZeroDivisionError as e:
            check(False, f"MFDB interruption: an all-silent model crashes the scorer ({e})")
        else:
            check(res["average_take_turn"] == 0, f"MFDB interruption: an all-silent model scores TOR 0 ({res})")

    # turn taking: stub the audio stack, every clip without a transcript -> TOR 0, no latency
    fake_torch = types.SimpleNamespace(
        from_numpy=lambda a: types.SimpleNamespace(t=lambda: types.SimpleNamespace(contiguous=lambda: None))
    )
    stubs = {
        "moshi_fixes": types.SimpleNamespace(
            get_speech_timestamps=lambda *a, **k: [], non_greeting_speech_segments=None
        ),
        "torchaudio": types.SimpleNamespace(),
        "soundfile": types.SimpleNamespace(read=lambda *a, **k: (None, 24000)),
        "torch": fake_torch,
        "tqdm": types.SimpleNamespace(tqdm=lambda rows, **_: rows),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        ns: dict = {}
        exec(compile((FDB_EVAL / "eval_smooth_turn_taking.py").read_text(), "eval_smooth_turn_taking.py", "exec"), ns)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "0"
            d.mkdir()
            (d / "turn_taking.json").write_text(json.dumps([{"timestamp": [3, 4]}]))
            (d / "output.json").write_text(json.dumps({"text": "", "chunks": []}))
            (d / "output.wav").write_bytes(b"")
            try:
                res = ns["eval_smooth_turn_taking"](tmp)
            except (ZeroDivisionError, IndexError) as e:
                check(False, f"MFDB turn taking: an all-silent model crashes the scorer ({e!r})")
            else:
                check(
                    res["average_take_turn"] == 0 and res["median_latency"] is None,
                    f"MFDB turn taking: an all-silent model scores TOR 0, latency None ({res})",
                )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def check_fdb_retry_counts_once():
    """One silent clip + one answering clip whose first judge reply is unparseable: TOR must be 1/2, not 2/3."""
    replies = iter(["no verdict here", "Analysis: fine.\nI would rate the AI's response as 4"])

    def create(**_):
        msg = types.SimpleNamespace(content=next(replies))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda rows, **_: rows))
    ns: dict = {}
    exec(compile((FDB_EVAL / "eval_user_interruption.py").read_text(), "eval_user_interruption.py", "exec"), ns)
    words = [{"text": w, "timestamp": [5.5 + 0.5 * i, 5.9 + 0.5 * i]} for i, w in enumerate("sure let us talk".split())]
    with tempfile.TemporaryDirectory() as tmp:
        for name, chunks in (("0", []), ("1", words)):
            d = Path(tmp) / name
            d.mkdir()
            (d / "interrupt.json").write_text(json.dumps([{"interrupt": "q", "context": "c", "timestamp": [0, 5]}]))
            (d / "output.json").write_text(json.dumps({"text": "sure let us talk", "chunks": chunks}))
        res = ns["eval_user_interruption"](tmp, client, "fake")
    check(res["average_take_turn"] == 0.5, f"MFDB interruption: a judge retry counts the clip's TOR once ({res})")


def check_rl_finite_grad():
    src = (REC / "speech_llm" / "full_duplex" / "moshi_family" / "rl" / "trainer.py").read_text()

    def ok(s):
        i_clip = s.find("clip_grad_norm_(self.params")
        i_fin = s.find("math.isfinite(grad_norm)", i_clip)
        i_step = s.find("self.opt.step()", i_clip)
        return -1 not in (i_clip, i_fin, i_step) and i_fin < i_step

    check(ok(src), "RL: the trainer checks the grad norm is finite before the optimizer step")
    check(not ok(src.replace("math.isfinite(grad_norm)", "True")), "RL: the source check fails without it")


def check_resume_world_size():
    import torch
    from moshi_family import train_loop

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        path = train_loop.save_trainer_state(Path(tmp), step=5, optimizer=opt)
        check(torch.load(path, weights_only=False).get("world") == 1, "resume: the sidecar records the world size")
        train_loop.restore_trainer_state(path, optimizer=opt, expect_step=5)
        check(True, "resume: the same world size resumes")
        orig = train_loop._world_size
        train_loop._world_size = lambda: 4
        try:
            check(
                raises(lambda: train_loop.restore_trainer_state(path, optimizer=opt)),
                "resume: a different world size is refused (effective batch would change)",
            )
        finally:
            train_loop._world_size = orig
        legacy = torch.load(path, weights_only=False)
        legacy.pop("world")
        torch.save(legacy, path)
        train_loop._world_size = lambda: 4
        try:
            train_loop.restore_trainer_state(path, optimizer=opt)
            check(True, "resume: a sidecar from before the key still resumes")
        finally:
            train_loop._world_size = orig


def main():
    for fn in (
        check_knowledge_judge_gate,
        check_mtr_parse_gate,
        check_annotate_gate,
        check_fdb_all_silent,
        check_fdb_retry_counts_once,
        check_rl_finite_grad,
        check_resume_world_size,
    ):
        print(fn.__name__)
        fn()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED")
        sys.exit(1)
    print("\nall validity gates hold")


if __name__ == "__main__":
    main()
