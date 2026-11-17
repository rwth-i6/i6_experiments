"""Guard: ChatterboxInference synthesizes with the batched path (``chatterbox_fast``) unless a caller explicitly asks
for the per-utterance one, and that choice never moves a hash.

The per-utterance path (upstream ``model.generate``, one utterance at a time) is ~25x slower on the same GPU; the
batched path keeps Parakeet WER and durations of upstream (measured, see ``chatterbox_fast``'s docstring). Asserted
against the REAL job and the REAL worker source:

  * the job: default -> the worker argv has no ``--legacy_per_utterance``; ``legacy_per_utterance=True`` -> it has;
    both give the same Sisyphus hash, and that hash equals the one computed without the parameter at all (so no
    existing TTS job moved); a job object without the attribute (pickled before it existed) runs batched;
    ``language='de'`` passes ``--language de`` and moves the hash (Multilingual output), None keeps it;
  * the worker: ``--legacy_per_utterance`` is a ``store_true`` flag, and in ``main`` every call of
    ``process_dialogue`` (the ``model.generate`` path) sits under ``if args.legacy_per_utterance``; the batched
    path's code never calls ``generate`` / ``gen_conversation``. The source check is also run on a mutated copy
    that calls ``process_dialogue`` unguarded and must fail there.

The GPU half (the batched path really runs without ``generate``, and its speakers / gaps / timeline equal the
per-utterance path's) is ``check_chatterbox_fast_worker.py``.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_chatterbox_fast_path.py
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

from sisyphus import job as sis_job  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import tts  # noqa: E402

K = SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm"
FLAG = "--legacy_per_utterance"


def _is_legacy_test(node) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "legacy_per_utterance"


def worker_problems(src: str) -> list[str]:
    """What is wrong with a ``chatterbox_inference.py`` source (empty = fine)."""
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    bad = []
    main = funcs.get("main")
    if main is None:
        return ["no main()"]
    flag_ok = False
    for n in ast.walk(main):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            if isinstance(n.args[0], ast.Constant) and n.args[0].value == FLAG:
                kw = {k.arg: k.value for k in n.keywords}
                flag_ok = isinstance(kw.get("action"), ast.Constant) and kw["action"].value == "store_true"
    if not flag_ok:
        bad.append(f"{FLAG} is not a store_true flag of main()")

    def visit(node, guarded):
        if isinstance(node, ast.If) and _is_legacy_test(node.test):
            for c in node.body:
                visit(c, True)
            for c in node.orelse:
                visit(c, guarded)
            return
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "process_dialogue" and not guarded:
            bad.append(
                f"main() line {node.lineno}: process_dialogue (model.generate path) outside 'if args.legacy_per_utterance'"
            )
        for c in ast.iter_child_nodes(node):
            visit(c, guarded)

    visit(main, False)
    calls = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(main) if isinstance(n, ast.Call)
    }
    for need in ("process_dialogues_batched", "speed_s3gen"):
        if need not in calls:
            bad.append(f"main() never calls {need}")
    for name in ("process_dialogues_batched", "_plan_dialogue", "_render"):
        f = funcs.get(name)
        if f is None:
            bad.append(f"no {name}()")
            continue
        for n in ast.walk(f):
            if isinstance(n, ast.Call) and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) in (
                "generate", "gen_conversation", "process_dialogue"):  # fmt: skip
                bad.append(f"{name}() line {n.lineno} calls the per-utterance path")
    return bad


def fast_module_problems(src: str) -> list[str]:
    tree = ast.parse(src)
    return [f"chatterbox_fast.py line {n.lineno} calls generate" for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "generate"]  # fmt: skip


def job_argv(*, drop_attr: bool = False, **kw) -> tuple[list[str], str]:
    sis_job.created_jobs.clear()  # same hash = same cached object: build each variant fresh
    job = tts.ChatterboxInference(
        venv_python_path=tk.Path("/nonexistent/venv/bin/python"),
        in_hf=tk.Path("/nonexistent/in_hf"),
        speaker_dir=tk.Path("/nonexistent/voices"),
        speaker_alias={"user": "user_voices/rng_a"},
        ffmpeg_path=tk.Path("/nonexistent/ffmpeg"),
        shard=3,
        num_shards=8,
        **kw,
    )
    if drop_attr:  # a job pickled before the parameter existed
        del job.legacy_per_utterance
    seen = []

    def fake_fanout(python, script, args_for, **_kw):
        seen.append([str(a) for a in args_for(0, 1)])
        return 1

    orig = tts.run_worker_script_per_gpu
    tts.run_worker_script_per_gpu = fake_fanout
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            job.run()
    finally:
        os.chdir(cwd)
        tts.run_worker_script_per_gpu = orig
    assert len(seen) == 1
    return seen[0], job._sis_id()


def main():
    fails = []
    src = (K / "chatterbox_inference.py").read_text()
    for p in worker_problems(src):
        fails.append(f"worker: {p}")
    mutated = src.replace("            batch.append((i, dialogue, passthrough))",
                          "            process_dialogue(model, dialogue, device, args.speaker_directory, "
                          "silence_length_sampler, args.out_dir, i)\n            batch.append((i, dialogue, passthrough))", 1)  # fmt: skip
    if mutated == src:
        fails.append("could not build the mutated worker (the HF loop changed): update this guard")
    elif not worker_problems(mutated):
        fails.append("the source check passed a worker that calls process_dialogue unguarded")
    for name in ("chatterbox_fast.py", "chatterbox_fast_mtl.py"):
        for p in fast_module_problems((K / name).read_text()):
            fails.append(p.replace("chatterbox_fast.py", name))

    argv_default, id_default = job_argv()
    argv_legacy, id_legacy = job_argv(legacy_per_utterance=True)
    if FLAG in argv_default:
        fails.append(f"default ChatterboxInference passes {FLAG}: {argv_default}")
    if FLAG not in argv_legacy:
        fails.append(f"legacy_per_utterance=True does not pass {FLAG}: {argv_legacy}")
    try:
        argv_old, _ = job_argv(drop_attr=True)
        if FLAG in argv_old:
            fails.append(f"a job pickled before the parameter passes {FLAG}")
    except AttributeError as e:
        fails.append(f"a job pickled before the parameter existed crashes in run(): {e}")
    argv_de, id_de = job_argv(language="de")
    if argv_de[-2:] != ["--language", "de"] or "--language" in argv_default:
        fails.append(f"language='de' does not pass --language de (or the default passes one): {argv_de}")
    if id_de == id_default:
        fails.append("language='de' keeps the Turbo hash: a German corpus would reuse an English job")
    if id_default != id_legacy:
        fails.append(f"legacy_per_utterance moves the hash: {id_default} vs {id_legacy}")
    # the hash of a job built before the parameter existed: its parsed args lack the key
    args = dict(tts.ChatterboxInference.__init__.__kwdefaults__ or {})
    parsed = {"venv_python_path": tk.Path("/x"), "in_hf": tk.Path("/y"), "speaker_dir": tk.Path("/z")}
    with_key = tts.ChatterboxInference.hash({**{k: v for k, v in args.items()}, **parsed})
    without = tts.ChatterboxInference.hash(
        {**{k: v for k, v in args.items() if k not in ("legacy_per_utterance", "language")}, **parsed}
    )
    if with_key != without:
        fails.append("the hash function keeps legacy_per_utterance / language=None: every existing TTS hash would move")

    for f in fails:
        print("FAIL", f)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
