"""Guard: a Job's function-local imports still resolve inside the Sisyphus worker, i.e. after it chdirs into the job dir.

The worker finds recipe packages via ``sisyphus.loader.RecipeFinder``, which resolves the RELATIVE ``gs.IMPORT_PATHS``
(``"recipe"``) with ``os.path.abspath`` at import time. Before the chdir that is the setup's ``recipe/``; after it, it
is ``<job dir>/recipe``, so a top-level recipe package (``i6_core``, ``speech_llm``, ...) that the job's module did not
already import at unpickle time is gone. Subpackages of an already imported package still load (their parent's
``__path__`` is absolute), which is why this only bites lazily. 2026-10-23: ``AudioJudgeAgreement.run`` did
``from . import knowledge_benchmark`` -> tts -> jobs/__init__ -> ``i6_core`` and died 10 s in with ``No module named
'i6_core'`` after its 3-hour prep had finished.

For every module under ``speech_llm/`` and ``jobs/`` that defines a Job, this imports it the way the worker does (only
``recipe/sisyphus`` on sys.path, RecipeFinder for the rest), chdirs to an empty temp dir, then runs each first-party
import inside the worker-side methods (the ``Task`` methods and the ``self.<m>()`` they call; manager-side code such as
``__init__`` or module helpers is not checked). One fresh interpreter per module, so nothing is preloaded by another module.

Run from the setup root, no GPU needed (~2 min):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_worker_lazy_imports.py [module ...]
"""

import ast
import concurrent.futures
import json
import os
import subprocess
import sys

SETUP = os.getcwd()
RECIPE = os.path.join(SETUP, "recipe")
PKG = "i6_experiments.users.dorian_koch"
DIRS = ["speech_llm", "jobs"]

CHILD = r"""
import importlib, json, os, sys, tempfile
sys.path.insert(0, "recipe/sisyphus")
import sisyphus  # installs RecipeFinder, as in the worker
mod, lazy = sys.argv[1], json.loads(sys.argv[2])
importlib.import_module(mod)
os.chdir(tempfile.mkdtemp())
bad = []
for name, package in lazy:
    try:
        importlib.import_module(name, package)
    except ImportError as e:
        bad.append(f"{name}: {type(e).__name__}: {e}")
print(json.dumps(bad))
"""


def recipe_tops():
    return {n for n in os.listdir(RECIPE) if os.path.isdir(os.path.join(RECIPE, n)) and not n.startswith((".", "_"))}


def task_methods(cls):
    """Methods the worker runs for a Job class: every ``Task("name")`` / ``Task(..., resume="name")`` in it, plus the
    ``self.<m>()`` methods those call, transitively."""
    meths = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    todo = []
    for n in ast.walk(cls):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "Task":
            consts = [a for a in n.args[:1] if isinstance(a, ast.Constant)]
            consts += [k.value for k in n.keywords if k.arg == "resume" and isinstance(k.value, ast.Constant)]
            todo += [c.value for c in consts if isinstance(c.value, str)]
    seen = set()
    while todo:
        m = todo.pop()
        if m in seen or m not in meths:
            continue
        seen.add(m)
        for n in ast.walk(meths[m]):
            f = getattr(n, "func", None)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                todo.append(f.attr)
    return [meths[m] for m in sorted(seen)]


def lazy_imports(path, mod, tops):
    """(name, package) of every first-party import inside the worker-side methods of the module's Job classes."""
    tree = ast.parse(open(path).read())
    package = mod if path.endswith("__init__.py") else mod.rsplit(".", 1)[0]
    jobs = [n for n in tree.body if isinstance(n, ast.ClassDef) and any("Job" in ast.unparse(b) for b in n.bases)]
    out = set()
    for fn in [m for c in jobs for m in task_methods(c)]:
        if "sys.path.insert" in ast.unparse(fn) or "sys.path.append" in ast.unparse(fn):
            continue  # pins recipe/ itself before importing (moshi.py, speech_inference.py)
        for n in ast.walk(fn):
            if isinstance(n, ast.ImportFrom):
                if n.level:
                    base = "." * n.level + (n.module or "")
                    names = [base] if n.module else [base + a.name for a in n.names]
                    out.update((x, package) for x in names)
                elif n.module.split(".")[0] in tops:
                    out.add((n.module, None))
            elif isinstance(n, ast.Import):
                out.update((a.name, None) for a in n.names if a.name.split(".")[0] in tops)
    return sorted(out, key=str)


def modules(only):
    for d in DIRS:
        root = os.path.join(RECIPE, *PKG.split("."), d)
        for dp, dns, fns in os.walk(root):
            dns[:] = [x for x in dns if x not in ("tests", "__pycache__")]
            for f in sorted(fns):
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    rel = os.path.relpath(p, RECIPE)[:-3].replace(os.sep, ".")
                    mod = rel[: -len(".__init__")] if rel.endswith(".__init__") else rel
                    if not only or any(o in mod for o in only):
                        yield p, mod


def check(mod, lazy):
    env = {**os.environ, "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr"), "PYTHONPATH": ""}
    r = subprocess.run([sys.executable, "-c", CHILD, mod, json.dumps(lazy)], capture_output=True, text=True, env=env)
    if r.returncode:
        return [
            f"(module import itself failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})"
        ]
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    tops = recipe_tops()
    todo = []
    for p, mod in modules(sys.argv[1:]):
        lazy = lazy_imports(p, mod, tops)
        if lazy:
            todo.append((mod, lazy))
    assert todo, "no Job with worker-side first-party imports found -- wrong cwd?"
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        res = dict(zip([m for m, _ in todo], ex.map(lambda t: check(*t), todo)))
    bad = {m: b for m, b in res.items() if b}
    for m, b in sorted(bad.items()):
        print(f"FAIL {m}:\n  " + "\n  ".join(b))
    assert not bad, (
        f"{len(bad)} Job module(s) import a recipe package lazily that the worker cannot find after its chdir: "
        "import it at module level"
    )
    print(f"check_worker_lazy_imports: OK ({len(todo)} Job modules, their lazy imports resolve after the worker chdir)")


if __name__ == "__main__":
    main()
