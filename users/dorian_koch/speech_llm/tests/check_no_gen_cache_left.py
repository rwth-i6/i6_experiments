"""Guard: every ``Dataset.from_generator`` passes a job-local ``cache_dir`` that the same file deletes again.

``from_generator`` writes the whole dataset as arrow under its cache dir, then ``save_to_disk`` writes the output copy.
Without ``cache_dir`` the cache lands in the shared ``$HF_HOME/datasets/generator`` (170 GB there on 2026-10-29); with a
job-local one that nobody deletes, every finished job keeps a second full copy of its corpus on the quota/inode-bound
volume (``PodcastCodesTrainData``, ``AttachPersonaPrompts`` & co. until 2026-10-29). Source scan over our packages:
each call passes ``cache_dir=`` and its file calls ``shutil.rmtree`` on it. Can fail: the pre-fix
``PodcastCodesTrainData`` call is flagged.

Run from the setup root: ``python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_no_gen_cache_left.py``
"""

import ast
import glob
import os

ROOTS = ["recipe/i6_experiments/users/dorian_koch/speech_llm", "recipe/speech_llm/full_duplex/moshi_family"]


def _calls(fn: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def offenders(src: str, name: str) -> list[str]:
    out = []
    tree = ast.parse(src)
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # helpers of this file that delete a tree (persona_prompts._drop_gen_cache)
    deleters = {f.name for f in funcs if "rmtree" in _calls(f)} | {"rmtree"}
    for fn in funcs:
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_generator"
            ):
                if not any(k.arg == "cache_dir" for k in node.keywords):
                    out.append(f"{name}:{node.lineno}: from_generator without cache_dir (lands in the shared HF home)")
                elif not _calls(fn) & deleters:
                    out.append(f"{name}:{node.lineno}: {fn.name} never deletes its from_generator cache_dir")
    return out


bad = []
for root in ROOTS:
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if "/tests/" in path:
            continue
        bad += offenders(open(path).read(), path)
old = 'def run(self):\n    ds = Dataset.from_generator(rows, features=f, cache_dir=os.path.abspath("hf_gen_cache"))\n'
assert offenders(old, "old"), "the pre-fix PodcastCodesTrainData call is not flagged"
print("  ok   can fail: a cache_dir nobody deletes is flagged")
elsewhere = old + "\n\ndef other():\n    shutil.rmtree('/tmp/scratch')\n"
assert offenders(elsewhere, "elsewhere"), "an rmtree in another function satisfies the check"
print("  ok   can fail: an rmtree in another function does not count")
for b in bad:
    print("  FAIL", b)
if bad:
    raise SystemExit(f"check_no_gen_cache_left: {len(bad)} from_generator call(s) leave their cache behind")
print("check_no_gen_cache_left: OK")
