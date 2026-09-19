"""Guard: the whole-episode arrow schema, and that adding it did not disturb the dialogue one.

Why this is worth a guard. A missing column does not raise -- it writes a part that is structurally
valid, loads fine, indexes fine, and is silently missing the data the entire whole-episode design
exists to capture. `speaker_embeddings` in particular is only obtainable during the one 173 GPU-h
pass we are paying for; noticing later that the column was never written means re-running all of it.

The other half is regression: `podcast_duplex_main.py` is the TESTED path and the reference the new
one is validated against, so its schema must be byte-for-byte what it was before the episode schema
was added. Adding a branch to a shared function is exactly how that gets broken quietly.

Runs on the login node: `datasets` and `moshi_family.podcast_ingest_main` are stubbed, and the real
`codes_features` key list is recovered from its SOURCE by AST so the base schema is still pinned to
the real thing rather than to the stub.

Run: CUDA_HOME=/usr .venv/bin/python recipe/.../tests/check_episode_schema.py
"""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
LIB = SETUP / "recipe/speech_llm/full_duplex/moshi_family"
os.environ.setdefault("CUDA_HOME", "/usr")

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail if not cond else ''}")
    if not cond:
        fails.append(name)


# ---- recover the REAL base schema's keys from source, without importing torch -------------------
def codes_features_keys() -> list[str]:
    tree = ast.parse((LIB / "podcast_ingest_main.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "codes_features")
    keys: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.append(k.value)
    return keys


BASE = codes_features_keys()
print("[1] the base schema, read from codes_features' own source")
check("codes_features declares a non-trivial schema", len(BASE) >= 15, f"{len(BASE)} keys")
for must in ("item_id", "episode_id", "codes_a", "codes_b", "spans_json", "duration_sec"):
    check(f"base has {must}", must in BASE)


# ---- stubs --------------------------------------------------------------------------------------
class _Value:
    def __init__(self, dtype):
        self.dtype = dtype

    def __eq__(self, o):
        return isinstance(o, _Value) and o.dtype == self.dtype

    def __repr__(self):
        return f"Value({self.dtype!r})"


class _Sequence:
    def __init__(self, feature):
        self.feature = feature

    def __eq__(self, o):
        return isinstance(o, _Sequence) and o.feature == self.feature

    def __repr__(self):
        return f"Sequence({self.feature!r})"


ds = types.ModuleType("datasets")
ds.Value, ds.Sequence, ds.Dataset = _Value, _Sequence, object
sys.modules["datasets"] = ds

pkg = types.ModuleType("moshi_family")
pkg.__path__ = [str(LIB)]
sys.modules["moshi_family"] = pkg
pim = types.ModuleType("moshi_family.podcast_ingest_main")
pim.codes_features = lambda: {k: _Value("string") for k in BASE}
pim.encode_pair = lambda *a, **k: None
pim.validate_codes = lambda *a, **k: None
sys.modules["moshi_family.podcast_ingest_main"] = pim

spec = importlib.util.spec_from_file_location("pme", LIB / "podcast_mimi_encode.py")
pme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pme)

print("[2] the DIALOGUE schema must be exactly what it was before the episode branch existed")
d = pme.part_features("dialogue")
expected_dialogue = set(BASE) - {"spans_json"} | {"xcorr", "turns_json", "n_speakers"}
check(
    "dialogue column set is unchanged",
    set(d) == expected_dialogue,
    f"extra {set(d) - expected_dialogue}, missing {expected_dialogue - set(d)}",
)
check("spans_json is dropped (the row IS the span)", "spans_json" not in d)
check("xcorr is float32", d["xcorr"] == _Value("float32"))
check("n_speakers is int32", d["n_speakers"] == _Value("int32"))

print("[3] the EPISODE schema carries everything that is otherwise unrecoverable")
e = pme.part_features("episode")
required = {
    "speaker_embeddings": _Sequence(_Value("float32")),
    "speaker_embedding_dim": _Value("int32"),
    "speaker_labels_json": _Value("string"),
    "diar_turns_json": _Value("string"),
    "diar_exclusive_json": _Value("string"),
    "overlap_sec": _Value("float32"),
    "chunk_meta_json": _Value("string"),
    "provenance_json": _Value("string"),
    "episode_duration_sec": _Value("float32"),
    "expected_duration_sec": _Value("float32"),
    "drift_sec": _Value("float32"),
    "download_bytes": _Value("int64"),
    "peak": _Value("float32"),
    "n_speakers_total": _Value("int32"),
}
for k, v in required.items():
    check(f"episode has {k} :: {v}", k in e and e[k] == v, f"got {e.get(k)!r}")
check(
    "episode keeps every base column except spans_json",
    set(BASE) - {"spans_json"} <= set(e),
    f"missing {set(BASE) - {'spans_json'} - set(e)}",
)
check(
    "episode does NOT carry the dialogue-only columns",
    "turns_json" not in e and "n_speakers" not in e,
    f"turns_json={'turns_json' in e} n_speakers={'n_speakers' in e}",
)

print("[4] non-vacuity and refusals")
check("the two schemas genuinely differ", set(d) != set(e))
check("...by the fields that matter", "speaker_embeddings" in e and "speaker_embeddings" not in d)
try:
    pme.part_features("nonsense")
    # part_features has no explicit whitelist: anything that is not "dialogue" takes the episode
    # branch. That is fine ONLY because argparse constrains the caller, which is asserted next.
    check("an unknown row_mode falls through to episode (argparse is the gate)", True)
except Exception as e_:  # noqa: BLE001
    check("an unknown row_mode falls through to episode", False, repr(e_)[:100])

src = (LIB / "podcast_mimi_encode.py").read_text()
tree = ast.parse(src)
choices = None
for node in ast.walk(tree):
    if (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and node.args
        and getattr(node.args[0], "value", "") == "--row_mode"
    ):
        for kw in node.keywords:
            if kw.arg == "choices":
                choices = {c.value for c in kw.value.elts}
check("--row_mode is constrained by argparse choices", choices == {"dialogue", "episode"}, f"got {choices}")
check("--row_mode defaults to the TESTED path", 'default="dialogue"' in src or "default='dialogue'" in src)

print("[5] the worker writes every column the episode schema declares")
# The schema and its producer are in different files and different VENVS, so nothing but this
# connects them. A column declared here and never set by the worker is a KeyError at the end of a
# multi-hour shard, after the GPU work is already spent.
worker = (LIB / "podcast_episode_main.py").read_text()
jobsrc = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/podcast_ingest.py").read_text()
missing = [k for k in e if k not in BASE and f'"{k}"' not in worker and f'"{k}"' not in src]
check("no episode column is unset by both worker and encoder", not missing, f"missing {missing}")

print("[6] the worker must SEND --row_mode episode, not rely on the default")
# The bug this catches was live: the worker invoked the encoder without --row_mode, whose default is
# "dialogue". Every episode-only column would simply not have been written -- a structurally valid
# part, indexable, loadable, and empty of the one thing the 173 GPU-h pass exists to capture. It
# raises nothing, so only an AST check on the call site can see it.
wtree = ast.parse(worker)
sent = False
for node in ast.walk(wtree):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "run":
        for arg in node.args:
            if not isinstance(arg, ast.List):
                continue
            vals = [x.value for x in arg.elts if isinstance(x, ast.Constant)]
            if "--row_mode" in vals and "episode" in vals:
                sent = True
check("the encoder subprocess passes --row_mode episode", sent)

print("[7] the DuplexChat-venv worker must not import the moshi_family PACKAGE")
# The bug: `from moshi_family.perm_repair import repair_permutation`. The module is pure numpy, but
# importing it as a package member executes moshi_family/__init__.py, which pulls in the moshi model
# stack. That venv is torch 2.11.0+cu128 for DialogueSidon and has no sentencepiece, so this died
# with ModuleNotFoundError AFTER the GPU was allocated. Caught by the smoke; guarded here.
wtree = ast.parse(worker)
pkg_imports = [
    n
    for n in ast.walk(wtree)
    if (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("moshi_family"))
    or (isinstance(n, ast.Import) and any(a.name.startswith("moshi_family") for a in n.names))
]
check(
    "no moshi_family package import anywhere in the worker", not pkg_imports, f"lines {[n.lineno for n in pkg_imports]}"
)
check("perm_repair is loaded by file path instead", "spec_from_file_location" in worker and "perm_repair_py" in worker)
check("--perm_repair_py is a REQUIRED arg", '"--perm_repair_py"' in worker and "required=True" in worker)
check("the job passes --perm_repair_py", '"--perm_repair_py"' in jobsrc)
# Non-vacuity: the isolation only matters if the package init really is heavy. If moshi_family ever
# becomes import-light this guard should be reconsidered rather than silently kept.
init = (LIB / "__init__.py").read_text()
check(
    "moshi_family/__init__ really does pull in the model stack (so the isolation is needed)",
    "from . import models" in init or "import models" in init,
    init[:120],
)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
