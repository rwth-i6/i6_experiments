"""Guard: the PersonaPlex training loader is built in ONE place, ``finetune_launcher.make_train_loader``.

Hand copies of the launcher's cfg -> loader block in analysis scripts had drifted (audit SCRIPTS-03): a replay that
drops ``aux_data`` / row weights / the word mask draws a different item population than the run trained on, and its
listening page shows a stream the run never saw. ``train_loader_args(cfg)`` reads every loader knob (main() calls it;
check_launcher_config_reads follows it), ``make_train_loader`` builds the loader, ``make_train_tokenizer`` the tokenizer.

  [1] the launcher's main() builds its loader and tokenizer only through them (no direct loader-builder call);
  [2] no analysis script imports the PersonaPlex loader builders (``build_data_loader`` / ``build_knowledge_mix_loader``
      from ``moshi_family.personaplex.train_data``); CAN FAIL: the pre-refactor dryrun script is flagged.

Run from the setup root: python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_shared_train_loader.py
"""

import ast
import glob
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
LAUNCHER = SETUP / "recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py"
BUILDERS = {"build_data_loader", "build_knowledge_mix_loader"}  # a bare tokenizer (eval probes) is fine
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        FAILURES.append(what)


def called_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None)
    return out


def direct_builder_use(src: str) -> set:
    """PersonaPlex loader builders a script imports from train_data or calls as ``td.<builder>``."""
    tree = ast.parse(src)
    hits = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("personaplex.train_data"):
            hits |= {a.name for a in n.names} & BUILDERS
        if isinstance(n, ast.Attribute) and n.attr in BUILDERS and isinstance(n.value, ast.Name) and n.value.id == "td":
            hits.add(n.attr)
    return hits


print("[1] the launcher builds through the shared functions")
tree = ast.parse(LAUNCHER.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
calls = called_names(main)
check({"train_loader_args", "make_train_loader", "make_train_tokenizer"} <= calls, "main() calls all three")
check(not (calls & BUILDERS), f"main() calls no builder directly ({sorted(calls & BUILDERS)})")

print("[2] analysis scripts use them too")
MUTANT = (
    "from moshi_family.personaplex.train_data import PersonaPlexDataConfig, PersonaPlexTokenizer, build_data_loader\n"
)
check(direct_builder_use(MUTANT) == {"build_data_loader"}, "CAN FAIL: the old dryrun import")
scripts = sorted(glob.glob(str(SETUP / "analysis" / "*.py")))
bad = {os.path.basename(p): sorted(h) for p in scripts if (h := direct_builder_use(open(p).read()))}
check(len(scripts) > 20 and not bad, f"{len(scripts)} analysis scripts, none builds the loader itself {bad}")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
