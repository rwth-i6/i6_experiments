"""Guard: one copy of each text scorer / normaliser, ``recipe/speech_llm/full_duplex/text_scoring.py``.

``alias_match`` existed twice (the in-training probe and the quick eval, kept equal by a test) and Whisper's text
normaliser three times (``moshi_family.text_norm``, ``corpus_wer``, ``analysis/evalstats``). Now every environment
imports the one module: job venvs as ``text_scoring``, the manager as ``speech_llm.full_duplex.text_scoring``,
``analysis/`` through ``evalstats``.

  [1] no other definition of ``alias_match`` / ``normalize_answer`` and no other ``EnglishTextNormalizer(``
      construction in the recipe trees or ``analysis/``; CAN FAIL: the pre-refactor copy is flagged;
  [2] the module imports with the login node's system python3 (py3.9, stdlib only) and pulls in no third-party
      package at import time.

Run from the setup root with the login node's python3:
    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_text_scoring.py
"""

import glob
import os
import re
import subprocess
import sys

SETUP = os.getcwd()
FD = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex")
HOME = os.path.realpath(os.path.join(FD, "text_scoring.py"))
DEF = re.compile(r"^\s*def (alias_match|normalize_answer)\(|EnglishTextNormalizer\(", re.M)
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        FAILURES.append(what)


def copies(text):
    return sorted(set(m.group(1) or "EnglishTextNormalizer" for m in DEF.finditer(text)))


print("[1] one definition of each")
MUTANT = "def alias_match(prediction: str, answer: str, aliases) -> tuple[int, int]:\n    return 0, 1\n"
check(copies(MUTANT) == ["alias_match"], "CAN FAIL: a pasted alias_match is flagged")
roots = [FD, os.path.join(SETUP, "recipe", "i6_experiments", "users", "dorian_koch"), os.path.join(SETUP, "analysis")]
found = {}
n = 0
for root in roots:
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if "/tests/" in p or "/old/" in p or os.path.realpath(p) == HOME:
            continue
        n += 1
        c = copies(open(p, errors="replace").read())
        if c:
            found[os.path.relpath(p, SETUP)] = c
check(n > 500 and not found, f"{n} files scanned, no other copy {found}")
check(
    copies(open(HOME).read()) == ["EnglishTextNormalizer", "alias_match", "normalize_answer"], "text_scoring has them"
)

print("[2] stdlib-only import, py3.9")
code = (
    "import sys; sys.path.insert(0, %r); before = set(sys.modules); import text_scoring; "
    "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
    "bad = new & {'transformers', 'huggingface_hub', 'torch', 'numpy', 'moshi_family', 'sisyphus', 'i6_experiments'}; "
    "print(sys.version_info[:2], sorted(bad)); sys.exit(1 if bad else 0)"
) % FD
r = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True)
check(
    r.returncode == 0, f"system python3 imports it without third-party packages: {r.stdout.strip()} {r.stderr[-200:]}"
)

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
