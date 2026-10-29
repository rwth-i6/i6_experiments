"""Guard: a venv whose output is data or a metric is pinned, so a rebuild under its kept hash changes nothing.

``CreateVenv`` hashes ``hash_overwrite``, not its package list, and runs one ``pip install`` per package group. A
rebuild (``hpc-rerun --finished`` after adding a package, a repair after inode cleanup) re-resolves every unpinned
package to its latest release under the same job id: benchmark ASR, scorer and training-text outputs would then change
with no marker (``whisper_venv`` was unpinned until 2026-10-29). Checks, for each venv in ``PINNED``: every package
group passes a constraints file (``-c``, which must exist) or pins every requirement with ``==``. Can fail: the
pre-2026-10-29 ``whisper_venv`` package list is flagged.

Run from the setup root: ``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_venv_pins.py``
"""

import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

from speech_llm.full_duplex.sis_recipe.doriank import venvs  # noqa: E402

#: venvs whose outputs are data or metrics (the "pin venvs whose output is data" rule)
PINNED = (
    "whisper_venv",
    "voicebench_venv",
    "mtr_eval_venv",
    "moshi_family_venv",
    "faster_whisper_addon_venv",
    "duplexchat_venv",
    "npp_venv",
    "chatterbox_venv",
)
FLAGS_WITH_VALUE = {"--index-url", "--extra-index-url", "-f", "--find-links", "-c", "--constraint"}


def unpinned(packages) -> list[str]:
    out = []
    for group in packages:
        if "-c" in group:
            path = group[group.index("-c") + 1]
            if not os.path.exists(str(path)):
                out.append(f"constraints file missing: {path}")
            continue
        skip = False
        for tok in group:
            if skip:
                skip = False
                continue
            if tok in FLAGS_WITH_VALUE:
                skip = True
                continue
            if tok.startswith("-"):
                continue
            if "==" not in tok and not tok.startswith(("git+", "http", "/")):
                out.append(tok)
    return out


captured = {}
real = venvs._register_venv
venvs._register_venv = lambda name, *, packages, hash_overwrite: captured.setdefault(name, packages)
try:
    for fn in PINNED:
        f = getattr(venvs, fn)
        getattr(f, "cache_clear", lambda: None)()
        f()
finally:
    venvs._register_venv = real

bad = {}
for name, pk in captured.items():
    u = unpinned(pk)
    print(f"  {'ok  ' if not u else 'FAIL'} {name}{': ' + str(u[:6]) if u else ''}")
    if u:
        bad[name] = u
old_whisper = [
    ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"],
    ["transformers", "accelerate", "soundfile", "datasets", "faster-whisper"],
]
assert unpinned(old_whisper), "the pre-2026-10-29 whisper_venv list is not flagged"
print("  ok   can fail: the pre-2026-10-29 whisper_venv list is flagged")
if bad:
    raise SystemExit(f"check_venv_pins: {len(bad)} data/metric venv(s) with unpinned packages: {sorted(bad)}")
print("check_venv_pins: OK")
