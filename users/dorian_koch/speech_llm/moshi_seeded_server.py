"""Launch the pip-installed ``moshi.server`` with a caller-chosen RNG seed.

``moshi.server`` hardcodes ``seed_all(42424242)`` (server.py) and exposes no ``--seed`` flag, so
every eval we run comes from one fixed seed -- we cannot estimate seed sensitivity from it. This thin
wrapper -- which we own -- forces the RNG seed from ``$MOSHI_SERVER_SEED`` instead.

CRITICAL (why we can't just monkeypatch ``seed_all`` after import): ``moshi/server.py`` calls
``main()`` at *module scope* -- there is no ``if __name__ == "__main__"`` guard (it ends with
``with torch.no_grad(): main()``). So ``import moshi.server`` *runs the whole server*, and ``main()``
calls ``seed_all(42424242)`` before the import statement ever returns -- any patch we apply afterwards
is dead code. (This is exactly the bug that made every pip-server eval identical across seeds.)

So we instead override the torch RNG primitives that ``seed_all`` delegates to
(``torch.manual_seed`` + the two CUDA variants) to swallow moshi's literal and use our seed --
installed *before* ``import moshi.server``. When the import-time ``main()`` calls
``seed_all(42424242)`` -> ``torch.manual_seed(42424242)`` -> our override -> ``manual_seed(_SEED)``.

Run it as a *script* with the pip moshi venv python and the normal server argv (main()'s argparse
reads ``sys.argv``, so ``--host`` / ``--port`` / ``--hf-repo`` / ``--lora-weight`` / ... all behave
exactly as before):
    python moshi_seeded_server.py --host 127.0.0.1 --port 9535 --hf-repo kyutai/moshika-rl-seamless

Shadowing: the Sisyphus worker exports the recipe tree on PYTHONPATH, and one copy of it
(``.../speech_llm/moshi.py``) shadows the pip-installed ``moshi`` *package*. Before importing, drop
every sys.path entry that carries a bare ``moshi.py`` so the real ``moshi/`` package wins.
"""

import os
import sys

sys.path = [p for p in sys.path if not (p and os.path.isfile(os.path.join(p, "moshi.py")))]

import torch  # noqa: E402  (must precede the moshi.server import that runs the server)

_SEED = int(os.environ.get("MOSHI_SERVER_SEED", "42424242"))

# Force our seed at the primitives seed_all() delegates to. Installed BEFORE importing moshi.server,
# because that import immediately runs main() -> seed_all(42424242) (see module docstring).
_orig_manual_seed = torch.manual_seed
_orig_cuda_manual_seed = torch.cuda.manual_seed
_orig_cuda_manual_seed_all = torch.cuda.manual_seed_all
torch.manual_seed = lambda _ignored=None: _orig_manual_seed(_SEED)
torch.cuda.manual_seed = lambda _ignored=None: _orig_cuda_manual_seed(_SEED)
torch.cuda.manual_seed_all = lambda _ignored=None: _orig_cuda_manual_seed_all(_SEED)

print(f"[moshi_seeded_server] forcing RNG seed = {_SEED}", file=sys.stderr, flush=True)

import moshi.server  # noqa: E402,F401  -- importing runs the server (bare main() at module scope)
