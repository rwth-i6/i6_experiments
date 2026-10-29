"""Guard: streaming inference never shrinks n silently (audit I6JOBS-02).

``inference_harness.stream_inference`` used to print three RETRY lines for a clip whose server was dead and move on,
and a clip cut mid-write kept its partial wav, which a resumed (FDB) run then skipped as done.

  [1] a client that always fails makes stream_inference raise (above max_failed_frac) and leaves no output file;
  [2] a client killed mid-write leaves no ``out`` file (it writes a ``.partial`` name, renamed only on success);
  [3] a client that succeeds writes ``out`` and no ``.partial`` file remains.
CAN FAIL: with max_failed_frac=1.0 the dead-server run returns normally (shows [1] is what raises).

Run from the setup root, no GPU, ~1 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_stream_inference_failures.py
"""

import contextlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")

from i6_experiments.users.dorian_koch.speech_llm.inference_harness import StreamOptions, stream_inference  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("[ok] " if cond else "[FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


@contextlib.contextmanager
def server(**_):
    yield "handle"


class Dead:
    def __init__(self, url, inp, out, **_):
        self.out = out

    def run(self):
        raise ConnectionError("server is dead")


class KilledMidWrite(Dead):
    def run(self):
        Path(self.out).write_bytes(b"RIFF partial")
        raise TimeoutError("killed mid-write")


class Good(Dead):
    def run(self):
        Path(self.out).write_bytes(b"RIFF ok")


def run(client, opts, n=3):
    tmp = Path(tempfile.mkdtemp())
    items = [(tmp / f"in{i}.wav", tmp / "out" / f"{i}.wav") for i in range(n)]
    try:
        stream_inference(
            server=server, file_client=client, ws_url=lambda h: "ws://x", items=items, server_kwargs={}, opts=opts
        )
        raised = False
    except RuntimeError:
        raised = True
    return raised, sorted(p.name for p in (tmp / "out").glob("*"))


raised, files = run(Dead, StreamOptions(progress_every=0))
check(raised and files == [], f"[1] a dead server raises and writes nothing (raised={raised}, files={files})")
raised, files = run(KilledMidWrite, StreamOptions(progress_every=0, max_failed_frac=1.0))
check(not raised and files == [], f"[2] a clip killed mid-write leaves no output (files={files})")
raised, files = run(Good, StreamOptions(progress_every=0))
check(not raised and files == ["0.wav", "1.wav", "2.wav"], f"[3] successful clips land under their own names ({files})")
raised, _ = run(Dead, StreamOptions(progress_every=0, max_failed_frac=1.0))
check(not raised, "CAN FAIL: with the failure budget at 100% the dead-server run returns normally")

print("FAILED" if FAILS else "ALL CHECKS PASSED")
sys.exit(1 if FAILS else 0)
