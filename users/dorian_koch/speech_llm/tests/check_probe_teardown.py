"""Guard: the in-loop knowledge probe tears down the streaming state on the shared ``mf.lm`` / ``mf.mimi`` even when
building its ``InferenceState`` raises after entering streaming (the next training forward would run on it).

Drives the real ``run_knowledge_probe`` with a fake ``mf`` and an ``InferenceState`` that enters streaming, then
raises. Can fail: teardown only through a probe model that was never built (the old ``if pm is not None``).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_probe_teardown.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402

from moshi_family import knowledge_probe  # noqa: E402


class _Streamable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(1, 1)
        self.inner._streaming_state = None
        self._streaming_state = None
        self.detached = False
        self.sample_rate = 24000

    def set_streaming_detached(self, v):
        self.detached = v


class _Boom(Exception):
    pass


def _dirty_then_raise(ckpt, mimi, tok, lm, *a, **k):
    for root in (lm, mimi):
        for m in root.modules():
            m._streaming_state = "live"
    lm.set_streaming_detached(True)
    raise _Boom


def main():
    mf = SimpleNamespace(lm=_Streamable(), mimi=_Streamable(), text_tokenizer=None,
                         checkpoint=SimpleNamespace(lm_gen_config={}))  # fmt: skip
    old = knowledge_probe.InferenceState
    knowledge_probe.InferenceState = _dirty_then_raise
    try:
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "probe.jsonl"
            probe.write_text(json.dumps({"wav": "/nonexistent.wav", "answer": "x"}) + "\n")
            try:
                knowledge_probe.run_knowledge_probe(mf, str(probe), device="cpu")
                raise AssertionError("the probe did not raise")
            except _Boom:
                pass
    finally:
        knowledge_probe.InferenceState = old
    left = [n for root in (mf.lm, mf.mimi) for n, m in root.named_modules() if m._streaming_state is not None]
    assert not left, f"CAN FAIL: streaming state left on {left}"
    assert not mf.lm.detached, "the LM stayed detached from its streaming tree"
    print("check_probe_teardown OK")


if __name__ == "__main__":
    main()
