"""Guard: the live-mic server starts every conversation from a fresh model (no state carried over).

The server keeps ONE model loaded and serves conversation after conversation. PersonaPlex's LM is
streaming-detached from its LMGen, so a reset that does not reach it silently continues the previous
conversation's KV cache and positions (the 2026-08-01 -> 09-22 reset bug, check_ppx_reset.py).
This drives the server's own session code, ``live_server.LiveCore.begin_sync`` (what the server runs on
its model thread since 2026-09-24) + ``step_codes``, on a
tiny random PersonaPlex LM (real LMModel + LMGen, greedy, float64):

  1. SECOND SESSION == FRESH. Session 1 (prompt A, 30 frames of codes), then session 2 (prompt B, other
     codes) on the same core must equal session 2 on a freshly built core, token for token.
  2. THE PROMPT IS IN. Two sessions with different role prompts and the same input differ.
  3. CAN FAIL. A core whose ``begin_sync`` skips ``lm_gen.reset_streaming()`` must violate check 1.
  4. SYNC == ASYNC. ``begin_sync`` (model thread, no disconnect check) gives the same session as the async
     ``begin`` it duplicates.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && PYTHONPATH=recipe/speech_llm/full_duplex:recipe \
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_live_server_reset.py'
"""

import asyncio
import contextlib
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex.live_server import LiveCore  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
KW = dict(loaders._lm_kwargs)
KW.update(
    dim=64,
    num_heads=4,
    num_layers=2,
    hidden_scale=2.0,
    text_card=100,
    context=256,
    depformer_dim=32,
    depformer_dim_feedforward=64,
    depformer_num_heads=2,
    depformer_num_layers=1,
    dep_q=16,
)
BASE_LM = LMModel(device="cpu", dtype=torch.float64, **KW).eval()
# At the default init the tiny model's output barely depends on its history (measured: a different
# role prompt, or no reset at all, changed no token), which would make checks 1-3 vacuous. Scaling the
# matrices x3 makes attention sharp enough that history matters (a different prompt changes ~96%
# of the generated tokens).
with torch.no_grad():
    for _name, _p in BASE_LM.named_parameters():
        if _p.dim() >= 2 and "emb" not in _name:
            _p.mul_(3.0)


class _NoMimi:
    _streaming_state = None

    def reset_streaming(self):
        pass

    def streaming_forever(self, _b):
        pass


class _Tok:
    """Stand-in tokenizer: characters -> ids inside the tiny model's text card."""

    def encode(self, text):
        return [4 + (ord(c) % 90) for c in text]


class _NoResetCore(LiveCore):
    """The failure mode: a session that starts without resetting the LM."""

    async def begin(self, voice, text_prompt, is_alive=None):
        from moshi_family.personaplex.components import encode_system_prompt

        self.lm_gen.text_prompt_tokens = encode_system_prompt(self.text_tokenizer, text_prompt)
        await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)

    def begin_sync(self, voice, text_prompt):
        from moshi_family.personaplex.components import encode_system_prompt

        self.lm_gen.text_prompt_tokens = encode_system_prompt(self.text_tokenizer, text_prompt)
        self.lm_gen.step_system_prompts(self.mimi)


def make_core(cls=LiveCore):
    g = LMGen(
        copy.deepcopy(BASE_LM),
        device="cpu",
        use_sampling=False,
        audio_silence_frame_cnt=6,
        sample_rate=24000,
        frame_rate=12.5,
    )
    g.streaming_forever(1)
    return cls(_NoMimi(), _NoMimi(), _Tok(), g, torch.device("cpu"))


def codes(seed, n=30):
    return torch.as_tensor(np.random.default_rng(seed).integers(0, 2048, size=(1, 8, n)))


def session(core, prompt, seed, use_async=False):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        if use_async:
            asyncio.run(core.begin(None, prompt))
        else:
            core.begin_sync(None, prompt)
        out = core.step_codes(codes(seed))
    return torch.cat(out, dim=-1).numpy()


def second_equals_fresh(cls) -> bool:
    fresh = session(make_core(cls), "You are a pirate.", 2)
    core = make_core(cls)
    session(core, "You enjoy having a good conversation.", 1)
    second = session(core, "You are a pirate.", 2)
    return fresh.shape == second.shape and np.array_equal(fresh, second)


def main():
    assert second_equals_fresh(LiveCore), "a second session differs from a fresh core: state leaked across sessions"
    print("[ok] 1 second session on the same core == the same session on a fresh core")

    a = session(make_core(), "You are a pirate.", 3)
    b = session(make_core(), "You are a nurse on a night shift.", 3)
    # rows 0..8 = the model's own text + audio (rows 9.. echo the user codes it was given)
    assert not np.array_equal(a[:, :9], b[:, :9]), "the role prompt had no effect on the output"
    print("[ok] 2 different role prompts give different outputs (the prompt reaches the model)")

    assert not second_equals_fresh(_NoResetCore), "check 1 cannot fail: it passes without the reset"
    print("[ok] 3 without lm_gen.reset_streaming() check 1 fails (the guard can fail)")

    core = make_core()
    session(core, "You enjoy having a good conversation.", 1)
    s = session(core, "You are a pirate.", 2)
    core = make_core()
    session(core, "You enjoy having a good conversation.", 1, use_async=True)
    a = session(core, "You are a pirate.", 2, use_async=True)
    assert np.array_equal(s, a), "begin_sync and begin give different sessions"
    print("[ok] 4 begin_sync (the server's model-thread path) == the async begin, token for token")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
