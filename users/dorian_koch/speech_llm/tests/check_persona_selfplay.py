"""Guard: PersonaPlex self-play (``moshi_family.personaplex.selfplay``) -- prompt parity and the cross-feed.

  1. PRIME PARITY. ``prime_from_codes`` drives the real ``LMGen.step_system_prompts`` with the voice
     given as stored codes; the frames it feeds must equal ``train_data.system_prompt_prefix`` (the
     prefix every PersonaPlex arm was trained on). A voice with its frames reversed must NOT match,
     so the comparison can fail.
  2. CROSS-FEED. With two recording stand-in models (real ``converse``): each side's input at frame t
     is the OTHER side's output of frame t-1, silence codes until the other has produced output, and
     a side's own output is never fed back to itself. Outputs are collected per side in order.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_persona_selfplay.py'
"""

import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex.models.lm import LMGen, SILENCE_TOKENS  # noqa: E402
from moshi_family.personaplex.selfplay import converse, prime_from_codes  # noqa: E402
from moshi_family.personaplex.train_data import system_prompt_prefix  # noqa: E402

N_Q, SIL = 16, 6
rng = np.random.default_rng(0)
voice = rng.integers(0, 2048, size=(8, 50))
text_ids = [11, 222, 3333, 44, 5]


def rec_lmgen():
    class Rec:
        pass

    for name in (
        "step_system_prompts",
        "_step_voice_prompt",
        "_step_voice_prompt_core",
        "_step_voice_prompt_frame",
        "_step_audio_silence",
        "_step_audio_silence_core",
        "_step_text_prompt",
        "_step_text_prompt_core",
        "_encode_zero_frame",
        "_encode_sine_frame",
    ):
        setattr(Rec, name, getattr(LMGen, name))
    r = Rec()
    r.lm_model = types.SimpleNamespace(device="cpu")
    r.zero_text_code = 3
    r.audio_silence_frame_cnt = SIL
    r.save_voice_prompt_embeddings = False
    r.frames = []

    def step(moshi_tokens=None, text_token=None, input_tokens=None, return_embeddings=False):
        t = int(text_token) if not torch.is_tensor(text_token) else int(text_token.reshape(-1)[0])
        r.frames.append(np.concatenate([[t], moshi_tokens.reshape(-1).numpy(), input_tokens.reshape(-1).numpy()]))

    r.step = step
    return r


# ---- 1 ----
want = system_prompt_prefix(text_ids, voice, n_q=N_Q, silence_frames=SIL).numpy()
r = rec_lmgen()
with contextlib.redirect_stdout(open(os.devnull, "w")):
    prime_from_codes(r, None, voice, text_ids)
got = np.stack(r.frames, axis=1)
assert got.shape == want.shape and np.array_equal(got, want), "self-play prime != training prefix"
r2 = rec_lmgen()
with contextlib.redirect_stdout(open(os.devnull, "w")):
    prime_from_codes(r2, None, voice[:, ::-1].copy(), text_ids)
assert not np.array_equal(np.stack(r2.frames, axis=1), want), "guard is vacuous: a reversed voice matched"
print(f"[1] prime_from_codes == system_prompt_prefix ({want.shape[1]} frames); a reversed voice is rejected")

# ---- 2 ----
DELAY = 2


class FakeGen:
    """Stands in for LMGen after priming: records its inputs, emits (side, frame) codes after DELAY."""

    def __init__(self, side):
        self.side, self.t, self.inputs = side, 0, []
        self.lm_model = types.SimpleNamespace(device="cpu")

    def _encode_zero_frame(self):
        return torch.as_tensor(SILENCE_TOKENS).view(1, 8, 1)

    def step(self, input_tokens):
        self.inputs.append(input_tokens.reshape(-1).clone())
        t, self.t = self.t, self.t + 1
        if t < DELAY:
            return None
        return torch.full((1, 9, 1), 1000 * (self.side + 1) + t, dtype=torch.long)

    def reset_streaming(self):
        pass


def fake_model(side):
    g = FakeGen(side)
    mimi = types.SimpleNamespace(reset_streaming=lambda: None)
    return types.SimpleNamespace(lm_gen=g, mimi=mimi, other_mimi=mimi)


import moshi_family.personaplex.selfplay as sp  # noqa: E402

sp.prime_from_codes = lambda *a, **k: None  # priming is covered by [1]
A, B = fake_model(0), fake_model(1)
T = 12
(ca, ta), (cb, tb) = converse(A, B, voice_a=voice, voice_b=voice, text_a=text_ids, text_b=text_ids, n_frames=T)
sil = torch.as_tensor(SILENCE_TOKENS)
for me, other, base_other in ((A, B, 2000), (B, A, 1000)):
    ins = me.lm_gen.inputs
    assert len(ins) == T
    for t in range(T):
        prev_out = t - 1  # the other side's output of the previous frame
        if prev_out < DELAY:
            assert torch.equal(ins[t], sil), f"frame {t}: expected silence before the other side speaks"
        else:
            assert torch.equal(ins[t], torch.full((8,), base_other + prev_out)), (
                f"frame {t}: input is not the other side's frame {prev_out}"
            )
assert ca.shape == (8, T - DELAY) and list(ca[0]) == [1000 + t for t in range(DELAY, T)]
assert list(tb) == [2000 + t for t in range(DELAY, T)]
print(
    f"[2] cross-feed: each side hears the other's previous frame (silence for the first {DELAY + 1}), outputs in order"
)
print("OK")
