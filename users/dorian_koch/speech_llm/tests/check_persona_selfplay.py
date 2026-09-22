"""Guard: PersonaPlex self-play (``moshi_family.personaplex.selfplay``) -- prompt parity and the cross-feed.

  1. PRIME PARITY. ``prime_from_codes`` drives the real ``LMGen.step_system_prompts`` with the voice
     given as stored codes; the frames it feeds must equal ``train_data.system_prompt_prefix`` (the
     prefix every PersonaPlex arm was trained on). A voice with its frames reversed must NOT match,
     so the comparison can fail.
  2. CROSS-FEED. With two recording stand-in models (real ``converse``): each side's input at frame t
     is the OTHER side's output of frame t-1, silence codes until the other has produced output, and
     a side's own output is never fed back to itself. Outputs are collected per side in order.
  3. BATCHED == B=1. A tiny random PersonaPlex LM (real LMModel + LMGen, greedy, float64): three
     conversations whose prompts differ in length across rows go through ``converse_batched`` at
     once, and every row must equal its own B=1 ``converse`` run token for token -- rows still in
     their prompt are forced, rows past it keep their own output. With the per-row forcing switched
     off the comparison must FAIL.
  4. SMALL CONTEXT. The same weights built with a context just large enough for the frames stepped
     give exactly the full-context output (the context sizes only the KV cache and the attention
     window); a context smaller than that is refused, not silently wrapped.

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
# ---- 3 ----
import copy  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMModel  # noqa: E402

sp.prime_from_codes = prime_from_codes  # undo [2]'s stub

torch.manual_seed(0)
kw = dict(loaders._lm_kwargs)
kw.update(
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
torch.set_default_dtype(torch.float64)  # greedy parity without float32 near-ties between batch shapes
base_lm = LMModel(device="cpu", dtype=torch.float64, **kw).eval()


class Dummy:
    """Stands in for the two mimi (self-play never encodes or decodes)."""

    _streaming_state = None

    def reset_streaming(self):
        pass

    def streaming_forever(self, _b):
        pass

    def _stop_streaming(self):
        pass


def make_model():
    g = LMGen(
        copy.deepcopy(base_lm),
        device="cpu",
        use_sampling=False,
        audio_silence_frame_cnt=SIL,
        sample_rate=24000,
        frame_rate=12.5,
    )
    g.streaming_forever(1)
    return types.SimpleNamespace(lm_gen=g, mimi=Dummy(), other_mimi=Dummy())


# Rows differ in prompt length ACROSS rows (voice 50/44/47 frames, text 5/9/2 tokens); within a row
# both sides have equal length, which is where B=1 and batched are the same protocol.
rows = []
for vlen, tlen in ((50, 5), (44, 9), (47, 2)):
    va = rng.integers(0, 2048, size=(8, vlen))
    vb = rng.integers(0, 2048, size=(8, vlen))
    ta = [int(x) for x in rng.integers(4, 100, size=tlen)]
    tb = [int(x) for x in rng.integers(4, 100, size=tlen)]
    rows.append((va, ta, vb, tb))
NF = 20
ref = []
for va, ta, vb, tb in rows:
    a, b = make_model(), make_model()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        ref.append(converse(a, b, voice_a=va, voice_b=vb, text_a=ta, text_b=tb, n_frames=NF))


def run_batched():
    a, b = make_model(), make_model()
    return sp.converse_batched(
        a,
        b,
        voices_a=[r[0] for r in rows],
        texts_a=[r[1] for r in rows],
        voices_b=[r[2] for r in rows],
        texts_b=[r[3] for r in rows],
        n_frames=NF,
        silence_frames=SIL,
    )


(((ca, ta_), (cb, tb_)), (la, lb)) = run_batched()
for i, ((rca, rta), (rcb, rtb)) in enumerate(ref):
    assert np.array_equal(ca[i], rca[:, :NF]) and np.array_equal(cb[i], rcb[:, :NF]), f"row {i}: batched codes != B=1"
    assert np.array_equal(ta_[i], rta[:NF]) and np.array_equal(tb_[i], rtb[:NF]), f"row {i}: batched text != B=1"
assert len(set(la.tolist())) == 3, la
real_force = sp._force_rows
sp._force_rows = lambda *a, **k: None
try:
    (((ca2, _), _), _) = run_batched()
finally:
    sp._force_rows = real_force
assert not all(np.array_equal(ca2[i], ref[i][0][0][:, :NF]) for i in range(3)), (
    "guard is vacuous: unforced prompts matched"
)
print(
    f"[3] batched == B=1 for 3 rows with prompt lengths {la.tolist()} ({NF} frames each); without per-row forcing it differs"
)
# ---- 4 ----
def make_model_ctx(ctx):
    kw_c = dict(kw, context=ctx)
    lm = LMModel(device="cpu", dtype=torch.float64, **kw_c).eval()
    lm.load_state_dict(base_lm.state_dict())
    g = LMGen(lm, device="cpu", use_sampling=False, audio_silence_frame_cnt=SIL, sample_rate=24000, frame_rate=12.5)
    g.streaming_forever(1)
    return types.SimpleNamespace(lm_gen=g, mimi=Dummy(), other_mimi=Dummy())


steps = int(max(la.max(), lb.max())) + NF
kw_rows = dict(voices_a=[r[0] for r in rows], texts_a=[r[1] for r in rows], voices_b=[r[2] for r in rows],
               texts_b=[r[3] for r in rows], n_frames=NF, silence_frames=SIL)
(((cs, _), (cs_b, _)), _) = sp.converse_batched(make_model_ctx(steps), make_model_ctx(steps), **kw_rows)
assert np.array_equal(cs, ca) and np.array_equal(cs_b, cb), "a just-sufficient context changed the output"
try:
    sp.converse_batched(make_model_ctx(steps - 1), make_model_ctx(steps - 1), **kw_rows)
    raise RuntimeError("a context shorter than the run was accepted")
except AssertionError as e:
    assert "exceed the LM context" in str(e), e
print(f"[4] context {steps} (= frames stepped) == context {kw['context']}; context {steps - 1} is refused")
print("OK")
