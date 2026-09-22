"""Guard: PersonaPlex ``LMGen.reset_streaming()`` must reset the wrapped LM (KV cache + positions).

The PersonaPlex ``LMGen`` detaches its ``lm_model`` from the parent's streaming traversal, so the
parent's reset never reaches it. Until 2026-09-22 ``_LMGenState.reset`` did not forward the reset
either, and every clip / conversation after the first one in a process ran on the previous one's
KV cache (p0/p1 knowledge fell from ~0.2 in the first batch group to 0.00 by group 6).

  1. RESET REACHES THE LM. After 30 steps and ``lm_gen.reset_streaming()`` the transformer's
     streaming offsets and the first layer's KV ``end_offset`` are 0.
  2. SECOND GROUP == FRESH. Two tiny random PersonaPlex LMs (real LMModel + LMGen, greedy, float64)
     run one ``converse_batched`` group, then a second group; the second group's codes must equal
     the same group run on fresh instances, token for token.
  3. CAN FAIL. With the reset hook removed, check 2 must NOT hold.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_reset.py'
"""

import contextlib
import copy
import os
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402
import moshi_family.personaplex.selfplay as sp  # noqa: E402

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
SIL = 6


class _NoMimi:
    _streaming_state = None

    def reset_streaming(self):
        pass

    def streaming_forever(self, _b):
        pass

    def _stop_streaming(self):
        pass


class _NoHookLMGen(LMGen):
    """The pre-fix behaviour: every streaming state it creates has no LM reset hook."""

    def _init_streaming_state(self, batch_size):
        state = super()._init_streaming_state(batch_size)
        state.reset_callback = None
        return state


def make_model(hook: bool = True):
    cls = LMGen if hook else _NoHookLMGen
    g = cls(
        copy.deepcopy(BASE_LM),
        device="cpu",
        use_sampling=False,
        audio_silence_frame_cnt=SIL,
        sample_rate=24000,
        frame_rate=12.5,
    )
    g.streaming_forever(1)
    return types.SimpleNamespace(lm_gen=g, mimi=_NoMimi(), other_mimi=_NoMimi())


def lm_positions(m):
    tr = m.lm_gen.lm_model.transformer
    kv = getattr(tr.layers[0].self_attn._streaming_state, "kv_cache", None)
    return tr._streaming_state.offsets.tolist(), (None if kv is None else kv.end_offset.tolist())


def rows(seed):
    r = np.random.default_rng(seed)
    return [
        (
            r.integers(0, 2048, size=(8, v)),
            [int(x) for x in r.integers(4, 100, size=t)],
            r.integers(0, 2048, size=(8, v)),
            [int(x) for x in r.integers(4, 100, size=t)],
        )
        for v, t in ((50, 5), (44, 9))
    ]


def run(a, b, rs, n_frames=30):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        ((ca, _), (cb, _)), _ = sp.converse_batched(
            a,
            b,
            voices_a=[x[0] for x in rs],
            texts_a=[x[1] for x in rs],
            voices_b=[x[2] for x in rs],
            texts_b=[x[3] for x in rs],
            n_frames=n_frames,
            silence_frames=SIL,
        )
    return np.asarray(ca), np.asarray(cb)


def second_group_equals_fresh(hook: bool) -> bool:
    fresh = run(make_model(hook), make_model(hook), rows(2))
    a, b = make_model(hook), make_model(hook)
    run(a, b, rows(1))
    second = run(a, b, rows(2))
    return all(np.array_equal(x, y) for x, y in zip(fresh, second))


def main():
    m = make_model()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        for _ in range(30):
            m.lm_gen.step(m.lm_gen._encode_zero_frame())
    before = lm_positions(m)
    m.lm_gen.reset_streaming()
    after = lm_positions(m)
    assert before[0] != [0] and after[0] == [0], f"transformer offsets not reset: {before} -> {after}"
    assert after[1] in (None, [0]), f"KV end_offset not reset: {before} -> {after}"
    print(f"[ok] 1 reset reaches the LM: offsets/kv {before} -> {after}")

    assert second_group_equals_fresh(hook=True), "second conversation group differs from fresh instances"
    print("[ok] 2 a second converse_batched group on the same instances == fresh instances")

    assert not second_group_equals_fresh(hook=False), "check 2 cannot fail: it passes without the reset hook"
    print("[ok] 3 without the reset hook check 2 fails (the guard can fail)")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
