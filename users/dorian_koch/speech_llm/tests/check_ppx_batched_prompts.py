"""Guard: batched PersonaPlex with a different system prompt per row (``engine.step_system_prompts_per_row``) gives
each row exactly its B=1 run.

The prompts are right-aligned: a row with a shorter prompt idles first with its exec mask off. Tiny random
PersonaPlex LM (real LMModel + LMGen, greedy, float64, CPU), batch 4, text prompts of 9 / 4 / 0 / 6 tokens, then 25
frames of random user codes:

  1. SAME AS B=1. Every row's output tokens (text + 8 agent codebooks per frame) equal a B=1 run of that row (its
     prompt through the stock ``step_system_prompts``, the same user codes), token for token.
  2. SAME POSITIONS. After the prompts each row's transformer offset and first-layer KV ``end_offset`` equal its B=1
     run's.
  3. CAN FAIL. With ``set_exec_mask`` a no-op (idle steps advance the LM), check 2 fails for every row with a shorter
     prompt and check 1 for at least one (the tiny greedy model keeps some rows' tokens despite the extra frames).

moshi_family venv, CPU, seconds:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_batched_prompts.py'
"""

import contextlib
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex.engine import step_system_prompts_per_row  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402
from moshi_family.personaplex.selfplay import set_voice_codes  # noqa: E402

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
SIL = 4
B = 4
N_USER = 25
RNG = np.random.default_rng(0)
VOICE = RNG.integers(0, 2048, size=(8, 12))
PROMPTS = [[int(x) for x in RNG.integers(4, 100, size=n)] for n in (9, 4, 0, 6)]
USER = torch.as_tensor(RNG.integers(0, 2048, size=(B, 8, N_USER)))


class _NoMimi:
    _streaming_state = None


def make_gen(batch: int) -> LMGen:
    g = LMGen(
        copy.deepcopy(BASE_LM),
        device="cpu",
        use_sampling=False,
        audio_silence_frame_cnt=SIL,
        sample_rate=24000,
        frame_rate=12.5,
    )
    g.streaming_forever(batch)
    set_voice_codes(g, VOICE)
    return g


def positions(g: LMGen, row: int) -> tuple:
    tr = g.lm_model.transformer
    kv = tr.layers[0].self_attn._streaming_state.kv_cache
    return int(tr._streaming_state.offsets[row]), int(kv.end_offset[row])


def stream(g: LMGen, user: torch.Tensor) -> np.ndarray:
    out = []
    for t in range(user.shape[-1]):
        tok = g.step(user[:, :, t : t + 1])
        if tok is not None:
            out.append(tok[:, :, 0].numpy())
    return np.stack(out, axis=-1)  # [batch, 1 + 8, frames]


def run_b1(row: int) -> tuple:
    g = make_gen(1)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        g.reset_streaming()
        g.text_prompt_tokens = PROMPTS[row] or None
        g.step_system_prompts(_NoMimi())
        pos = positions(g, 0)
        return stream(g, USER[row : row + 1])[0], pos


def run_batched(mask: bool) -> tuple:
    g = make_gen(B)
    if not mask:
        g.lm_model.set_exec_mask = lambda _m: None
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        g.reset_streaming()
        step_system_prompts_per_row(g, _NoMimi(), PROMPTS, B)
        pos = [positions(g, b) for b in range(B)]
        return stream(g, USER), pos


def compare(mask: bool) -> tuple[list[bool], list[bool]]:
    toks, pos = run_batched(mask)
    same_tok, same_pos = [], []
    for b in range(B):
        ref_tok, ref_pos = run_b1(b)
        same_tok.append(bool(np.array_equal(toks[b], ref_tok)))
        same_pos.append(pos[b] == ref_pos)
    return same_tok, same_pos


def main():
    tok, pos = compare(mask=True)
    assert all(pos), f"positions after the prompts differ from B=1 (rows: {pos})"
    print(f"[ok] 2 positions after the prompts == B=1 for all {B} rows")
    assert all(tok), f"output tokens differ from B=1 (rows: {tok})"
    print(
        f"[ok] 1 output tokens == B=1 for all {B} rows ({N_USER} user frames, prompt lengths {[len(p) for p in PROMPTS]})"
    )

    tok, pos = compare(mask=False)
    shorter = [b for b in range(B) if len(PROMPTS[b]) < max(map(len, PROMPTS))]
    assert not any(pos[b] for b in shorter), f"check 2 cannot fail: positions match without the mask ({pos})"
    assert not all(tok[b] for b in shorter), f"check 1 cannot fail: tokens match without the mask ({tok})"
    print(
        f"[ok] 3 without the exec mask rows {shorter} differ in positions, rows "
        f"{[b for b in shorter if not tok[b]]} in tokens (the guard can fail)"
    )
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
