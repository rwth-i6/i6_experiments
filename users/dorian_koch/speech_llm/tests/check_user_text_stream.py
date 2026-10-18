"""Guard: the user-text INPUT stream (``moshi_family.user_text_stream``, 2026-10-18; ``LMGen(user_stream=)``).

WHY. The user-text head's token of frame t-1 is fed back as an input at frame t (``proj(text_emb(u))`` added to the
embedding sum). It can go silently wrong in five ways: the one-frame shift is off (training then teaches a shortcut or
a lag that inference does not reproduce), a zero-init stream is not a no-op, streaming adds a different input than the
training forward, prompt frames feed something, or a reset leaks one clip's stream into the next. Real code throughout:
a tiny real PersonaPlex ``LMModel`` (float64) and the real ``LMGen``.
  1. DATA. ``user_stream_tokens``: IGNORE -> NONE, every other token kept; ``shift_one``: position 0 = NONE, position p
     = frame p-1. CAN FAIL: a shift of 0.
  2. ZERO INIT. ``forward_train`` with a zero-init stream's input == without it (``torch.equal``, text and audio
     logits); greedy ``LMGen`` with a zero-init stream (self mode) == without one, token for token (B=1, B=3).
  3. TRAINING == STREAMING. Random non-zero ``proj``: a fully teacher-forced ``LMGen`` (prompt text PROVIDED, dialogue
     text forced through ``text_sample_fn``, audio provided) in ORACLE mode with the gold dialogue tokens reproduces
     ``forward_train(input_extra=embed(shift_one(user_stream_tokens(U))))``'s text logits at every position (prompt
     included). CAN FAIL: the oracle read one frame late; the stream off in streaming.
  4. SELF MODE. The same forced run in self mode: the tokens it fed (NONE on every prompt step) are what the head
     predicts in ``forward_train`` fed those very tokens (argmax at every dialogue position) -- the fed token at t is
     the head's output at t-1. CAN FAIL: the head read from the wrong position (an off-by-one feed).
  5. RESET. Self mode, non-zero stream: a second group after ``reset_streaming`` == the same group on a fresh
     instance. CAN FAIL: a reset that does not clear the stream.
  6. SAVE / LOAD. ``save`` -> ``load`` gives identical head and proj weights and the meta back.
  7. ORACLE TOKENS. ``oracle_tokens`` (the engine's text-only tokenizer, the run's saved delay / text knobs) == the
     training loader's ``user_text_targets`` through a real ``PersonaPlexTokenizer`` (released sentencepiece + Mimi
     from the HF cache), on seeded random word timings, EPAD on and off. CAN FAIL: another delay.
  8. REFUSAL. The rollout core (``ppx_rollout.Recorder.install``: RL, forced choice) raises on an LMGen with a stream.

moshi_family venv or the setup .venv, CPU, seconds (setup root):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_user_text_stream.py
"""

import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import AUDIO_TOKENS_PER_STREAM, LMGen, LMModel  # noqa: E402
from moshi_family.train_data_common import TEXT_PADDING_ID, USER_TEXT_IGNORE  # noqa: E402
from moshi_family.user_text_stream import USER_STREAM_NONE, UserTextStream, shift_one, user_stream_tokens  # noqa: E402

torch.set_default_dtype(torch.float64)
FAILS = []
T, PP = 28, 6  # frames; prompt frames (text provided in streaming, IGNORE user targets in training)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def tiny_lm(seed=0):
    torch.manual_seed(seed)
    kw = dict(loaders._lm_kwargs)
    kw.update(
        dim=64,
        num_heads=4,
        num_layers=2,
        hidden_scale=2.0,
        text_card=100,
        context=64,
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=1,
        dep_q=16,
    )
    return LMModel(device="cpu", dtype=torch.float64, **kw).eval()


def stream_for(lm, seed=1, zero=False):
    s = UserTextStream.build(lm)
    s.double()
    if not zero:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            s.head.weight.copy_(torch.randn(s.head.weight.shape, generator=g) * 0.5)
            s.proj.weight.copy_(torch.randn(s.proj.weight.shape, generator=g) * 0.3)
    s.eval()
    return s


def random_codes(lm, B, seed=2):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, lm.card, (B, lm.num_codebooks, T), generator=g)
    codes[:, 0] = torch.randint(0, lm.text_card, (B, T), generator=g)
    return codes


def random_user(B, seed=3, text_card=100):
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(0, text_card, (B, T), generator=g)
    u[:, PP : PP + 2] = TEXT_PADDING_ID  # the delay fill at the dialogue start
    u[:, :PP] = USER_TEXT_IGNORE
    return u


def train_logits(lm, codes, stream=None, user=None, capture=False):
    extra = None
    if stream is not None:
        extra = stream.embed(lm.text_emb, shift_one(user_stream_tokens(user)))
    hidden = {}
    h = None
    if capture:
        h = lm.text_linear.register_forward_hook(lambda _m, i, _o: hidden.__setitem__("h", i[0]))
    with torch.no_grad():
        out = lm.forward_train(codes, input_extra=extra)
    if h is not None:
        h.remove()
    return out, hidden.get("h")


def forced_stream(lm, codes, stream=None, mode="oracle", oracle=None, gen=None):
    """Every token teacher-forced through the real LMGen: returns (text logits per codes frame [T] x [B, V] -- the
    float64 ``text_linear`` output of each step, the tokens fed after each step [T] x [B])."""
    B = codes.shape[0]
    if gen is None:
        gen = LMGen(lm, device="cpu", use_sampling=True, user_stream=stream, user_stream_mode=mode)
    with gen.streaming(B):
        if oracle is not None:
            gen.set_user_text_oracle(oracle)
        return _forced_steps(gen, lm, codes)


def _forced_steps(gen, lm, codes):
    """The forced steps on an instance that is already streaming: prompt frames (f < PP) PROVIDE the text token,
    dialogue frames force it through ``text_sample_fn``; audio always provided."""
    k = {"call": 0}
    gen.text_sample_fn = lambda tl: codes[:, :1, k["call"] - 1 : k["call"]].clone()
    out = []
    h = gen.lm_model.text_linear.register_forward_hook(lambda _m, _i, o: out.append(o[:, -1].clone()))
    need = lm.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
    logits, fed = [], []
    try:
        for call in range(T + 1):
            k["call"] = call
            if call == 0:
                gen.step()
                continue
            f = call - 1
            moshi, user = codes[:, 1 : 1 + need, f : f + 1], codes[:, 1 + need : 1 + 2 * need, f : f + 1]
            n = len(out)
            if f < PP:
                gen.step(input_tokens=user, moshi_tokens=moshi, text_token=codes[:, 0, f])
            else:
                gen.step(input_tokens=user, moshi_tokens=moshi)
            assert len(out) == n + 1, "one text_linear call per step"
            logits.append(out[n])
            st = gen._streaming_state
            fed.append(None if st.user_in is None else st.user_in.clone())
    finally:
        h.remove()
    return logits, fed


def part_data():
    t = torch.tensor([[USER_TEXT_IGNORE, 3, 7, USER_TEXT_IGNORE, 0]])
    s = user_stream_tokens(t)
    check(
        "1 data: IGNORE -> NONE, tokens kept", s.tolist() == [[USER_STREAM_NONE, 3, 7, USER_STREAM_NONE, 0]], s.tolist()
    )
    p = shift_one(torch.tensor([[11, 12, 13]]))
    check("1 data: position 0 NONE, p reads p-1", p.tolist() == [[USER_STREAM_NONE, 11, 12]], p.tolist())
    check("1 CAN FAIL: a shift of 0 differs", torch.tensor([[11, 12, 13]]).tolist() != p.tolist())


def part_zero_init():
    lm = tiny_lm()
    z = stream_for(lm, zero=True)
    codes, user = random_codes(lm, 2), random_user(2)
    a, _ = train_logits(lm, codes)
    b, _ = train_logits(lm, codes, z, user)
    same = torch.equal(a.text_logits.nan_to_num(), b.text_logits.nan_to_num()) and torch.equal(
        a.logits.nan_to_num(), b.logits.nan_to_num()
    )
    check("2 zero init: forward_train unchanged (text + audio logits, exact)", same)
    nz = stream_for(lm)
    c, _ = train_logits(lm, codes, nz, user)
    check(
        "2 CAN FAIL: a non-zero stream changes forward_train",
        not torch.equal(a.text_logits.nan_to_num(), c.text_logits.nan_to_num()),
    )
    for B in (1, 3):
        outs = []
        for s in (None, z):
            torch.manual_seed(7)
            gen = LMGen(copy.deepcopy(lm), device="cpu", use_sampling=False, user_stream=s)
            toks = []
            with gen.streaming(B):
                g = torch.Generator().manual_seed(9)
                for _ in range(20):
                    o = gen.step(input_tokens=torch.randint(0, lm.card, (B, 8, 1), generator=g))
                    if o is not None:
                        toks.append(o.clone())
            outs.append(torch.cat(toks, dim=-1))
        check(
            f"2 zero init: greedy LMGen identical with the stream (B={B})",
            torch.equal(outs[0], outs[1]),
            tuple(outs[0].shape),
        )


def part_parity():
    lm = tiny_lm()
    s = stream_for(lm)
    B = 2
    codes, user = random_codes(lm, B), random_user(B)
    out, _ = train_logits(lm, codes, s, user)
    ref = out.text_logits[:, 0]  # [B, T, V]
    oracle = user[:, PP:].clone()
    got, _ = forced_stream(lm, codes, s, "oracle", oracle)
    gaps = [float((got[f] - ref[:, f]).abs().max()) for f in range(T) if got[f] is not None]
    check(
        "3 training == streaming (oracle mode): text logits at every forced position",
        max(gaps) < 1e-9,
        f"max {max(gaps):.2e} over {len(gaps)} frames",
    )
    # CAN FAIL: oracle one frame late
    late = torch.cat([oracle[:, 1:], oracle[:, -1:]], dim=1)
    got2, _ = forced_stream(lm, codes, s, "oracle", late)
    g2 = max(float((got2[f] - ref[:, f]).abs().max()) for f in range(PP, T))
    check("3 CAN FAIL: the oracle read one frame late", g2 > 1e-4, f"max {g2:.2e}")
    got3, _ = forced_stream(lm, codes, s, "off")
    g3 = max(float((got3[f] - ref[:, f]).abs().max()) for f in range(PP, T))
    check("3 CAN FAIL: the stream off in streaming", g3 > 1e-4, f"max {g3:.2e}")


def part_self():
    lm = tiny_lm()
    s = stream_for(lm)
    B = 2
    codes = random_codes(lm, B)
    _, fed = forced_stream(lm, codes, s, "self")
    fed = torch.stack(fed, dim=1)  # [B, T]: the token fed AFTER the step of codes frame f
    check(
        "4 self: NONE after every prompt step", bool((fed[:, :PP] == USER_STREAM_NONE).all()), fed[0, : PP + 2].tolist()
    )
    user = fed.clone()
    user[:, :PP] = USER_TEXT_IGNORE
    _, h = train_logits(lm, codes, s, user, capture=True)
    pred = s.logits(h).argmax(-1)  # [B, T]: head at position f
    ok = torch.equal(pred[:, PP:], fed[:, PP:])
    check("4 self: fed token at f+1 == head argmax at f under the training forward", ok)
    # CAN FAIL: the comparison one position off
    check("4 CAN FAIL: off by one position", not torch.equal(pred[:, PP:-1], fed[:, PP + 1 :]))


def part_reset():
    """The engines' path: one instance under ``streaming_forever``, ``reset_streaming`` between groups."""
    lm = tiny_lm()
    s = stream_for(lm)
    B = 2
    c1, c2 = random_codes(lm, B, seed=11), random_codes(lm, B, seed=12)

    def logits_of(gen, codes):
        return torch.stack(_forced_steps(gen, lm, codes)[0], 1)

    fresh_gen = LMGen(copy.deepcopy(lm), device="cpu", use_sampling=True, user_stream=s)
    fresh_gen.streaming_forever(B)
    fresh = logits_of(fresh_gen, c2)
    gen = LMGen(copy.deepcopy(lm), device="cpu", use_sampling=True, user_stream=s)
    gen.streaming_forever(B)
    logits_of(gen, c1)
    gen.reset_streaming()
    again = logits_of(gen, c2)
    check(
        "5 reset: group 2 after reset_streaming == group 2 on a fresh instance (logits, exact)",
        torch.equal(again, fresh),
    )
    # CAN FAIL: a reset that keeps the stream's last token
    from moshi_family.personaplex.models import lm as lmmod

    orig = lmmod._LMGenState.reset

    def leaky(self, mask):
        keep = None if self.user_in is None else self.user_in.clone()
        orig(self, mask)
        if keep is not None:
            self.user_in.copy_(keep)

    gen2 = LMGen(copy.deepcopy(lm), device="cpu", use_sampling=True, user_stream=s)
    gen2.streaming_forever(B)
    logits_of(gen2, c1)
    lmmod._LMGenState.reset = leaky
    try:
        gen2.reset_streaming()
    finally:
        lmmod._LMGenState.reset = orig
    leaked = logits_of(gen2, c2)
    gap = float((leaked - fresh).abs().max())
    check("5 CAN FAIL: a reset that keeps the stream token differs from fresh", gap > 1e-6, f"max {gap:.2e}")


def part_save_load():
    lm = tiny_lm()
    s = stream_for(lm).float()
    with tempfile.TemporaryDirectory() as d:
        s.save(d, delay_frames=6, text_kwargs={"emit_epad": True})
        t, meta = UserTextStream.load(d, lm, device="cpu")
    ok = torch.equal(t.head.weight, s.head.weight) and torch.equal(t.proj.weight, s.proj.weight)
    check("6 save/load: identical weights", ok)
    check("6 save/load: meta", meta.get("delay_frames") == 6 and meta.get("text_kwargs") == {"emit_epad": True}, meta)


def part_oracle_tokens():
    import random

    from huggingface_hub import hf_hub_download
    from moshi_family.personaplex.train_data import PersonaPlexDataConfig, PersonaPlexTokenizer
    from moshi_family.train_data_common import user_text_targets
    from moshi_family.user_text_stream import oracle_tokens

    repo = "nvidia/personaplex-7b-v1"
    hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)  # offline: from the cache
    rng = random.Random(4)
    vocab = ["who", "wrote", "the", "novel", "uncle", "toms", "cabin", "in", "eighteen", "fifty", "two", "paris"]
    for epad in (True, False):
        tok = PersonaPlexTokenizer(repo, device="cpu", cfg=PersonaPlexDataConfig(duration_sec=10.0, emit_epad=epad))
        meta = {"delay_frames": 6, "text_kwargs": tok._text_kwargs()}
        ok, bad = True, None
        for case in range(20):
            t, words = 0.3 + rng.random(), []
            for _ in range(rng.randint(1, 12)):
                d = 0.15 + 0.5 * rng.random()
                words.append((rng.choice(vocab), (t, t + d), "B"))
                t += d + 0.3 * rng.random()
            n = int(12.5 * (t + 2))
            ref = user_text_targets(tok, words, n, delay_frames=6, text_kwargs=tok._text_kwargs())
            got = oracle_tokens(tok.text_tok, 12.5, words, n, meta=meta)
            if not torch.equal(ref, got):
                ok, bad = False, case
        check(f"7 oracle tokens == the loader's user_text_targets (emit_epad={epad}, 20 cases)", ok, f"first bad {bad}")
    wrong = oracle_tokens(tok.text_tok, 12.5, words, n, meta={**meta, "delay_frames": 5})
    check("7 CAN FAIL: another delay differs", not torch.equal(wrong, ref))


def part_refusal():
    from moshi_family.rl.ppx_rollout import Recorder

    lm = tiny_lm()
    gen = LMGen(lm, device="cpu", use_sampling=True, user_stream=stream_for(lm))
    gen.streaming_forever(1)
    try:
        Recorder(gen).install()
        raised = False
    except NotImplementedError:
        raised = True
    check("8 the rollout core refuses an LMGen with a user-text stream", raised)


def main():
    part_data()
    part_zero_init()
    part_parity()
    part_self()
    part_reset()
    part_save_load()
    part_oracle_tokens()
    part_refusal()
    print("\nALL CHECKS PASSED" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
