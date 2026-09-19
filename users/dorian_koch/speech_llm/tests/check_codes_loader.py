"""Guard: the pre-encoded (mimi codes) corpus path, backlog E6.

WHY. Storing codes instead of waveforms removes the mimi encode from the training step and makes a
2,400-hour podcast corpus fit in ~3.5 GB. Every way of getting it wrong produces an array of the
right dtype and a plausible length, trains without complaint, and yields noise or -- worse -- a
perfectly fluent model trained on the wrong speaker. Nothing here is visible in a loss curve.

The three failure modes, each guarded BOTH ways so the check cannot pass vacuously:

  1. THE RESHAPE. Codes are stored codebook-major (`k * n_frames + f`). Recovering [K, F] is a
     reshape and a time window is a slice of the SECOND axis; a contiguous slice of the flat array
     takes a band of CODEBOOKS instead. The guard asserts the right reading is right AND that the
     wrong one really differs.
  2. STREAM ORDER. `encode_stereo_window` produces [assistant K codebooks; user K codebooks]. The
     codes path must concatenate in that order or the model is trained to speak its interlocutor's
     lines while hearing its own -- a silent role swap. Checked against the REAL
     `encode_stereo_window` driven through a stub mimi, not against a restatement of its docstring.
  3. THE OPTION REFUSALS FIRING AT THE RIGHT TIME. The row loop wraps every row in
     `except Exception: log + continue`, so a waveform-only option rejected per-row would skip every
     row and train on an empty stream while logging. They must raise at loader CONSTRUCTION.

Login node, no GPU, ~5 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_codes_loader.py
"""

import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def find_setup_root(start):
    d = start
    for _ in range(12):
        if os.path.isdir(os.path.join(d, "recipe")) and os.path.isdir(os.path.join(d, "work")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise SystemExit(f"could not locate the setup root above {start}")


SETUP = find_setup_root(HERE)
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))

from moshi_family.moshi_train_data import (  # noqa: E402
    MAX_CONSECUTIVE_ROW_SKIPS,
    MoshiDataConfig,
    MoshiTokenizer,
    UserAugmentCfg,
    build_mixed_data_loader,
)
from moshi_family.train_data_common import (  # noqa: E402
    TEXT_PADDING_ID,
    TEXT_ROW,
    decode_codes_row,
    is_codes_table,
    slice_random_window_codes,
)

K = 8  # codebooks per channel; n_q = 16 total
FR = 12.5
SR = 24000


class _StubMimi:
    """Just the two attributes `_init_frame_geometry` reads, plus a deterministic `encode`."""

    frame_rate = FR
    sample_rate = SR

    def encode(self, wav):  # wav: [2, 1, T] -> [2, K, Ta]
        n = wav.shape[-1] // 1920
        out = torch.zeros(2, K, n, dtype=torch.long)
        for c in range(2):
            for k in range(K):
                # Content that identifies (channel, codebook, frame) uniquely, so any transposition
                # or mis-ordering downstream is detectable rather than merely plausible.
                out[c, k] = torch.arange(n) + 1000 * k + 100000 * c
        return out


class _StubText:
    """`interleave_text` only ever calls `text_tok.encode(word)` -> ids."""

    def encode(self, word):
        return [10 + (abs(hash(word)) % 900)]


class _Tok(MoshiTokenizer):
    """The REAL tokenizer with the checkpoint load bypassed.

    Subclassing MoshiTokenizer rather than DuplexTokenizerBase is deliberate: `build_codes_stored`
    lives on MoshiTokenizer, and a stub that subclasses the base would silently not have the method
    under test. (It did, on the first run -- and because the loader swallows per-row exceptions, the
    symptom was an infinite skip loop rather than an error.)
    """

    def __init__(self, duration_sec=40.0, cfg=None):
        self.cfg = cfg or MoshiDataConfig(duration_sec=duration_sec)
        self.mimi = _StubMimi()
        self.text_tok = _StubText()
        self.device = "cpu"
        self._init_frame_geometry(duration_sec)


def codes_table(ca, cu, alignments, n_cb, n_fr):
    """An arrow table in the canonical codes-corpus schema."""
    return pa.table(
        {
            "codes_assistant": pa.array([ca.reshape(-1).tolist()], type=pa.list_(pa.int16())),
            "codes_user": pa.array([cu.reshape(-1).tolist()], type=pa.list_(pa.int16())),
            "n_codebooks": pa.array([n_cb], type=pa.int32()),
            "n_frames": pa.array([n_fr], type=pa.int32()),
            "frame_rate": pa.array([FR], type=pa.float32()),
            "alignments": pa.array([alignments]),
        }
    )


def main():
    ok = 0
    F = 50

    # ------------------------------------------------------------------ [1] the reshape
    ca = np.array([[k * 1000 + f for f in range(F)] for k in range(K)], dtype=np.int16)
    cu = np.array([[k * 1000 + f + 500 for f in range(F)] for k in range(K)], dtype=np.int16)
    al = [{"text": "hi", "start": 0.5, "end": 0.8, "speaker": "assistant"}]
    t = codes_table(ca, cu, al, K, F)
    assert is_codes_table(t), "schema detection must find a codes corpus"
    got_a, got_u, fr, got_al = decode_codes_row(t, 0)
    assert got_a.shape == (K, F) and got_u.shape == (K, F), (got_a.shape, got_u.shape)
    assert np.array_equal(got_a, ca) and np.array_equal(got_u, cu), "codebook-major round trip"
    assert fr == FR and got_al == al
    ok += 1

    # NON-VACUOUS: the frame-major reading must give something DIFFERENT. If it did not, this fixture
    # could not detect the bug it exists for.
    wrong = ca.reshape(-1).reshape(F, K).T if K != F else None
    assert wrong is None or not np.array_equal(wrong, ca), "fixture cannot distinguish layouts"
    # ...and the live version of the trap: taking frames [f0, f0+W) as a CONTIGUOUS run of the flat
    # array must differ from the correct second-axis slice. Both have shape (K, W) and dtype int16,
    # which is why the wrong one trains happily.
    flat = ca.reshape(-1)
    f0, W = 10, 10
    right = flat.reshape(K, F)[:, f0 : f0 + W]
    wrong = flat[f0 * K : (f0 + W) * K].reshape(K, W)
    assert right.shape == wrong.shape, "the trap is only dangerous because the shapes agree"
    assert not np.array_equal(right, wrong), (
        "fixture cannot distinguish a codebook-band slice from a time slice"
    )
    ok += 1

    # wrong declared size must raise rather than silently reshape
    bad = codes_table(ca, cu, al, K, F + 1)
    try:
        decode_codes_row(bad, 0)
        raise AssertionError("a wrong n_frames must raise")
    except ValueError:
        pass
    ok += 1

    # ------------------------------------------------- [2] stream order vs the REAL wav path
    tok = _Tok(duration_sec=F / FR)
    n_samples = F * 1920
    a_wav = np.zeros(n_samples, dtype=np.float32)
    u_wav = np.zeros(n_samples, dtype=np.float32)
    ref_tokens, ref_frames, _ = tok.encode_stereo_window(a_wav, u_wav, SR, [])
    # The stub's encode gives channel c, codebook k the value arange+1000k+100000c, so rebuild the
    # same [K, F] pair and check the codes path produces a BIT-IDENTICAL tensor.
    sa = np.stack([np.arange(F) + 1000 * k for k in range(K)]).astype(np.int64)
    su = np.stack([np.arange(F) + 1000 * k + 100000 for k in range(K)]).astype(np.int64)
    got_tokens, got_frames, _ = tok.stored_codes_window(sa, su, [])
    assert got_frames == ref_frames, (got_frames, ref_frames)
    assert torch.equal(got_tokens, ref_tokens), (
        "the codes path must reproduce encode_stereo_window's arrangement exactly"
    )
    ok += 1

    # NON-VACUOUS: swapping the two channels must NOT reproduce it -- otherwise the check above
    # would pass for a role-swapped implementation, which is the bug it guards.
    swapped, _, _ = tok.stored_codes_window(su, sa, [])
    assert not torch.equal(swapped, ref_tokens), "a role swap must be detectable"
    assert torch.equal(got_tokens[:K], torch.from_numpy(sa)), "rows 0:K must be the ASSISTANT"
    assert torch.equal(got_tokens[K:], torch.from_numpy(su)), "rows K:2K must be the USER"
    ok += 1

    # wrong codebook count, mismatched shapes, and the clamp knob must all raise
    for bad_call in (
        lambda: tok.stored_codes_window(sa[:4], su[:4], []),
        lambda: tok.stored_codes_window(sa, su[:, :10], []),
        lambda: tok.stored_codes_window(sa, su, [], clamp_text_to_speech=True),
    ):
        try:
            bad_call()
            raise AssertionError("expected a ValueError")
        except ValueError:
            pass
    ok += 1

    # ------------------------------------------------------- [3] frame-domain windowing
    LONG = 500  # 40 s at 12.5 Hz
    la = np.stack([np.arange(LONG) + 1000 * k for k in range(K)]).astype(np.int16)
    lu = la + 100
    words = [("w%d" % i, (i * 1.0, i * 1.0 + 0.3), "assistant") for i in range(40)]
    rng = np.random.default_rng(0)
    wa, wu, wal = slice_random_window_codes(la, lu, FR, words, window_sec=10.0, rng=rng)
    assert wa.shape == (K, 125) and wu.shape == (K, 125), (wa.shape, wu.shape)
    ok += 1

    # alignments re-based by the QUANTISED start: (orig - new) must be an exact frame multiple, or
    # every word in every window carries a constant sub-frame error.
    orig_by_text = {w: s for w, (s, _e), _ in words}
    for w, (s, _e), _ in wal:
        shift = orig_by_text[w] - s
        assert abs(shift * FR - round(shift * FR)) < 1e-9, (
            f"{w}: re-based by {shift}s, which is {shift * FR} frames -- not a whole frame"
        )
    ok += 1

    # words fully outside the window are dropped, and the kept ones really lie in it
    assert all(e > 0.0 and s < 10.0 for _, (s, e), _ in wal), wal
    assert len(wal) < len(words), "a 10 s window of a 40 s row must drop words"
    ok += 1

    # a row no longer than the window is returned untouched (identity, not a copy-with-same-values)
    sa2, su2, al2 = slice_random_window_codes(
        ca, cu, FR, words, window_sec=1000.0, rng=np.random.default_rng(0)
    )
    assert sa2 is ca and su2 is cu and al2 is words, "short rows must pass straight through"
    ok += 1

    # same seed -> same window (a loader must be reproducible)
    r1 = slice_random_window_codes(la, lu, FR, words, window_sec=10.0, rng=np.random.default_rng(7))
    r2 = slice_random_window_codes(la, lu, FR, words, window_sec=10.0, rng=np.random.default_rng(7))
    assert np.array_equal(r1[0], r2[0]) and r1[2] == r2[2], "seeded slicing must be deterministic"
    ok += 1

    # ------------------------------- [4] waveform-only knobs are refused AT CONSTRUCTION
    # This is the one that matters operationally: raised per-row it would skip every row silently.
    from datasets import Dataset  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        ds_dir = os.path.join(td, "codes_corpus")
        Dataset.from_dict(
            {
                "codes_assistant": [la.reshape(-1).tolist()],
                "codes_user": [lu.reshape(-1).tolist()],
                "n_codebooks": [K],
                "n_frames": [LONG],
                "frame_rate": [FR],
                "alignments": [[{"text": "hi", "start": 0.5, "end": 0.9, "speaker": "assistant"}]],
            }
        ).save_to_disk(ds_dir)

        def make(cfg):
            return build_mixed_data_loader(
                [(ds_dir, 1.0, 10.0)],
                tokenizer=_Tok(duration_sec=10.0, cfg=cfg),
                batch_size=1,
                seed=0,
            )

        base = MoshiDataConfig(duration_sec=10.0)
        for label, cfg, kw in (
            ("audio_jitter_sec", MoshiDataConfig(duration_sec=10.0, audio_jitter_sec=0.5), {}),
            ("clamp_text_to_speech", MoshiDataConfig(duration_sec=10.0, clamp_text_to_speech=True), {}),
            (
                "user_augment",
                MoshiDataConfig(duration_sec=10.0, user_augment=UserAugmentCfg(gain_prob=1.0)),
                {},
            ),
        ):
            try:
                it = make(cfg)
                next(iter(it))
                raise AssertionError(f"{label} must be refused on a codes corpus")
            except ValueError as e:
                assert label.split("_")[0] in str(e).lower() or label in str(e), str(e)
            except AssertionError:
                raise
        ok += 1

        # ...and with no waveform-only knobs it must actually PRODUCE a batch. Without this, the
        # refusals above could be satisfied by a path that rejects everything.
        loader = build_mixed_data_loader(
            [(ds_dir, 1.0, 10.0)],
            tokenizer=_Tok(duration_sec=10.0, cfg=base),
            batch_size=1,
            seed=0,
        )
        codes, mask = next(iter(loader))
        assert codes.shape[0] == 1, codes.shape
        assert codes.shape[1] == 1 + 2 * K, f"expected 1 text row + {2 * K} audio rows, got {codes.shape}"
        assert codes.shape[2] == 125, codes.shape
        # the text row must not be all-PAD: that is the exact silent failure this corpus risks
        text = codes[0, TEXT_ROW]
        assert (text != TEXT_PADDING_ID).any(), (
            "the text row is entirely PAD -- this is the pad-collapse failure mode, not a pass"
        )
        ok += 1

        # ------------------------------ [5] a corpus that ALWAYS fails must RAISE, not hang
        # The per-row `except ... continue` exists so one bad row cannot kill a run. Combined with
        # infinite=True that made a wholly-unreadable corpus spin forever: no batch, no error, a
        # warning per attempt, and a live GPU sitting at step 0. Found the hard way while writing
        # this guard. The breaker must convert that into a loud failure.
        bad_dir = os.path.join(td, "bad_corpus")
        half = np.stack([np.arange(LONG) + k for k in range(K // 2)]).astype(np.int16)
        Dataset.from_dict(
            {
                "codes_assistant": [half.reshape(-1).tolist()],
                "codes_user": [half.reshape(-1).tolist()],
                "n_codebooks": [K // 2],  # 2*4 = 8 != n_q = 16 -> every row raises
                "n_frames": [LONG],
                "frame_rate": [FR],
                "alignments": [[{"text": "hi", "start": 0.5, "end": 0.9, "speaker": "assistant"}]],
            }
        ).save_to_disk(bad_dir)
        bad_loader = build_mixed_data_loader(
            [(bad_dir, 1.0, 10.0)],
            tokenizer=_Tok(duration_sec=10.0, cfg=base),
            batch_size=1,
            seed=0,
        )
        try:
            next(iter(bad_loader))
            raise AssertionError("an unreadable corpus must raise, not spin forever")
        except RuntimeError as e:
            assert "consecutive" in str(e), str(e)
        assert MAX_CONSECUTIVE_ROW_SKIPS < 100000, "the breaker must actually bound the loop"
        ok += 1

    print(f"check_codes_loader: {ok}/{ok} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
