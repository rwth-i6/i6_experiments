"""The text stream's two conventions: EPAD emission (A17b) and the speech clamp (A19).

Both were found on 2026-09-09 and both are invisible in a loss curve -- a model trained without
EPAD fits its corpus beautifully and then declines to speak, and words planted over silence make the
loss *better* defined, not worse. So neither can be guarded by a metric; they need this.

The EPAD half is checked against the REFERENCE implementation
(``moshi_finetune/finetune/data/interleaver.py``), not against a restatement of our own logic --
that is the whole point, since the bug was that ours had silently diverged from it.

Run:  CUDA_HOME=/usr .venv/bin/python recipe/.../tests/check_text_stream.py
"""

import os
import sys

os.environ.setdefault("CUDA_HOME", "/usr")
# Same convention as the sibling checks: run from the setup root.
sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/speech_llm/full_duplex")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.train_data_common import (  # noqa: E402
    TEXT_EPAD_ID,
    TEXT_PADDING_ID,
    DuplexTokenizerBase,
    clamp_alignments_to_speech,
    speech_onset_sec,
)

failures: list[str] = []


def ok(msg, since):
    if len(failures) == since:
        print(f"[ok] {msg}")


FRAME_RATE = 12.5


class _StubTok:
    """Deterministic word -> ids. Ids stay well clear of TEXT_PADDING_ID/TEXT_EPAD_ID."""

    def encode(self, word):
        return [100 + len(word), 200 + len(word)][: 1 if len(word) % 2 else 2]


class _Tokenizer(DuplexTokenizerBase):
    def __init__(self):
        self.text_tok = _StubTok()
        self.frame_rate = FRAME_RATE
        self.sample_rate = 24000
        self.num_frames = 64
        self.duration_sec = 64 / FRAME_RATE
        self.device = "cpu"


TOK = _Tokenizer()
#: Onsets on EXACT frame boundaries. Ours rounds a word's onset to the nearest frame and the
#: reference floors it; on frame boundaries the two coincide, which isolates the EPAD semantics
#: under test from a +/- 1 frame (80 ms) difference that is not what this check is about.
ALIGNS = [
    ("alpha", (4 / FRAME_RATE, 6 / FRAME_RATE), "SPEAKER_MAIN"),
    ("be", (10 / FRAME_RATE, 11 / FRAME_RATE), "SPEAKER_MAIN"),
    ("gamma", (11 / FRAME_RATE, 14 / FRAME_RATE), "SPEAKER_MAIN"),
    ("d", (30 / FRAME_RATE, 31 / FRAME_RATE), "SPEAKER_MAIN"),
]

# --- 1. emit_epad=False must be byte-identical to the pre-2026-09-09 behaviour -------------------
_n = len(failures)
off = TOK.interleave_text(ALIGNS, 64)[0]
if (off == TEXT_EPAD_ID).any():
    failures.append(
        "interleave_text(emit_epad=False) emitted EPAD. The default MUST reproduce the old stream "
        "exactly -- every finished run's hash and meaning depend on it."
    )
ok("default off    no EPAD in the stream (finished runs keep their meaning)", _n)

# --- 2. EPAD lands exactly on padding->text transitions -----------------------------------------
_n = len(failures)
on = TOK.interleave_text(ALIGNS, 64, emit_epad=True)[0]
is_text_off = off != TEXT_PADDING_ID
for t in range(64):
    prev_pad = t > 0 and not bool(is_text_off[t - 1])
    want_epad = t > 0 and bool(is_text_off[t]) and prev_pad
    got = int(on[t - 1]) == TEXT_EPAD_ID if t > 0 else False
    if want_epad and not got:
        failures.append(f"frame {t - 1}: expected EPAD before the word starting at {t}")
    if got and not want_epad:
        failures.append(f"frame {t - 1}: EPAD where there is no padding->text transition")
if int(on[0]) == TEXT_EPAD_ID:
    failures.append("EPAD written at frame 0, which has no preceding frame (reference guards t > 0)")
# the words themselves must be untouched
if not torch.equal(on[is_text_off], off[is_text_off]):
    failures.append("emit_epad changed the word tokens themselves, not just the padding before them")
ok("epad placement only at padding->text transitions; word tokens untouched", _n)

# --- 3. ...and it agrees with the REFERENCE interleaver -----------------------------------------
# The bug was a silent divergence from upstream, so the check compares against upstream itself.
_n = len(failures)
try:
    from moshi_finetune.finetune.data.interleaver import Interleaver

    ref = Interleaver(
        tokenizer=None,  # build_token_stream never touches it; only _tokenize does
        audio_frame_rate=FRAME_RATE,
        text_padding=TEXT_PADDING_ID,
        end_of_text_padding=TEXT_EPAD_ID,
        zero_padding=-1,
        device="cpu",
    )
    ref_row = ref.build_token_stream(
        [(TOK.text_tok.encode(w), ts, spk) for w, ts, spk in ALIGNS], 64 / FRAME_RATE
    ).view(-1)
    if ref_row.shape[0] != on.shape[0]:
        failures.append(f"reference produced {ref_row.shape[0]} frames, ours {on.shape[0]}")
    elif not torch.equal(ref_row.cpu(), on.cpu()):
        diff = [(t, int(ref_row[t]), int(on[t])) for t in range(64) if int(ref_row[t]) != int(on[t])]
        failures.append(f"our text stream differs from the reference interleaver at {diff[:8]}")
    ok("reference parity  our EPAD stream == moshi_finetune's Interleaver, frame for frame", _n)
except ImportError as exc:
    failures.append(f"could not import the reference interleaver to compare against: {exc!r}")

# --- 3b. with onset_floor, parity holds for ARBITRARY onsets, not just frame boundaries ---------
# This is the stronger form of the test above: rounding vs flooring can only differ OFF the grid, so
# off-grid fixtures are the only ones that can see it. Non-vacuity is asserted explicitly -- the
# rounding mode must FAIL the same comparison, or this proves nothing.
_n = len(failures)
OFF_GRID = [
    ("alpha", (4.37 / FRAME_RATE, 6.1 / FRAME_RATE), "SPEAKER_MAIN"),
    ("be", (10.62 / FRAME_RATE, 11.4 / FRAME_RATE), "SPEAKER_MAIN"),
    ("gamma", (17.51 / FRAME_RATE, 20.2 / FRAME_RATE), "SPEAKER_MAIN"),
    ("d", (30.94 / FRAME_RATE, 31.7 / FRAME_RATE), "SPEAKER_MAIN"),
]
try:
    ref_off = (
        ref.build_token_stream([(TOK.text_tok.encode(w), ts, spk) for w, ts, spk in OFF_GRID], 64 / FRAME_RATE)
        .view(-1)
        .cpu()
    )
    floored = TOK.interleave_text(OFF_GRID, 64, emit_epad=True, onset_floor=True)[0].cpu()
    rounded = TOK.interleave_text(OFF_GRID, 64, emit_epad=True, onset_floor=False)[0].cpu()
    if not torch.equal(ref_off, floored):
        diff = [(t, int(ref_off[t]), int(floored[t])) for t in range(64) if int(ref_off[t]) != int(floored[t])]
        failures.append(f"onset_floor does not reproduce the reference off-grid; differs at {diff[:8]}")
    if torch.equal(ref_off, rounded):
        failures.append(
            "the ROUNDING mode also matches the reference off-grid -- the fixture cannot see the "
            "difference, so this check proves nothing"
        )
    # ...and the direction matters: flooring may only move a word EARLIER, never later.
    for w, (s, _e), _sp in OFF_GRID:
        fl = int(s * FRAME_RATE // 1)
        rd = int(round(s * FRAME_RATE))
        if fl > rd:
            failures.append(f"floor put {w!r} at frame {fl}, later than round's {rd}")
    ok("off-grid parity   onset_floor == reference for arbitrary onsets; rounding does not", _n)
except NameError:
    failures.append("reference interleaver unavailable, cannot check off-grid parity")

# --- 4. the speech clamp ------------------------------------------------------------------------
_n = len(failures)
sr = 24000
onset_s = 2.0
sig = np.zeros(int(5 * sr), dtype=np.float32)
sig[int(onset_s * sr) :] = np.random.default_rng(0).standard_normal(sig.size - int(onset_s * sr)).astype(np.float32)
measured = speech_onset_sec(sig, sr)
if measured is None or abs(measured - onset_s) > 0.05:
    failures.append(f"speech_onset_sec found {measured} for a channel that starts at {onset_s}s")
if speech_onset_sec(np.zeros(100, dtype=np.float32), sr) is not None:
    failures.append(
        "speech_onset_sec must return None for a silent channel, not 0.0 -- a caller "
        "that reads 0.0 as 'starts at the beginning' would clamp nothing"
    )

# the real defect shape: word 0 anchored at 0.000 with a 4 s span, the rest correct
bad = [
    ("That's", (0.0, 4.48), "SPEAKER_MAIN"),
    ("Eric", (4.48, 4.74), "SPEAKER_MAIN"),
    ("Clapton.", (4.74, 5.22), "SPEAKER_MAIN"),
]
fixed = clamp_alignments_to_speech(bad, 4.11, FRAME_RATE)
if abs(fixed[0][1][0] - 4.11) > 1e-9:
    failures.append(f"clamp left word 0 at {fixed[0][1][0]}, not the 4.11s audio onset")
if any(s < 4.11 - 1e-9 for _w, (s, _e), _sp in fixed):
    failures.append("a word is still placed before the assistant makes a sound")
starts = [s for _w, (s, _e), _sp in fixed]
if len(set(int(round(s * FRAME_RATE)) for s in starts)) != len(starts):
    failures.append("two clamped words land on the same frame -- interleave_text would overwrite one")
if abs(fixed[2][1][0] - 4.74) > 1e-9:
    failures.append("the clamp moved a word that was never early -- it must be minimal")
if clamp_alignments_to_speech(bad, None, FRAME_RATE) != bad:
    failures.append("an unknown onset must leave the alignments alone, not clamp to 0")
ok("speech clamp   word 0 moved to the onset, later words untouched, no frame collisions", _n)

# --- 5. non-vacuity: the clamp must actually change the stream ----------------------------------
# A repair that is a no-op on the very row it was written for would pass everything above.
_n = len(failures)
before = TOK.interleave_text(bad, 64)[0]
after = TOK.interleave_text(fixed, 64)[0]
onset_frame = int(round(4.11 * FRAME_RATE))
if bool((before[:onset_frame] != TEXT_PADDING_ID).any()) is False:
    failures.append("fixture is vacuous: the unclamped stream emits no text before the audio onset")
if bool((after[:onset_frame] != TEXT_PADDING_ID).any()):
    failures.append("text is STILL emitted before the assistant makes a sound after clamping")
ok("non-vacuous    the unclamped row really does speak over silence; the clamped one does not", _n)

# --- 6. alignment ORDER: the reference sorts by start before placing; we must too ------------------
# With overwrite-on-overlap the LAST word placed wins a contested frame, so an unsorted list lets an
# earlier word clobber a later one. normalize_alignments now sorts (stable). Fixture: two words whose
# token runs overlap, given in reverse file order -- the stream must equal the sorted one, and the
# unsorted placement must genuinely differ (else this proves nothing).
from moshi_family.train_data_common import normalize_alignments  # noqa: E402

_n = len(failures)
_late = ("late", (1.20, 1.60), "assistant")
_early = ("before", (1.12, 1.30), "assistant")  # even length -> 2 stub tokens: frames 14, 15; "late" starts at 15
if len(TOK.text_tok.encode("before")) < 2:
    failures.append("fixture needs a multi-token first word so the two runs overlap")
sorted_row = TOK.interleave_text(normalize_alignments([_late, _early]), 32)[0]
ref_row = TOK.interleave_text([_early, _late], 32)[0]
raw_row = TOK.interleave_text([_late, _early], 32)[0]
if not torch.equal(sorted_row, ref_row):
    failures.append("normalize_alignments must put words in time order before placement")
if torch.equal(raw_row, ref_row):
    failures.append("fixture is vacuous: reverse order placed identically, so the sort is untested")
if not all(
    a[1][0] <= b[1][0]
    for a, b in zip(normalize_alignments([_late, _early])[:-1], normalize_alignments([_late, _early])[1:])
):
    failures.append("normalize_alignments output is not sorted by start")
ok("time order     unsorted alignments place like the reference (sorted); reverse order really differed", _n)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("text stream conventions OK")
