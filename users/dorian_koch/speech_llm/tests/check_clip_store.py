"""The benchmark clip store: arrow and legacy-wav layouts must be indistinguishable to a consumer.

Guards the migration from one-wav-per-example to one arrow dataset per job. The whole safety
argument is that a consumer reads either layout through ``open_clips`` and cannot tell the
difference -- so that is what this checks, by building the SAME clips both ways and diffing what
comes back.

What would go wrong without it, in rising order of nastiness:
  * a resample/dtype slip in one reader -> transcripts differ between old and new runs, and the
    benchmark number moves for a storage reason
  * index taken from row order instead of the ``index`` column -> a sharded run reassembles in the
    wrong order and every clip is graded against the wrong reference. Loss curves and accuracy both
    look plausible; only the per-row join is wrong
  * a shard boundary dropping or duplicating a clip -> silently short benchmark

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_clip_store.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.clip_store import (  # noqa: E402
    is_clip_dataset,
    merge_clip_datasets,
    open_clips,
    write_clips,
)

failures = []


def ok(msg, since):
    if len(failures) == since:
        print(f"[ok] {msg}")


# Deliberately irregular: differing lengths, a zero-length clip (a legitimate outcome -- a cascaded
# backend that stayed silent), and non-contiguous indices so row order != index.
rng = np.random.default_rng(0)


def _audio(n):
    """In-range mono audio, like a real clip. Values MUST stay inside [-1, 1]: soundfile writes
    PCM_16 by default, so anything outside clips and the two layouts genuinely diverge."""
    return (rng.standard_normal(n).astype(np.float32) * 0.25).clip(-0.99, 0.99)


CLIPS = {
    0: (_audio(1600), 16000),
    3: (_audio(800), 16000),
    7: (np.zeros(0, dtype=np.float32), 24000),
    11: (_audio(2400), 24000),
}

#: The legacy layout stores PCM_16, the arrow layout exact float32, so the two agree only to one
#: 16-bit step (~3.1e-5). That is far below anything Whisper or the graders resolve, but it is a
#: real difference and the tolerance states it rather than hiding it behind a loose atol.
PCM16_STEP = 2.0 / 32768

tmp = Path(tempfile.mkdtemp(prefix="clipstore_check_"))
try:
    # --- build the same clips both ways ---------------------------------------------------------
    wav_dir = tmp / "wavs"
    wav_dir.mkdir()
    for i, (audio, sr) in CLIPS.items():
        sf.write(wav_dir / f"{i}.wav", audio, sr)
    (wav_dir / "0.txt").write_text("monologue zero")

    arrow_dir = tmp / "arrow"
    write_clips(
        arrow_dir,
        [(i, a, sr) for i, (a, sr) in CLIPS.items()],
        monologues={0: "monologue zero"},
    )

    # --- layout sniffing ------------------------------------------------------------------------
    _n = len(failures)
    if is_clip_dataset(wav_dir):
        failures.append("is_clip_dataset called a wav dir an arrow dataset")
    if not is_clip_dataset(arrow_dir):
        failures.append("is_clip_dataset did not recognise an arrow dataset")
    if is_clip_dataset(tmp / "does_not_exist"):
        failures.append("is_clip_dataset must treat a missing path as the legacy layout, not arrow")
    ok("layout sniff        wav dir / arrow / missing path all classified correctly", _n)

    # --- the core claim: both layouts read back identically -------------------------------------
    _n = len(failures)
    a_wav, a_arrow = open_clips(wav_dir), open_clips(arrow_dir)
    if sorted(a_wav) != sorted(CLIPS) or sorted(a_arrow) != sorted(CLIPS):
        failures.append(f"index sets differ: wav={sorted(a_wav)} arrow={sorted(a_arrow)} want={sorted(CLIPS)}")
    else:
        for i, (want_audio, want_sr) in CLIPS.items():
            for label, store in (("wav", a_wav), ("arrow", a_arrow)):
                got_audio, got_sr = store[i]
                if got_sr != want_sr:
                    failures.append(f"{label}[{i}] sample rate {got_sr} != {want_sr}")
                if got_audio.dtype != np.float32:
                    failures.append(f"{label}[{i}] dtype {got_audio.dtype} != float32")
                if got_audio.shape != want_audio.shape:
                    failures.append(f"{label}[{i}] shape {got_audio.shape} != {want_audio.shape}")
                elif want_audio.size and not np.allclose(got_audio, want_audio, atol=PCM16_STEP):
                    failures.append(
                        f"{label}[{i}] samples differ by up to {np.abs(got_audio - want_audio).max():.2e}, "
                        f"more than one PCM-16 step ({PCM16_STEP:.2e})"
                    )
    ok(f"round trip          {len(CLIPS)} clips identical via both layouts (incl. a 0-sample clip)", _n)

    # --- index is identity, not row order -------------------------------------------------------
    # Written in an order that does NOT match sorted index, so a reader relying on row position
    # returns the wrong clip for every key.
    _n = len(failures)
    shuffled = tmp / "shuffled"
    write_clips(shuffled, [(i, CLIPS[i][0], CLIPS[i][1]) for i in (11, 0, 7, 3)])
    store = open_clips(shuffled)
    for i in CLIPS:
        got, _ = store[i]
        if got.shape != CLIPS[i][0].shape:
            failures.append(f"index {i} resolved to a clip of shape {got.shape} -- reader is using row order")
    ok("index identity      out-of-order write still resolves every clip by index", _n)

    # --- sidecars ------------------------------------------------------------------------------
    _n = len(failures)
    for label, store in (("wav", a_wav), ("arrow", a_arrow)):
        if store.sidecar(0, "monologue") != "monologue zero":
            failures.append(f"{label} lost the inner-monologue sidecar for clip 0")
        if store.sidecar(3, "monologue") not in (None, ""):
            failures.append(f"{label} invented a monologue for clip 3")
    ok("sidecars            inner monologue survives both layouts", _n)

    # --- sharded write reassembles --------------------------------------------------------------
    _n = len(failures)
    s0, s1 = tmp / "shard0", tmp / "shard1"
    write_clips(s0, [(i, CLIPS[i][0], CLIPS[i][1]) for i in (0, 3)])
    write_clips(s1, [(i, CLIPS[i][0], CLIPS[i][1]) for i in (7, 11)])
    merged = tmp / "merged"
    merge_clip_datasets(merged, [s0, s1])
    m = open_clips(merged)
    if sorted(m) != sorted(CLIPS):
        failures.append(f"merged shards give indices {sorted(m)}, want {sorted(CLIPS)}")
    ok("shard merge         two shards reassemble to the full index set", _n)

    # --- overlapping shards must be REJECTED, not silently merged -------------------------------
    _n = len(failures)
    try:
        merge_clip_datasets(tmp / "bad", [s0, s0])
        failures.append("merging overlapping shards was accepted -- duplicate clips would be graded twice")
    except ValueError:
        pass
    try:
        write_clips(tmp / "dupe", [(1, CLIPS[0][0], 16000), (1, CLIPS[3][0], 16000)])
        failures.append("write_clips accepted a duplicate index -- one clip would vanish")
    except ValueError:
        pass
    ok("collisions rejected duplicate index and overlapping shards both raise", _n)

finally:
    shutil.rmtree(tmp, ignore_errors=True)


if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nclip store: arrow and legacy wav layouts are interchangeable")
