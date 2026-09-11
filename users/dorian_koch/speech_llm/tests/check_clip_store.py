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
    materialise_clips,
    merge_clip_datasets,
    open_clips,
    repack_wav_dir,
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

    # --- an EMPTY shard must not crash the merge (regression, 2026-08-02) -------------------------
    # A job that produced no clips writes a dataset dir whose state.json lists no data files.
    # load_from_disk on that raises IndexError from inside pyarrow ("list index out of range"),
    # which names neither the empty shard nor the job that produced it -- the failure surfaces in
    # the merge, far from its cause. Merging must skip empty shards and, if nothing is left, say so.
    _n = len(failures)
    empty = tmp / "empty_shard"
    write_clips(empty, [])
    mixed = tmp / "mixed"
    try:
        merge_clip_datasets(mixed, [s0, empty, s1])
        got = sorted(open_clips(mixed))
        if got != sorted(CLIPS):
            failures.append(f"merge with an empty shard gave {got}, want {sorted(CLIPS)}")
    except Exception as exc:
        failures.append(f"merge blew up on an empty shard ({type(exc).__name__}: {exc})")
    try:
        merge_clip_datasets(tmp / "all_empty", [empty])
        failures.append("merging only-empty shards was accepted -- an empty result would look valid")
    except ValueError:
        pass  # the clear, named error
    except Exception as exc:
        failures.append(f"only-empty merge raised {type(exc).__name__}, expected a ValueError naming the cause")
    ok("empty shards        skipped when mixed in, and rejected clearly when that is all there is", _n)

    # --- repack: rewriting a finished wav dir in place (backlog E13a) ---------------------------
    # This runs over FINISHED experiment outputs and deletes the originals, so the checks that
    # matter are that what comes back is identical, that it refuses the cases where it would break
    # something, and that the refusals are not vacuous.
    _n = len(failures)
    rp_dir = tmp / "repack_me"
    rp_dir.mkdir()
    for i, (audio, sr) in CLIPS.items():
        sf.write(rp_dir / f"{i}.wav", audio, sr)
    (rp_dir / "0.txt").write_text("monologue zero")
    before = {i: open_clips(rp_dir)[i] for i in CLIPS}

    dry = repack_wav_dir(rp_dir, dry_run=True)
    if dry["action"] != "would-repack" or dry["clips"] != len(CLIPS):
        failures.append(f"dry run should report would-repack over {len(CLIPS)} clips, got {dry}")
    if is_clip_dataset(rp_dir):
        failures.append("dry run CONVERTED the directory -- it must not touch anything")
    if not sorted(p.name for p in rp_dir.glob("*.wav")):
        failures.append("dry run deleted the wavs")
    ok("repack dry run      reports the work and changes nothing", _n)

    _n = len(failures)
    res = repack_wav_dir(rp_dir, dry_run=False)
    if res["action"] != "repacked":
        failures.append(f"repack should have repacked, got {res}")
    if not is_clip_dataset(rp_dir):
        failures.append("after repack the dir is still not an arrow dataset")
    if list(rp_dir.glob("*.wav")):
        failures.append("repack left the original wavs behind -- it reclaims no inodes")
    after = open_clips(rp_dir)
    if sorted(after) != sorted(CLIPS):
        failures.append(f"repack changed the index set: {sorted(after)} != {sorted(CLIPS)}")
    for i in CLIPS:
        got, got_sr = after[i]
        was, was_sr = before[i]
        if got_sr != was_sr or got.shape != was.shape or not np.array_equal(got, was):
            failures.append(f"repack changed clip {i}")
    if after.sidecar(0, "monologue") != "monologue zero":
        failures.append("repack dropped the monologue sidecar")
    ok("repack in place     same clips, same indices, sidecar kept, originals gone", _n)

    # Idempotence matters because the real run is a list of ~1400 dirs that may partly fail; a
    # re-run must skip what already succeeded rather than double-convert or raise.
    _n = len(failures)
    again = repack_wav_dir(rp_dir, dry_run=False)
    if again["action"] != "already-arrow":
        failures.append(f"repacking an arrow dir should be a no-op, got {again}")
    ok("repack idempotent   an already-arrow dir is skipped", _n)

    # A merge dir's entries are symlinks INTO shard dirs. Repacking the merge is the point (that is
    # where 119k of our 122k symlinks are), but it has to be asked for, because repacking a SHARD
    # that a merge still points at leaves the merge dangling -- silently, since os.symlink does.
    _n = len(failures)
    shard = tmp / "link_shard"
    shard.mkdir()
    merged = tmp / "link_merged"
    merged.mkdir()
    for i, (audio, sr) in CLIPS.items():
        sf.write(shard / f"{i}.wav", audio, sr)
        os.symlink(shard / f"{i}.wav", merged / f"{i}.wav")
    try:
        repack_wav_dir(merged, dry_run=False)
        failures.append("a symlink-merge dir was repacked without follow_symlinks -- shards can dangle")
    except ValueError:
        pass
    if not (merged / "0.wav").is_symlink():
        failures.append("the refused repack still modified the merge dir")
    ok("repack refuses      a symlink-merge dir unless follow_symlinks is passed", _n)

    _n = len(failures)
    res = repack_wav_dir(merged, dry_run=False, follow_symlinks=True)
    if res["action"] != "repacked":
        failures.append(f"follow_symlinks should repack the merge, got {res}")
    if list(merged.glob("*.wav")):
        failures.append("repacking the merge left the symlinks -- those ARE the inodes being freed")
    got = open_clips(merged)
    for i in CLIPS:
        a, sr = got[i]
        was, was_sr = before[i]
        if sr != was_sr or not np.array_equal(a, was):
            failures.append(f"merge repack changed clip {i}")
    if not (shard / "0.wav").is_file():
        failures.append("repacking the merge deleted the SHARD's file -- it must only drop its own links")
    ok("repack merge        follows links, frees them, and leaves the shard intact", _n)

    # --- the arrow 2 GiB chunk ceiling ----------------------------------------------------------
    # A list<float32> chunk is addressed with int32 offsets, so one chunk cannot hold more than
    # 2 GiB of audio. A 1000-clip merge dir of ~4 MB clips does exceed that, and the failure is
    # ArrowInvalid("Value 2148637440 too large to fit in C integer type") -- it killed the first
    # dir of the 2026-09-11 repack run. Building a real 2 GiB fixture here would cost more than the
    # guard is worth, so the cap is lowered instead: the fixture's 4,800 samples then span several
    # chunks, which is the code path that was broken.
    _n = len(failures)
    import i6_experiments.users.dorian_koch.speech_llm.clip_store as _cs

    total_samples = sum(a.size for a, _ in CLIPS.values())
    tiny_cap = 1000
    assert total_samples > tiny_cap, "fixture must exceed the cap or this proves nothing"
    batched = tmp / "batched"
    saved_cap = _cs._MAX_SAMPLES_PER_CHUNK
    try:
        _cs._MAX_SAMPLES_PER_CHUNK = tiny_cap
        write_clips(batched, [(i, a, sr) for i, (a, sr) in CLIPS.items()], monologues={0: "monologue zero"})
    finally:
        _cs._MAX_SAMPLES_PER_CHUNK = saved_cap

    got = open_clips(batched)
    if sorted(got) != sorted(CLIPS):
        failures.append(f"batched write changed the index set: {sorted(got)}")
    for i, (audio, sr) in CLIPS.items():
        a, s = got[i]
        if s != sr or a.shape != audio.shape or not np.allclose(a, audio, atol=0):
            failures.append(f"batched write changed clip {i}")
    if got.sidecar(0, "monologue") != "monologue zero":
        failures.append("batched write dropped the monologue sidecar")
    ok("oversized write     splitting into sub-2GiB chunks is bit-identical to one chunk", _n)

    # --- materialise_clips: arrow -> loose wavs, with a BOUNDED memory profile ------------------
    # SpeechInference unpacks an arrow corpus for the fork-era drivers, which glob *.wav. The
    # correctness half is obvious; the memory half is what actually bit. The old loop went through
    # _ArrowClips.__getitem__, which returns each clip as a Python LIST of floats -- on B6's 12,000
    # clips that churned the allocator into ~0.6 GB/min of linear RSS growth and OOM-killed a 16 G
    # eval 46 min in. So this pins BOTH: the wavs are identical, and the random-access path is not
    # used for an arrow source. Without the second assert the fix silently regresses to a shape
    # that only fails at corpus scale, i.e. never in a test.
    _n = len(failures)
    import i6_experiments.users.dorian_koch.speech_llm.clip_store as _cs2

    mat_out = tmp / "materialised"
    calls = {"n": 0}
    real_getitem = _cs2._ArrowClips.__getitem__

    def _counting_getitem(self, key):
        calls["n"] += 1
        return real_getitem(self, key)

    _cs2._ArrowClips.__getitem__ = _counting_getitem
    try:
        materialise_clips(arrow_dir, mat_out)
    finally:
        _cs2._ArrowClips.__getitem__ = real_getitem

    for i, (audio, sr) in CLIPS.items():
        f = mat_out / f"{i}.wav"
        if not f.is_file():
            failures.append(f"materialise did not write clip {i}")
            continue
        got, got_sr = sf.read(f, dtype="float32")
        if got_sr != sr or got.shape != audio.shape or not np.allclose(got, audio, atol=PCM16_STEP):
            failures.append(f"materialise changed clip {i}")
    if (mat_out / "0.txt").read_text() != "monologue zero":
        failures.append("materialise dropped the monologue sidecar")
    if calls["n"] != 0:
        failures.append(
            f"materialise used the random-access _ArrowClips.__getitem__ {calls['n']}x -- that is the "
            "Python-list path whose memory grows with the corpus"
        )
    ok("materialise         identical wavs, and never via the corpus-scaling random-access path", _n)

    # The legacy wav-dir source must still work (it takes the other branch entirely).
    _n = len(failures)
    mat_legacy = tmp / "materialised_legacy"
    materialise_clips(wav_dir, mat_legacy)
    if sorted(p.name for p in mat_legacy.glob("*.wav")) != sorted(f"{i}.wav" for i in CLIPS):
        failures.append("materialise from a legacy wav dir did not reproduce the clip set")
    ok("materialise legacy  a wav-dir source still round-trips", _n)

    # --- write_clips must be BOUNDED, not just correct (2026-09-11) ------------------------------
    # The four B6 evals OOM-died in SpeechInference._pack_clips AFTER inference had already
    # succeeded: write_clips accumulated every clip in `rows`, built `parts` as a second copy and
    # concatenated as a third (~3x the corpus), while the caller handed it a fully materialised
    # list. ~22 GB wanted against a 16 GB limit, as a monotone 166 MB -> 10.8 GB ramp in 55 s.
    #
    # check_common_helpers.py already asserted the TTS worker's ClipSpool streams -- and it does.
    # It proved nothing about THIS writer or about its caller: the "a guard must exercise the
    # caller's wiring, not an idealised version of it" trap in CLAUDE.md.
    #
    # The meter is weakrefs, not tracemalloc: tracemalloc does not see numpy's buffers, so it
    # reports the same few MB of Python overhead for a streaming and an accumulating writer alike
    # (measured -- it was the first thing tried here, and it could not tell them apart). Counting
    # how many clip arrays are simultaneously ALIVE tests the property directly and cannot be fooled
    # by an allocator.
    _n = len(failures)
    import weakref

    import i6_experiments.users.dorian_koch.speech_llm.clip_store as _cs

    N_CLIPS, CHUNK_ROWS = 30, 3
    SAMPLES = 100_000
    _saved_cap = _cs._MAX_SAMPLES_PER_CHUNK

    def _make(i):
        # float32 and contiguous ON PURPOSE: write_clips does np.asarray(..., dtype=np.float32),
        # which must hand back THIS object rather than a copy, or the weakref would track an array
        # the writer never held and the whole measurement would read zero. Asserted below.
        return np.full(SAMPLES, (i + 1) / N_CLIPS, dtype=np.float32)

    _probe = _make(0)
    if np.asarray(_probe, dtype=np.float32) is not _probe:
        failures.append(
            "fixture arrays are copied by write_clips' asarray, so the liveness meter would track "
            "arrays the writer never retained -- this check would pass vacuously"
        )
    del _probe

    def _measure(consume):
        """Drive `consume(gen)` and return the peak number of clip arrays alive at once."""
        refs, peak = [], 0

        def gen():
            nonlocal peak
            for i in range(N_CLIPS):
                arr = _make(i)
                refs.append(weakref.ref(arr))
                peak = max(peak, sum(1 for r in refs if r() is not None))
                yield (i, arr, 24000)
                del arr

        consume(gen())
        return peak

    _cs._MAX_SAMPLES_PER_CHUNK = SAMPLES * CHUNK_ROWS
    try:
        live = _measure(lambda g: write_clips(tmp / "bounded", g))
    finally:
        _cs._MAX_SAMPLES_PER_CHUNK = _saved_cap

    # One chunk is CHUNK_ROWS clips; the generator frame and the writer's loop variable hold a
    # couple more. Anything near N_CLIPS means the corpus is being accumulated again.
    if live > CHUNK_ROWS + 3:
        failures.append(
            f"write_clips held {live} of {N_CLIPS} clips live at once (chunk is {CHUNK_ROWS}) -- "
            "it is accumulating the corpus again, which is the 2026-09-11 B6 OOM"
        )
    got = open_clips(tmp / "bounded")
    if sorted(got) != list(range(N_CLIPS)):
        failures.append(f"bounded write lost clips: {sorted(got)[:5]}... ({len(got)} of {N_CLIPS})")
    for i in (0, N_CLIPS // 2, N_CLIPS - 1):
        a, sr = got[i]
        if sr != 24000 or not np.array_equal(a, _make(i)):
            failures.append(f"bounded write changed clip {i}")
    ok(f"write_clips bounded {live} of {N_CLIPS} clips live at peak (chunk {CHUNK_ROWS})", _n)

    # Non-vacuity: the meter must report the full corpus for a consumer that really does accumulate.
    # Without this, a meter that always read 1 would pass the assertion above for the wrong reason.
    _n = len(failures)
    eager_live = _measure(list)
    if eager_live < N_CLIPS:
        failures.append(
            f"the liveness meter read {eager_live} for a consumer that accumulates all {N_CLIPS} "
            "clips -- it cannot detect accumulation and the bounded check above is a mute button"
        )
    ok(f"bounded check bites an accumulating consumer measures {eager_live}/{N_CLIPS}", _n)

    # The caller half. _pack_clips is where the materialised list actually lived, and an edit back
    # to a list comprehension would restore the OOM with every other check still green.
    _n = len(failures)
    import ast as _ast

    _src = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/speech_inference.py").read_text()
    _pack = next(
        (n for n in _ast.walk(_ast.parse(_src)) if isinstance(n, _ast.FunctionDef) and n.name == "_pack_clips"),
        None,
    )
    if _pack is None:
        failures.append("speech_inference._pack_clips not found -- this guard has rotted")
    else:
        _call = next(
            (
                n
                for n in _ast.walk(_pack)
                if isinstance(n, _ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "write_clips"
            ),
            None,
        )
        if _call is None:
            failures.append("_pack_clips no longer calls write_clips")
        elif not isinstance(_call.args[1], _ast.GeneratorExp):
            failures.append(
                f"_pack_clips passes a {type(_call.args[1]).__name__} to write_clips, not a generator"
                " -- that materialises the whole shard and is exactly the 2026-09-11 OOM"
            )
    ok("pack_clips streams  the caller hands write_clips a generator, not a list", _n)

finally:
    shutil.rmtree(tmp, ignore_errors=True)


if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nclip store: arrow and legacy wav layouts are interchangeable")
