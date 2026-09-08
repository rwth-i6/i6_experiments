"""Guard: the in-loop knowledge probe must not touch the filesystem.

The bug this exists for (2026-08-05): ``run_knowledge_probe`` generated 64 reply wavs plus 64
``.txt`` sidecars into a scratch directory and read the sidecars straight back, purely to move a
string between two lines of the same function. The audio was never looked at. A single transient
Lustre ``EIO`` on one of those writes raised, the probe caught it, disabled itself permanently, and
a8-long ran 5,200 more steps producing no readout while still reporting success.

The retry/fail-loud policy in ``moshi_finetune_launcher`` is the net for that class of failure. This
removes the exposure instead: work that never opens a file cannot fail that way.

**The probe writes reply wavs again (2026-09-08) -- and that is not a regression, because of where.**
Listening to a run across training needs audio, and the probe is the only place it exists at probe
resolution (checkpoints are 500 steps apart; a17's collapse happened entirely between two of them).
So ``run_pairs(keep_audio=N)`` keeps the first N replies in MEMORY, scoring completes from memory as
before, and only then does ``_write_probe_audio`` -- a function that cannot raise -- put them on
disk. The invariant is therefore no longer "the probe never writes" but the stronger, more useful
"**no write can affect the measurement**": the scoring path still creates zero files, and every
hostile filesystem condition costs audio and nothing else. Both halves are asserted below.

Both modes are driven through the REAL ``run_pairs`` with a fake model, rather than asserting on a
reimplementation -- a guard that builds its own idealised caller is how ``check_mixed_loader`` missed
a live DDP bug for months (see CLAUDE.md). The assertion that matters is ``os.listdir(tmp) == []``.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_knowledge_probe_inmemory.py
"""

import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from speech_llm.full_duplex.moshi_family.knowledge_probe import (  # noqa: E402
    CHANNEL_CHECK_TAIL_S,
    STEREO_CHANNELS,
    ProbeResult,
    _write_probe_audio,
    channels_look_swapped,
    coherence_stats,
)
from speech_llm.full_duplex.moshi_family.moshi_engine import (  # noqa: E402
    RunPairsResult,
    run_pairs,
)

SR = 24000
N_CLIPS = 5
BATCH = 2
#: Token ids the decoder must drop (pad / epad), mirroring run_pairs' own filter.
DROPPED = (0, 3)


class _FakeTokenizer:
    """Maps a token id to a piece, with the sentencepiece word-boundary marker on word starts."""

    def id_to_piece(self, t: int) -> str:
        return {1: "▁the", 2: "▁answer", 4: "▁is", 5: "▁paris"}.get(t, "?")


class _FakeState:
    def __init__(self, batch_size):
        self.text_tokenizer = _FakeTokenizer()
        self.batch_size = batch_size
        self.mimi = self
        self.lm_gen = self

    def reset_streaming(self):
        pass

    def run(self, in_pcms):
        """Return ``[(text_tokens, audio)]`` per batch row, the shape run_pairs unpacks.

        Row b gets a distinguishable monologue so a positional mix-up between clips is visible, and
        audible tail energy so the truncation branch is exercised on the disk path too.
        """
        bs = in_pcms.shape[0]
        out = []
        for b in range(bs):
            toks = torch.tensor([1, 2, 0, 4, 3, 5 if b % 2 == 0 else 1])
            audio = torch.zeros(1, SR // 2)
            audio[0, -100:] = 0.5  # non-silent tail -> "possibly truncated"
            out.append((toks, audio))
        return out


class _FakeModel:
    def __init__(self, batch_size=BATCH):
        self.sample_rate = SR
        self.batch_size = batch_size
        self.device = "cpu"
        self.state = _FakeState(batch_size)


def _inputs(tmp: Path) -> list:
    """Real wav files on disk -- run_pairs reads its INPUTS with sphn either way."""
    import sphn

    paths = []
    for i in range(N_CLIPS):
        p = tmp / f"in{i}.wav"
        sphn.write_wav(str(p), np.zeros(SR, dtype=np.float32), SR)
        paths.append(str(p))
    return paths


def check_disk_mode_still_writes():
    """The benchmark path is unchanged: one wav + one txt per clip, plus truncation.json."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        out = tmp / "out"
        pairs = [(p, str(out / f"{i}.wav")) for i, p in enumerate(ins)]
        res = run_pairs(_FakeModel(), pairs, capture_s=0.1, progress_every=0)
        assert isinstance(res, RunPairsResult), type(res)
        assert res.count == N_CLIPS, res
        wavs = sorted(out.glob("*.wav"))
        txts = sorted(out.glob("*.txt"))
        assert len(wavs) == N_CLIPS, wavs
        assert len(txts) == N_CLIPS, txts
        assert (out / "truncation.json").exists(), "the truncation rate must stay queryable"
        # The sidecar content must equal what the caller now gets in memory -- otherwise the
        # in-memory path is scoring something different from what the benchmark path records.
        for i, mono in enumerate(res.monologues):
            assert (out / f"{i}.txt").read_text() == mono, i
    print("PASS  disk mode writes one wav + one txt per clip, and truncation.json")


def check_memory_mode_writes_nothing():
    """out_wav=None: not one byte on disk, and the monologues come back identical."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        before = sorted(os.listdir(tmp))

        res = run_pairs(_FakeModel(), [(p, None) for p in ins], capture_s=0.1, progress_every=0)
        assert sorted(os.listdir(tmp)) == before, (
            f"the in-memory path created files: {set(os.listdir(tmp)) - set(before)}. This is the "
            f"exact exposure that disabled a8-long's probe."
        )
        assert res.count == N_CLIPS, res
        assert len(res.monologues) == N_CLIPS, res.monologues

        # Byte-identical to what the disk path produces for the same inputs.
        out = tmp / "out"
        disk = run_pairs(
            _FakeModel(),
            [(p, str(out / f"{i}.wav")) for i, p in enumerate(ins)],
            capture_s=0.1,
            progress_every=0,
        )
        assert res.monologues == disk.monologues, (res.monologues, disk.monologues)
        # ...and non-trivial: dropped ids gone, word markers decoded, clips distinguishable.
        assert res.monologues[0] == "the answer is paris", res.monologues
        assert res.monologues[1] != res.monologues[0], "clips must not be collapsed together"
        assert "?" not in "".join(res.monologues), "an unmapped token leaked into the monologue"
    print("PASS  in-memory mode writes ZERO files and returns the same monologues as disk mode")


def check_truncation_accounting_survives_without_a_directory():
    """The truncation counters are still right with nowhere to write truncation.json."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        # tail_check_s must be shorter than the fake reply (0.5 s) or the check never fires --
        # run_pairs skips clips shorter than the tail window, which is why the default 4 s would
        # silently make this assertion vacuous.
        res = run_pairs(
            _FakeModel(),
            [(p, None) for p in ins],
            capture_s=0.1,
            tail_check_s=0.01,
            max_truncated_frac=1.0,  # deliberately 100% here; the ceiling is checked below
            progress_every=0,
        )
        # Every fake clip has a non-silent tail, so every one counts as truncated.
        assert res.truncated == N_CLIPS, res
        assert abs(res.truncated_fraction - 1.0) < 1e-9, res

        # ...and the pathological-rate ceiling still fires without a directory to report into.
        # It is the only thing standing between "verbose model" and "capture_s is simply wrong",
        # and it must not be quietly skipped just because nothing is being written.
        try:
            run_pairs(
                _FakeModel(),
                [(p, None) for p in ins],
                capture_s=0.1,
                tail_check_s=0.01,
                progress_every=0,
            )
        except AssertionError as e:
            assert "capture window" in str(e), e
        else:
            raise SystemExit("FAIL: a 100% truncation rate did not trip max_truncated_frac")
    print("PASS  truncation counters and the pathological-rate ceiling survive in-memory mode")


def check_probe_scores_positionally_against_its_own_entries():
    """run_knowledge_probe zips monologues to entries by POSITION, so lengths must be asserted."""
    import inspect

    from speech_llm.full_duplex.moshi_family import knowledge_probe

    src = inspect.getsource(knowledge_probe.run_knowledge_probe)
    assert "out_dir" not in src, "out_dir is gone; the probe must not name a scratch directory"
    assert "len(monologues) == len(entries)" in src, (
        "the positional zip between monologues and entries must be asserted -- a mismatch would "
        "score every clip against the wrong gold answer and still report a plausible accuracy"
    )
    sig = inspect.signature(knowledge_probe.run_knowledge_probe)
    assert "out_dir" not in sig.parameters, sig
    print("PASS  run_knowledge_probe takes no out_dir and asserts its positional join")


def check_keep_audio_returns_pcm_without_writing():
    """``keep_audio=N`` must hand back N replies as PCM and STILL create zero files.

    This is the load-bearing half of the listening feature: if run_pairs wrote them itself, the
    audio would be back on the scoring path and one Lustre EIO would again cost a probe step.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        before = sorted(os.listdir(tmp))
        res = run_pairs(
            _FakeModel(),
            [(p, None) for p in ins],
            capture_s=0.1,
            progress_every=0,
            keep_audio=2,
        )
        assert sorted(os.listdir(tmp)) == before, (
            f"keep_audio created files: {set(os.listdir(tmp)) - set(before)} -- the audio must "
            f"reach the caller in memory, never through the filesystem."
        )
        assert len(res.audio) == 2, f"expected 2 retained clips, got {len(res.audio)}"
        for clip in res.audio:
            # Keyed by ROLE. A tuple here would read identically whichever way round it was built,
            # which is exactly how a duplex channel map gets silently inverted.
            assert isinstance(clip, dict), type(clip)
            assert set(clip) == set(STEREO_CHANNELS), sorted(clip)
            u, a = clip[STEREO_CHANNELS[0]], clip[STEREO_CHANNELS[1]]
            assert isinstance(u, np.ndarray) and isinstance(a, np.ndarray), (type(u), type(a))
            # Sample-aligned means SAME LENGTH; a length mismatch is a time offset by another name.
            assert u.shape == a.shape, (u.shape, a.shape)
            # Trimmed to the shorter of (input, output) -- the fake model's reply is SR//2.
            assert u.shape == (SR // 2,), u.shape
        # The retained clips are the FIRST ones in input order, so the same questions are captured
        # at every probe step and the run is comparable to itself over time.
        # Default stays off: a run that does not ask for audio must retain nothing at all.
        plain = run_pairs(_FakeModel(), [(p, None) for p in ins], capture_s=0.1, progress_every=0)
        assert plain.audio == [], plain.audio
    print("PASS  keep_audio returns PCM in memory, writes nothing, and is off by default")


def check_audio_dump_cannot_break_a_probe():
    """``_write_probe_audio`` must NEVER raise, whatever the filesystem or the data does.

    Scoring has already finished by the time it runs, so its only possible cost is missing audio.
    Each case below would have propagated out of the old on-path write and killed the probe step.
    """
    pcm = np.zeros(SR // 4, dtype=np.float32)
    clip = {STEREO_CHANNELS[0]: pcm, STEREO_CHANNELS[1]: pcm}
    rows = [{"question": "q", "answer": "a", "binary_correct": 1, "monologue": "the answer"}]

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # (a) happy path -- it does actually write, or the other cases prove nothing.
        good = tmp / "good"
        n = _write_probe_audio(str(good), [clip], SR, rows)
        assert n == 1, n
        assert (good / "000.wav").exists() and (good / "000.txt").exists(), sorted(good.iterdir())
        txt = (good / "000.txt").read_text()
        assert "the answer" in txt and "gold: a" in txt, txt
        # The sidecar must state the channel layout: a listener opening a two-channel file has no
        # other way to know which side is which, and guessing wrong inverts the whole reading.
        assert f"ch0={STEREO_CHANNELS[0]}" in txt and f"ch1={STEREO_CHANNELS[1]}" in txt, txt

        # (b) unwritable parent -> makedirs fails. Returns 0, does not raise.
        ro = tmp / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)
        try:
            n = _write_probe_audio(str(ro / "nested"), [clip], SR, rows)
            assert n == 0, n
        finally:
            os.chmod(ro, 0o700)

        # (c) unwritable data -> raises per clip. Returns 0, does not raise.
        n = _write_probe_audio(
            str(tmp / "bad"),
            [{STEREO_CHANNELS[0]: "not-a-waveform", STEREO_CHANNELS[1]: pcm}],
            SR,
            rows,
        )
        assert n == 0, n

        # (c2) a clip missing a channel entirely -> KeyError, absorbed the same way.
        n = _write_probe_audio(str(tmp / "half"), [{STEREO_CHANNELS[1]: pcm}], SR, rows)
        assert n == 0, n

        # (d) fewer transcript rows than clips -> no IndexError; the wav still lands.
        n = _write_probe_audio(str(tmp / "short"), [clip, clip], SR, [])
        assert n == 2, n
    print("PASS  the audio dump absorbs every filesystem/data failure and never raises")


def check_dump_happens_after_scoring():
    """Source-level: the dump must come AFTER the result is built, not inside the scoring path.

    check (b)/(c) above prove the helper is safe; this proves the caller cannot have moved the call
    back up into the generate/score block, where a failure would once again cost the measurement.
    """
    import inspect

    from speech_llm.full_duplex.moshi_family import knowledge_probe

    src = inspect.getsource(knowledge_probe.run_knowledge_probe)
    i_score = src.index("res = ProbeResult(")
    i_dump = src.index("_write_probe_audio(")
    assert i_score < i_dump, (
        "the audio dump appears BEFORE the ProbeResult is assembled -- a filesystem failure would "
        "again be able to cost a probe step, which is the 2026-08-05 regression."
    )
    assert "out_dir" not in src, "out_dir is gone; the probe must not name a scratch directory"
    print("PASS  the audio dump runs only after scoring is complete in memory")


def check_coherence_separates_forgetting_from_incoherence():
    """``coherence_stats`` must actually discriminate the two failure modes it exists to tell apart.

    A guard that only asserted "returns a dict with these keys" would pass on a function that
    returned constants -- and constants are exactly what an incoherence metric must not be. So each
    metric is asserted to MOVE in the right direction between hand-built populations.
    """
    fluent_right = ["The capital of France is Paris, one of the oldest cities in Europe."] * 8
    fluent_wrong = ["The capital of France is Lyon, which sits on the Rhone river valley."] * 8
    empty = [""] * 8
    looping = ["I think I think I think I think I think it is"] * 8

    c_right, c_wrong = coherence_stats(fluent_right), coherence_stats(fluent_wrong)
    c_empty, c_loop = coherence_stats(empty), coherence_stats(looping)

    # FORGETTING: the reply is just as fluent, only the fact is gone. Every coherence metric must be
    # essentially unchanged -- this is the case the accuracy number alone cannot distinguish.
    assert abs(c_right["words_mean"] - c_wrong["words_mean"]) <= 1.0, (c_right, c_wrong)
    # The claim is not "identical" -- two different sentences differ a little by construction (the
    # right answer happens to repeat "the"). It is that the forgetting gap is small NEXT TO the
    # incoherence gap, so a threshold placed between them separates the two cases. Asserting both
    # sides keeps that comparative, rather than pinning a tolerance that means nothing on its own.
    forget_gap = abs(c_right["distinct_word_ratio"] - c_wrong["distinct_word_ratio"])
    assert forget_gap < 0.10, (forget_gap, c_right, c_wrong)
    assert c_right["empty_frac"] == c_wrong["empty_frac"] == 0.0, (c_right, c_wrong)
    assert c_right["looping_frac"] == c_wrong["looping_frac"] == 0.0, (c_right, c_wrong)

    # INCOHERENCE, mode 1: the model stops answering.
    assert c_empty["empty_frac"] == 1.0, c_empty
    assert c_empty["words_mean"] == 0.0, c_empty

    # INCOHERENCE, mode 2: the model loops. Note it is not SHORT -- a mean-length metric would call
    # this healthy, which is precisely why distinct_word_ratio and looping_frac exist.
    assert c_loop["looping_frac"] == 1.0, c_loop
    assert c_loop["words_mean"] >= c_right["words_mean"] * 0.5, (c_loop, c_right)
    loop_gap = abs(c_right["distinct_word_ratio"] - c_loop["distinct_word_ratio"])
    assert c_loop["distinct_word_ratio"] < c_right["distinct_word_ratio"] * 0.6, (c_loop, c_right)
    assert loop_gap > 3 * forget_gap, (
        f"forgetting moves distinct_word_ratio by {forget_gap:.3f} and looping by {loop_gap:.3f} -- "
        f"too close to place a threshold between them, so the metric cannot tell the two apart."
    )

    # A short reply must never be *called* a loop: below LOOP_MIN_WORDS the test is not meaningful.
    assert coherence_stats(["yes yes yes"])["looping_frac"] == 0.0

    # Degenerate input must not raise -- the probe runs this on whatever the model produced.
    assert coherence_stats([])["n"] == 0
    assert coherence_stats(["", "a b c"])["n"] == 2
    print("PASS  coherence_stats separates fluent-but-wrong from empty and from looping")


def check_transcripts_are_the_full_reply():
    """The probe's ``transcripts`` must carry the complete reply text, not a truncated preview.

    The whole reason this exists is that the old probe kept a 0/1 bit and discarded the reply, so
    every A-series run recorded that recall collapsed and nothing about how the output changed.
    Storing a truncated reply would reproduce that loss in a subtler form.
    """
    import inspect

    from speech_llm.full_duplex.moshi_family import knowledge_probe

    src = inspect.getsource(knowledge_probe.run_knowledge_probe)
    assert '"monologue": mono,' in src, (
        "the transcript row must store the whole monologue -- a slice here would silently cap "
        "every reply and the incoherence question would be unanswerable again."
    )
    r = ProbeResult(summary={"overall": {"accuracy": 0.0, "avg_quality": 1.0, "n": 0}})
    assert r.transcripts == [] and r.coherence == {} and r.audio_written == 0, r
    print("PASS  transcripts hold the complete reply and ProbeResult defaults are inert")


def check_stereo_dump_is_role_ordered_and_aligned():
    """The dump must put the QUESTION on channel 0 and the REPLY on channel 1, same length.

    This is the assertion CLAUDE.md's channel-seam rule exists for. Both orders produce a
    well-formed stereo file that plays; only the content says which one was written. So the two
    channels here are given *distinguishable* content and the file is read back and compared, rather
    than checking that a stereo file merely appeared.
    """
    import sphn

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        n = 3 * SR
        t = np.arange(n, dtype=np.float32)
        # The question: energy early, then EXACTLY silent for the last second -- which is what
        # run_pairs really produces (it pads the prompt with np.zeros over the capture window).
        user = (0.3 * np.sin(t * 0.02)).astype(np.float32)
        user[-SR:] = 0.0
        # The reply: lands in that trailing window, where the question is silent.
        assistant = np.zeros(n, dtype=np.float32)
        assistant[-SR:] = (0.3 * np.sin(t[-SR:] * 0.05)).astype(np.float32)

        rows = [{"question": "q", "answer": "a", "binary_correct": 1, "monologue": "reply"}]
        out = tmp / "st"
        got = _write_probe_audio(
            str(out),
            [{STEREO_CHANNELS[0]: user, STEREO_CHANNELS[1]: assistant}],
            SR,
            rows,
        )
        assert got == 1, got

        data, sr = sphn.read(str(out / "000.wav"))
        assert sr == SR, sr
        assert data.shape == (2, n), data.shape
        # Channel 0 IS the question and channel 1 IS the reply -- compared by content, so a swap
        # cannot pass. Tolerance is wav quantisation, not a fudge factor.
        assert np.max(np.abs(data[0] - user)) < 1e-4, "channel 0 is not the question"
        assert np.max(np.abs(data[1] - assistant)) < 1e-4, "channel 1 is not the reply"
        # ...and they really are distinguishable, or the two assertions above are vacuous.
        assert np.max(np.abs(user - assistant)) > 0.1, "the fixture channels are too alike"
    print("PASS  the stereo dump puts the question on ch0 and the reply on ch1, sample-aligned")


def check_a_swapped_channel_map_is_detected():
    """The swap detector must FIRE on a swapped clip and stay silent on a correct one.

    A detector that never fires is indistinguishable from a correct channel map, so both directions
    are asserted -- the failure this guards against is silent by construction.
    """
    n = 3 * SR
    t = np.arange(n, dtype=np.float32)
    user = (0.3 * np.sin(t * 0.02)).astype(np.float32)
    user[-SR:] = 0.0
    assistant = np.zeros(n, dtype=np.float32)
    assistant[-SR:] = (0.3 * np.sin(t[-SR:] * 0.05)).astype(np.float32)

    assert not channels_look_swapped(user, SR), "a correct channel map was flagged as swapped"
    assert channels_look_swapped(assistant, SR), (
        "the reply was accepted as the question channel -- the detector is a no-op, and a swapped "
        "dump would ship looking exactly like a correct one"
    )
    # A clip shorter than the tail window cannot be judged and must not be guessed at.
    short = np.ones(int(CHANNEL_CHECK_TAIL_S * SR) // 2, dtype=np.float32)
    assert not channels_look_swapped(short, SR), "a too-short clip must not be flagged"

    # ...and the writer actually warns, rather than the detector being computed and dropped.
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as d:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _write_probe_audio(
                os.path.join(d, "swapped"),
                [{STEREO_CHANNELS[0]: assistant, STEREO_CHANNELS[1]: user}],
                SR,
                [],
            )
        assert "probably swapped" in buf.getvalue(), (
            f"the writer swallowed a swapped channel map silently: {buf.getvalue()!r}"
        )
    print("PASS  a swapped channel map is detected and warned about, a correct one is not")


if __name__ == "__main__":
    check_disk_mode_still_writes()
    check_memory_mode_writes_nothing()
    check_truncation_accounting_survives_without_a_directory()
    check_probe_scores_positionally_against_its_own_entries()
    check_keep_audio_returns_pcm_without_writing()
    check_audio_dump_cannot_break_a_probe()
    check_dump_happens_after_scoring()
    check_coherence_separates_forgetting_from_incoherence()
    check_transcripts_are_the_full_reply()
    check_stereo_dump_is_role_ordered_and_aligned()
    check_a_swapped_channel_map_is_detected()
    print("ALL PASS")
