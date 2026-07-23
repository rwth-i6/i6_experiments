"""Fisher (fe_03) corpus prep: LDC 2-channel ``.sph`` -> split mono wavs.

LDC ships Fisher English Training Speech as **2-channel** ``.sph`` with coding
``ulaw,embedded-shorten-v2.00`` (NIST SPHERE, µ-law compressed with embedded shorten): Part 1 =
``LDC2004S13``, Part 2 = ``LDC2005S13``. The Lhotse manifest builder
``datasets/cutset_jobs.create_fisher_manifest`` wants **per-channel mono wavs** named
``fe_03_{NNNNN}A.wav`` (channel 1 -> user) and ``fe_03_{NNNNN}B.wav`` (channel 2 -> agent) in one
flat directory. This job does that split.

Decoder: **sph2pipe** (the NIST/LDC tool), built here from OpenSLR source. **ffmpeg does NOT work** --
ffmpeg 8.1's nistsphere demuxer reports ``coding ulaw,embedded-shorten-v2.00 is not implemented``
(``Audio: none ... unknown codec``) even though it has standalone ``shorten``/``pcm_mulaw`` decoders;
the embedded-shorten-in-µ-law SPHERE variant is sph2pipe's job. sph2pipe links only libc/libm, so
(unlike the ffmpeg build) it has no missing-shared-lib portability issue across partitions.

Wire-up (after the rsync of raw ``.sph`` into ``/hpcwork/tt201262/corpora/fisher/LDC200{4,5}S13`` and
the transcripts into ``.../LDC200{4,5}T19``):

    sp = InstallSph2pipe().out_path
    audio = FisherSphToWav(
        sph_dirs=[tk.Path(".../LDC2004S13"), tk.Path(".../LDC2005S13")],
        sph2pipe_path=sp,
    ).out_audio
    # -> feed `audio` as fisher_recording_dir to PrepareFisherDatasetJob (datasets/cutset_jobs.py).
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sisyphus import Job, Task, tk

SPH2PIPE_URL = "https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz"


class InstallSph2pipe(Job):
    """Build the NIST ``sph2pipe`` decoder from OpenSLR source; expose the binary at ``out/sph2pipe``.

    Tiny single-binary C program (links libc/libm only) -- no autotools, no shared-lib portability
    problems. ``sph2pipe -p -f wav -c {1,2} in.sph out.wav`` decodes ulaw+embedded-shorten -> 16-bit
    PCM wav for one channel.
    """

    def __init__(self):
        self.out_path = self.output_path("out", directory=True)
        self.rqmt = {"cpu": 2, "mem": 2, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        subprocess.run(["wget", "-q", SPH2PIPE_URL], check=True)
        subprocess.run(["tar", "-xf", "sph2pipe_v2.5.tar.gz"], check=True)
        os.chdir("sph2pipe_v2.5")
        # Documented build: one gcc over the .c files, link math. No Makefile deps needed.
        subprocess.run("gcc -o sph2pipe *.c -lm", shell=True, check=True)
        subprocess.run(["./sph2pipe", "-h"], check=False)  # smoke: binary runs
        dst = Path(self.out_path.get_path()) / "sph2pipe"
        shutil.copy("sph2pipe", dst)
        os.chmod(dst, 0o755)
        print(f"sph2pipe installed to {dst}", flush=True)


class FisherSphToWav(Job):
    """Split every Fisher ``fe_03_{NNNNN}.sph`` (2ch) into ``{stem}A.wav`` + ``{stem}B.wav`` (mono).

    ``sph_dirs`` are the extracted LDC speech trees (rglob'd for ``*.sph``, DVD nesting is fine).
    sph2pipe decodes ulaw+embedded-shorten -> 16-bit PCM and extracts one channel per call; native
    8 kHz is preserved (mimi resamples at train time). A/B naming + flat output match exactly what
    ``create_fisher_manifest`` globs. Parallelised across ``rqmt['cpu']`` sph2pipe subprocesses.
    """

    def __init__(
        self,
        *,
        sph_dirs: list[tk.Path],
        sph2pipe_path: tk.Path,
        shard: int | None = None,
        num_shards: int | None = None,
    ):
        self.sph_dirs = sph_dirs
        self.sph2pipe_path = sph2pipe_path
        self.shard = shard
        self.num_shards = num_shards
        self.out_audio = self.output_path("audio", directory=True)
        self.rqmt = {"cpu": 16, "mem": 8, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from i6_experiments.users.dorian_koch.speech_llm.common import job_progress_fraction

        return job_progress_fraction(self)

    def _list_sphs(self) -> list[str]:
        sphs: list[str] = []
        for d in self.sph_dirs:
            root = d.get_path()
            for pat in ("**/*.sph", "**/*.SPH"):
                sphs.extend(glob.glob(os.path.join(root, pat), recursive=True))
        sphs = sorted(set(sphs))
        if self.num_shards is not None:
            sphs = sphs[self.shard :: self.num_shards]
        return sphs

    def run(self):
        sph2pipe = os.path.join(self.sph2pipe_path.get_path(), "sph2pipe")
        out = Path(self.out_audio.get_path())
        sphs = self._list_sphs()
        total = len(sphs)
        assert total > 0, f"no .sph files found under {[d.get_path() for d in self.sph_dirs]}"
        print(f"[fisher-sph2wav] {total} .sph files to split with {sph2pipe}", flush=True)

        def convert(sph: str) -> tuple[str, str | None]:
            stem = Path(sph).stem  # e.g. fe_03_00001
            a, b = out / f"{stem}A.wav", out / f"{stem}B.wav"
            if a.exists() and b.exists():
                return sph, None
            try:
                for ch, dst in ((1, a), (2, b)):
                    subprocess.run(
                        [sph2pipe, "-p", "-f", "wav", "-c", str(ch), sph, str(dst)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                return sph, None
            except subprocess.CalledProcessError as e:
                return sph, (e.stderr or b"").decode("utf-8", "replace")[-300:]

        done = 0
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=self.rqmt["cpu"]) as ex:
            futs = {ex.submit(convert, s): s for s in sphs}
            for fut in as_completed(futs):
                sph, err = fut.result()
                done += 1
                if err is not None:
                    failures.append(sph)
                    print(f"[fisher-sph2wav] FAILED {Path(sph).name}: {err}", flush=True)
                # Early abort: if the first batch is ALL failures, the decode is broken -- bail in
                # seconds rather than grinding through all 11.7k files and flooding the log.
                if done >= 32 and len(failures) == done:
                    raise RuntimeError(
                        f"[fisher-sph2wav] first {done} conversions all failed -- sph2pipe decode "
                        f"broken; last error: {err}"
                    )
                if done % 200 == 0 or done == total:
                    print(f"[fisher-sph2wav] {done}/{total} ({100 * done / total:.1f}%)", flush=True)
                    try:
                        Path("progress.json").write_text(json.dumps({"done": done, "total": total}))
                    except OSError:
                        pass

        # Fail loud: a partial corpus silently poisons every downstream training read.
        assert not failures, (
            f"[fisher-sph2wav] {len(failures)}/{total} conversions failed "
            f"(first few: {failures[:5]}) -- inspect the SPHERE/shorten decode"
        )
        n_wavs = sum(1 for _ in out.glob("*.wav"))
        print(f"[fisher-sph2wav] done: {total} sph -> {n_wavs} wavs (expected {2 * total})", flush=True)
        assert n_wavs == 2 * total, f"expected {2 * total} wavs, got {n_wavs}"


# --------------------------------------------------------------------------------------------
# Fisher -> duplex training windows
# --------------------------------------------------------------------------------------------

# --- role naming ---------------------------------------------------------------------------
# The same two speakers carry DIFFERENT role labels on either side of this job, so keep them as
# named constants rather than bare strings -- silently mixing them up is the single easiest way to
# train the model on the wrong channel.
#
#   Fisher / CutSet side          moshi loader side
#   --------------------          -----------------
#   channel A = "user"    ------> `audio_user`      (the interlocutor; conditioning only)
#   channel B = "agent"   ------> `audio_assistant` (what the model learns to produce)
#                                 alignments[*]["speaker"] == "assistant"
#
# The A->user / B->agent mapping is fixed upstream by `cutset_jobs.create_fisher_manifest`; every
# supervision also carries `recording_id == f"{conv}_{user|agent}"`, which `_plan_windows` asserts
# against so this convention can never drift silently.
CUTSET_AGENT_SPEAKER = "agent"  # role label inside the Fisher CutSet supervisions (channel B)
MONOLOGUE_SPEAKER = "assistant"  # role label the moshi loader expects on alignment entries

# Fisher transcript conventions that must NOT reach the text stream:
#   [laughter] [noise] [mn] [sigh] [lipsmack]  -- non-speech events
#   (( ... ))                                  -- uncertain transcription ("(( ))" = unintelligible)
#   t._v.                                      -- underscore-joined spelled abbreviations
_NONSPEECH_RE = re.compile(r"\[[^\]]*\]")
_UNCERTAIN_RE = re.compile(r"\(\(|\)\)")


def _clean_fisher_text(text: str) -> str:
    """Strip Fisher markup to plain spoken words.

    Non-speech events are dropped outright; the ``(( ))`` uncertainty *markers* are dropped but the
    words inside them are kept (they are still speech, just flagged low-confidence).
    """
    t = _NONSPEECH_RE.sub(" ", text or "")
    t = _UNCERTAIN_RE.sub(" ", t)
    t = t.replace("_", "")  # "t._v." -> "t.v."
    return " ".join(t.split())


def _overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds of overlap between the intervals [a_start, a_end) and [b_start, b_end)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


class FisherToMoshiTrainData(Job):
    """Slice the Fisher CutSet into fixed-length duplex training windows for Moshi finetuning.

    Fisher is *real* 2-channel conversational telephone speech (A -> user, B -> agent), so unlike the
    synthetic TriviaQA pipeline it needs no TTS and no forced-alignment pass: the two channels already
    ARE the duplex tracks, and the LDC transcripts already supply timed, speaker-labelled text. This
    job just cuts conversations into windows and emits the **same arrow layout** that
    ``moshi_family.moshi_train_data.build_data_loader`` reads -- ``audio_assistant`` / ``audio_user``
    raw wav bytes plus word-level ``alignments`` -- so the result drops into
    ``SpeechFinetune(train_data=...)`` with no loader changes.

    Four things here are load-bearing; each is a trap we hit or would have hit:

    1. **Only the AGENT's words become ``alignments``.** The text stream is Moshi's *inner monologue*
       -- what the assistant itself says. ``_interleave_text`` places every alignment it is given and
       **ignores the speaker field**, so leaving the user's words in would train the model to speak the
       user's lines. (The synthetic path gets this for free: its annotate job runs
       ``annotate_array(stereo[0])``, i.e. the assistant channel only.)
    2. **Word timings are interpolated.** Fisher timestamps are utterance-level, not word-level, so a
       utterance's words are spread evenly across its span. This is a faithful approximation for our
       purposes because ``_interleave_text`` only ever uses each word's *onset* frame.
    3. **Audio is stored as plain ``{bytes, path}`` structs, NOT an HF ``Audio()`` feature.** Encoding
       ``Audio()`` requires torchcodec+FFmpeg, which does not load on the login node or c25g; the
       loader only does ``.as_py()["bytes"]`` -> ``soundfile``, so a plain binary struct is both
       sufficient and portable. Native 8 kHz is preserved (mimi resamples at train time).
    4. **Windows must contain real agent speech** (``min_agent_speech_sec``). A window where the agent
       is mostly silent is ~all text-pad, which is exactly the pad-collapse failure mode the finetune
       degradation investigation is chasing -- feeding those in would make the problem worse.

    Diversity: windows are taken round-robin over *conversations* (``windows_per_conv`` each, from a
    seeded shuffle) rather than exhaustively per conversation, so a capped run spans many speakers.
    """

    def __init__(
        self,
        *,
        cutset: tk.Path,
        wav_dir: tk.Path,
        duration_sec: float = 40.0,
        stride_sec: float | None = None,
        min_agent_speech_sec: float = 4.0,
        min_agent_words: int = 8,
        windows_per_conv: int = 2,
        max_windows: int | None = None,
        seed: int = 0,
    ):
        self.cutset = cutset
        self.wav_dir = wav_dir
        self.duration_sec = duration_sec
        self.stride_sec = stride_sec if stride_sec is not None else duration_sec
        self.min_agent_speech_sec = min_agent_speech_sec
        self.min_agent_words = min_agent_words
        self.windows_per_conv = windows_per_conv
        self.max_windows = max_windows
        self.seed = seed
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 8, "mem": 16, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from i6_experiments.users.dorian_koch.speech_llm.common import job_progress_fraction

        return job_progress_fraction(self)

    # -- window planning (manifest only; no audio IO) ------------------------------------------

    def _plan_windows(self) -> tuple[list[dict], dict]:
        """Decide which windows to cut, reading only the manifest (no audio IO -- that is the
        expensive half and happens later, in parallel, in :meth:`_build_row`).

        :return: ``(windows, stats)`` where each window is
            ``{"conv": <cut id>, "start": <offset into the conversation, sec>,
               "alignments": <the agent's words in that span, relative to its start>}``.
        """
        import random

        # One JSON object per line, each a lhotse MonoCut for a whole ~10 min conversation.
        cuts = []
        with open(self.cutset.get_path()) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cuts.append(json.loads(line))
        assert cuts, f"empty Fisher cutset at {self.cutset.get_path()}"

        # Shuffle so that a `max_windows`-capped run draws from across the whole corpus (many
        # speakers/topics) instead of the first N conversations in file order.
        rng = random.Random(self.seed)
        rng.shuffle(cuts)

        window_sec = self.duration_sec
        windows: list[dict] = []
        stats = {
            # `conversations_total` is how many the corpus HAS; `conversations_used` is how many
            # actually contributed a window. They differ a lot under `max_windows`, and conflating
            # them badly overstates the corpus diversity in the run report.
            "conversations_total": len(cuts),
            "conversations_used": 0,
            "rejected_quiet": 0,  # windows dropped: agent barely speaks in them
            "rejected_short_text": 0,  # windows dropped: too few agent words after markup cleanup
        }

        for cut in cuts:
            if self.max_windows is not None and len(windows) >= self.max_windows:
                break
            agent_supervisions = [
                sup for sup in cut.get("supervisions", []) if sup.get("speaker") == CUTSET_AGENT_SPEAKER
            ]
            if not agent_supervisions:
                continue
            # Guard the role-naming contract at its seam: the speaker label we filter on must agree
            # with the recording the supervision came from, so a future change to either side of the
            # A/B <-> user/agent mapping fails loudly here instead of quietly training the model on
            # the interlocutor's channel.
            for sup in agent_supervisions:
                rec_id = sup.get("recording_id", "")
                assert rec_id.endswith(f"_{CUTSET_AGENT_SPEAKER}"), (
                    f"supervision {sup.get('id')!r} is labelled speaker="
                    f"{CUTSET_AGENT_SPEAKER!r} but belongs to recording {rec_id!r} -- the "
                    f"Fisher channel<->role mapping changed upstream"
                )

            conv_duration_sec = float(cut.get("duration", 0.0))
            windows_from_this_conv = 0
            win_start = 0.0
            while win_start + window_sec <= conv_duration_sec and windows_from_this_conv < self.windows_per_conv:
                if self.max_windows is not None and len(windows) >= self.max_windows:
                    break
                win_end = win_start + window_sec

                # Reject near-silent windows: with almost no agent speech the target is nearly all
                # text-pad, which feeds exactly the pad-collapse failure mode we are investigating.
                agent_speech_sec = sum(
                    _overlap_sec(sup["start"], sup["start"] + sup["duration"], win_start, win_end)
                    for sup in agent_supervisions
                )
                if agent_speech_sec < self.min_agent_speech_sec:
                    stats["rejected_quiet"] += 1
                    win_start += self.stride_sec
                    continue

                word_alignments = self._agent_word_alignments(agent_supervisions, win_start, win_end)
                if len(word_alignments) < self.min_agent_words:
                    stats["rejected_short_text"] += 1
                    win_start += self.stride_sec
                    continue

                windows.append({"conv": cut["id"], "start": win_start, "alignments": word_alignments})
                windows_from_this_conv += 1
                win_start += self.stride_sec

            if windows_from_this_conv:
                stats["conversations_used"] += 1

        stats["windows"] = len(windows)
        return windows, stats

    def _agent_word_alignments(self, agent_supervisions: list[dict], win_start: float, win_end: float) -> list[dict]:
        """The agent's words whose onset falls in ``[win_start, win_end)``, re-timed relative to
        ``win_start`` and labelled with :data:`MONOLOGUE_SPEAKER`.

        Deliberately agent-only: these entries become Moshi's inner-monologue text stream, and the
        loader's ``_interleave_text`` places every entry it is handed **without looking at the
        speaker field** -- so including the user's words here would teach the model to speak the
        user's lines.
        """
        word_alignments: list[dict] = []
        for sup in agent_supervisions:
            sup_start, sup_dur = float(sup["start"]), float(sup["duration"])
            if sup_start + sup_dur <= win_start or sup_start >= win_end or sup_dur <= 0.0:
                continue
            words = _clean_fisher_text(sup.get("text", "")).split()
            if not words:
                continue
            # Fisher timestamps are utterance-level, so spread the utterance's words evenly across
            # its span. Approximate, but adequate: the loader only ever uses each word's ONSET.
            sec_per_word = sup_dur / len(words)
            for word_index, word in enumerate(words):
                word_start = sup_start + word_index * sec_per_word
                if word_start < win_start or word_start >= win_end:
                    continue
                word_alignments.append(
                    {
                        "text": word,
                        "start": word_start - win_start,
                        "end": min(word_start + sec_per_word, win_end) - win_start,
                        "speaker": MONOLOGUE_SPEAKER,
                    }
                )
        # Utterances are per-speaker-turn and already ordered, but sort anyway: the loader assumes
        # monotonically non-decreasing onsets when it lays words onto the text stream.
        word_alignments.sort(key=lambda entry: entry["start"])
        return word_alignments

    # -- audio slicing ------------------------------------------------------------------------

    def _channel_wav_paths(self, conv_id: str) -> dict[str, str]:
        """Map a cut id to its two per-channel wavs, **keyed by role rather than by position**.

        Positional returns are how channel swaps happen: ``a, b = f(...)`` reads identically whether
        it is (user, agent) or (agent, user), and getting it backwards trains the model to speak the
        interlocutor's side -- a mistake that produces a plausible-looking loss curve and is only
        visible by listening. Keying by role makes the call site self-checking.

        ``"fe00001"`` -> ``{"user": ".../fe_03_00001A.wav", "agent": ".../fe_03_00001B.wav"}``.
        """
        assert conv_id.startswith("fe"), f"unexpected Fisher cut id {conv_id!r}"
        conv_number = conv_id[2:]
        wav_root = self.wav_dir.get_path()
        return {
            "user": os.path.join(wav_root, f"fe_03_{conv_number}A.wav"),  # channel A
            "agent": os.path.join(wav_root, f"fe_03_{conv_number}B.wav"),  # channel B
        }

    def _slice_window_wav(self, wav_path: str, window_start_sec: float) -> bytes:
        """Cut ``duration_sec`` starting at ``window_start_sec`` out of one channel's wav and
        re-encode it as 16-bit PCM wav bytes at the source rate (native 8 kHz is preserved; mimi
        resamples at train time)."""
        import numpy as np
        import soundfile as sf

        with sf.SoundFile(wav_path) as snd:
            sample_rate = snd.samplerate
            num_samples_wanted = int(round(self.duration_sec * sample_rate))
            snd.seek(int(round(window_start_sec * sample_rate)))
            samples = snd.read(num_samples_wanted, dtype="int16", always_2d=False)
        if samples.ndim > 1:  # per-channel files should be mono; take channel 0 if ever not
            samples = samples[:, 0]
        if samples.shape[0] < num_samples_wanted:
            # Short tail at the end of a conversation: zero-pad so both channels stay sample-aligned
            # with each other and with the alignment timestamps.
            samples = np.pad(samples, (0, num_samples_wanted - samples.shape[0]))
        wav_buf = io.BytesIO()
        sf.write(wav_buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return wav_buf.getvalue()

    def _build_row(self, window: dict) -> dict:
        """Turn one planned window into an arrow row in the layout `build_data_loader` reads."""
        wav_paths = self._channel_wav_paths(window["conv"])
        window_start_sec = window["start"]
        # Named per role, then assigned per loader column -- so the agent->assistant and
        # user->user mapping is spelled out rather than implied by argument order.
        agent_wav = self._slice_window_wav(wav_paths["agent"], window_start_sec)
        user_wav = self._slice_window_wav(wav_paths["user"], window_start_sec)
        # Both channels of a Fisher conversation are the same length (asserted upstream in
        # `create_fisher_manifest`), so equal slice sizes confirm we read the same span of both.
        assert len(agent_wav) == len(user_wav), (
            f"channel slices differ in size for {window['conv']} @ {window_start_sec}s: "
            f"agent {len(agent_wav)} B vs user {len(user_wav)} B"
        )
        return {
            "id": f"{window['conv']}_{window_start_sec:.1f}",
            "duration": float(self.duration_sec),
            # channel B (agent) is the track the model is trained to produce ...
            "audio_assistant": {"bytes": agent_wav, "path": ""},
            # ... channel A (user) is the interlocutor, conditioning input only.
            "audio_user": {"bytes": user_wav, "path": ""},
            "alignments": window["alignments"],
        }

    # -- run ----------------------------------------------------------------------------------

    def run(self):
        from datasets import Dataset, Features, Value

        windows, stats = self._plan_windows()
        num_windows = len(windows)
        assert num_windows > 0, (
            "[fisher-windows] no qualifying windows -- check min_agent_speech_sec / "
            "min_agent_words against the transcript coverage"
        )
        print(
            f"[fisher-windows] {num_windows} windows of {self.duration_sec}s from "
            f"{stats['conversations_used']}/{stats['conversations_total']} conversations "
            f"(rejected: {stats['rejected_quiet']} quiet, {stats['rejected_short_text']} short-text)",
            flush=True,
        )

        features = Features(
            {
                "id": Value("string"),
                "duration": Value("float32"),
                "audio_assistant": {"bytes": Value("binary"), "path": Value("string")},
                "audio_user": {"bytes": Value("binary"), "path": Value("string")},
                "alignments": [
                    {
                        "text": Value("string"),
                        "start": Value("float32"),
                        "end": Value("float32"),
                        "speaker": Value("string"),
                    }
                ],
            }
        )

        def generate_rows():
            """Yield rows in planned order, slicing audio in bounded parallel batches.

            Batching (rather than one big `pool.map`) keeps peak memory at roughly
            ``batch_size * 2 channels * duration_sec * 8 kHz * 2 B`` -- ~80 MB at the defaults --
            instead of materialising all 20k windows of PCM at once.
            """
            batch_size = 64
            with ThreadPoolExecutor(max_workers=self.rqmt["cpu"]) as pool:
                for batch_start in range(0, num_windows, batch_size):
                    batch = windows[batch_start : batch_start + batch_size]
                    for row in pool.map(self._build_row, batch):
                        yield row
                    num_done = min(batch_start + batch_size, num_windows)
                    # Heartbeat every ~512 rows: enough to tell progress from a hang, not so much
                    # that it floods the log.
                    if num_done % 512 == 0 or num_done == num_windows:
                        print(
                            f"[fisher-windows] {num_done}/{num_windows} ({100 * num_done / num_windows:.1f}%)",
                            flush=True,
                        )
                        try:
                            # Read back by `completed_fraction` so the manager shows a live %.
                            Path("progress.json").write_text(json.dumps({"done": num_done, "total": num_windows}))
                        except OSError:
                            pass

        arrow_cache_dir = os.path.join(os.getcwd(), "hf_cache")
        dataset = Dataset.from_generator(
            generate_rows, features=features, cache_dir=arrow_cache_dir, writer_batch_size=32
        )
        dataset.save_to_disk(self.out_dataset.get_path())

        # Fail loud rather than hand a silently-truncated corpus to a multi-hour finetune.
        assert dataset.num_rows == num_windows, f"wrote {dataset.num_rows} rows, planned {num_windows}"
        stats["rows"] = dataset.num_rows
        stats["mean_words_per_window"] = round(sum(len(window["alignments"]) for window in windows) / num_windows, 2)
        Path(self.out_stats.get_path()).write_text(json.dumps(stats, indent=2))
        print(f"[fisher-windows] done: {stats}", flush=True)
        shutil.rmtree(arrow_cache_dir, ignore_errors=True)
