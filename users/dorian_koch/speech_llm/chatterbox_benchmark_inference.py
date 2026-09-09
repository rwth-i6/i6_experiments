"""
Single-speaker TTS inference for knowledge benchmark questions.

Used by ChatterboxSingleSpeakerInference job via subprocess
(to run inside the chatterbox venv).
"""

import argparse
import gc
import os
import random

import numpy as np
import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS
from datasets import Dataset, load_from_disk

# NOTE: deliberately duplicated from speech_llm/clip_store.py -- see the note in
# chatterbox_inference.py. This runs under the chatterbox job venv, which has neither sisyphus nor
# i6_experiments importable (clip_store is fine on its own, but lives next to common.py whose first
# line is `from sisyphus import tk`). Keep the column names identical to clip_store's.
COL_INDEX, COL_AUDIO, COL_SR = "index", "audio", "sampling_rate"
COL_MONOLOGUE, COL_TRACE = "monologue", "trace_json"


class ClipSpool:
    """Append-only float32 spool: clips go to DISK as they are made, never accumulating in memory.

    The shape this replaces held every clip in a list and handed the whole set to
    ``Dataset.from_dict``, which copies it again into arrow -- three live copies of the corpus. On
    the 12,000-prompt rehearsal corpus that was 12.6 GB steady-state and 41.8 GB in the final
    fifteen seconds, and it OOM-killed the job three times (2026-09-07/08/09) *after* generating all
    12,000 clips. Each time the response was to raise ``rqmt["mem"]``, which moves the ceiling by one
    doubling and never removes it, because peak scaled with the corpus. Peak is now ONE clip, so the
    corpus size is bounded by disk rather than by a memory request.

    Two details are load-bearing:

    * **One file, not one per clip.** A file per clip would cost one transient inode per prompt on
      ``/hpcwork/p0023999``, whose binding limit is inodes -- which is the entire reason
      ``storage="hf"`` exists. Clips are appended to a single raw float32 file and addressed by
      ``(offset, count)``, so the spool is 1 inode at any corpus size.
    * **Raw float32, not wav.** No audio library and no 16-bit round trip, so what comes back out is
      bit-identical to what the TTS produced and the rewrite cannot alter the corpus.

    Node-local ``/tmp`` is deliberately NOT used: it is tmpfs on some nodes, which would put the
    spool back into RAM and reintroduce this bug in a form that only appears on those nodes.
    """

    def __init__(self, path):
        self.path = path
        self.spans = {}  # clip index -> (offset in samples, sample count)
        self._n = 0
        self._fh = open(path, "wb")

    def append(self, index, samples):
        """Spool one clip. Nothing is retained after this returns."""
        index = int(index)
        if index in self.spans:
            # Same rule as clip_store: downstream joins address clips BY INDEX, so a duplicate
            # silently drops one clip and misaligns every row after it. Enforced here at write
            # time, which is stricter than the post-hoc `seen` set it replaces.
            raise ValueError(f"duplicate clip index {index} -- downstream joins address clips by index")
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._fh.write(arr.tobytes())
        self.spans[index] = (self._n, int(arr.size))
        self._n += int(arr.size)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def clip_features():
    """The arrow schema, DERIVED from the ``Dataset.from_dict`` call the streaming write replaces.

    Inferred from one dummy row rather than written out by hand on purpose: a hand-written schema is
    exactly how this rewrite would silently change the corpus dtype, and no reader would complain.
    """
    return Dataset.from_dict(
        {
            COL_INDEX: [0],
            COL_AUDIO: [np.zeros(1, dtype=np.float32)],
            COL_SR: [0],
            COL_MONOLOGUE: [None],
            COL_TRACE: [None],
        }
    ).features


def write_clips(out_path, spool, sample_rate):
    """Stream a :class:`ClipSpool` into one arrow dataset. Mirrors clip_store's columns.

    The generator closes over plain data (a path, a span dict, an int) and nothing else: ``datasets``
    hashes the callable with dill for its cache, so a closure over a live generator or an open file
    handle dies here with ``TypeError: cannot pickle`` (measured, not hypothetical).
    """
    spool.close()
    spans, path = spool.spans, spool.path

    def gen():
        for i in sorted(spans):
            offset, count = spans[i]
            yield {
                COL_INDEX: i,
                COL_AUDIO: np.fromfile(path, dtype=np.float32, count=count, offset=offset * 4),
                COL_SR: int(sample_rate),
                COL_MONOLOGUE: None,
                COL_TRACE: None,
            }

    Dataset.from_generator(gen, features=clip_features()).save_to_disk(out_path)


#: Base seed for both the speaker draw and the per-clip TTS sampling. Changing it regenerates every
#: corpus this worker produces, so treat it as a version, not a knob.
SEED = 42


#: An ``rng_`` speaker is ONE draw from ``SEED`` over ``os.listdir`` order -- so which of the 128
#: user voices the benchmark asks its questions in has been a property of the FILESYSTEM, not of the
#: config. It happens to be stable on this cluster, which is why it was never noticed; the listing is
#: verifiably not in sorted order, so it is stable by luck.
#:
#: Sorting the listing is the obvious fix and is the wrong one HERE: with ``sorted()`` the same seed
#: draws ``prompt_12_prompt_0_voice_4`` instead, i.e. it would silently re-voice the benchmark and
#: make every existing knowledge number non-comparable with anything measured afterwards. So record
#: the voice that was actually used instead. This changes no past result and makes it reproducible
#: rather than incidental. `speaker_name` IS hashed, so the pin cannot live in the recipe without
#: re-hashing (and re-running) every benchmark that has ever been scored.
#:
#: Verified 2026-09-07 against the log of a finished ChatterboxSingleSpeakerInference:
#:     Using speaker: .../user_voices/prompt_9_prompt_0_voice_7.wav
PINNED_RNG_SPEAKERS = {"user_voices/rng_a": "prompt_9_prompt_0_voice_7.wav"}


def resolve_speaker_path(speaker_dir: str, speaker_name: str) -> str:
    """Resolve speaker path, supporting rng_ prefix for random selection."""
    based = os.path.basename(speaker_name)
    if based.startswith("rng_"):
        dirname = os.path.dirname(speaker_name)
        search_dir = os.path.join(speaker_dir, dirname) if dirname else speaker_dir
        pinned = PINNED_RNG_SPEAKERS.get(speaker_name)
        if pinned is not None:
            path = os.path.join(search_dir, pinned)
            assert os.path.exists(path), (
                f"pinned speaker {path} is missing. It is the voice every knowledge benchmark to "
                f"date was measured in -- falling back to a fresh draw would re-voice the benchmark "
                f"silently, so refuse instead."
            )
            return path
        # sorted(): an unsorted os.listdir makes the seeded draw depend on filesystem order, which
        # is the bug this whole block exists for. The corpus worker (chatterbox_inference.py) sorts
        # for exactly this reason; this copy did not.
        wavs = sorted(f for f in os.listdir(search_dir) if f.endswith(".wav"))
        return os.path.join(search_dir, random.choice(wavs))
    return os.path.join(speaker_dir, speaker_name + ".wav")


def main():
    parser = argparse.ArgumentParser(description="Single-speaker TTS for benchmark questions")
    parser.add_argument("--in_hf", required=True, help="Path to input HF dataset")
    parser.add_argument("--speaker_dir", required=True, help="Directory containing speaker wavs")
    parser.add_argument("--speaker_name", default="user_voices/rng_a", help="Speaker name (supports rng_ prefix)")
    parser.add_argument("--out_dir", required=True, help="Output directory for wav files / arrow dataset")
    parser.add_argument(
        "--storage",
        default="wav",
        choices=("wav", "hf"),
        help="wav: one <i>.wav per question (original). hf: one arrow dataset (~3 inodes total).",
    )
    args = parser.parse_args()

    random.seed(SEED)  # speaker choice
    ds = load_from_disk(args.in_hf)

    speaker_path = resolve_speaker_path(args.speaker_dir, args.speaker_name)
    print(f"Using speaker: {speaker_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")
    model.t3.inference_turbo = torch.compile(model.t3.inference_turbo, dynamic=True)

    model.prepare_conditionals(speaker_path, exaggeration=0.5, norm_loudness=True)

    os.makedirs(args.out_dir, exist_ok=True)
    # storage="hf" writes ONE arrow dataset instead of a wav per question. A 1000-question benchmark
    # drops from ~1000 inodes to ~3, which is what the shared /hpcwork project volume actually runs
    # out of. Clips are spooled to disk as they are produced (see ClipSpool) rather than held, so
    # this scales to a corpus without scaling rqmt["mem"].
    spool = ClipSpool(args.out_dir.rstrip("/") + ".spool") if args.storage == "hf" else None
    with torch.inference_mode():
        for i, example in enumerate(ds):
            # Seed PER CLIP, from the clip's index -- not once per run. `model.generate` is a
            # SAMPLING TTS, so without a torch seed the benchmark's questions are different audio on
            # every regeneration: measured 2026-08-03, two runs of this job on the same subsample
            # gave 61 of 64 clips differing in LENGTH (worst per-sample diff 0.725, where a PCM-16
            # step is 6.1e-5). That put an unmeasured noise floor under every knowledge number and
            # made a storage-format A/B uninterpretable.
            #
            # Index-derived rather than run-level so a clip's audio does not depend on iteration
            # order: a shard, a resumed run, or a re-run of a subset all reproduce the same audio
            # for the same clip.
            torch.manual_seed(SEED + i)
            wav = model.generate(text=example["question"], audio_prompt_path=None)
            if args.storage == "hf":
                spool.append(i, wav.cpu().numpy().reshape(-1))
            else:
                torchaudio.save(os.path.join(args.out_dir, f"{i}.wav"), wav.cpu(), model.sr)
            del wav
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Processed {i + 1}/{len(ds)} examples", flush=True)

    if args.storage == "hf":
        n_clips = len(spool.spans)
        assert n_clips == len(ds), f"spooled {n_clips} clips but the input has {len(ds)} rows"
        write_clips(args.out_dir, spool, model.sr)
        os.remove(spool.path)
        print(f"Done. Wrote {n_clips} clips as an arrow dataset in {args.out_dir}")
    else:
        print(f"Done. Generated {len(ds)} audio files in {args.out_dir}")


if __name__ == "__main__":
    main()
