"""
moshi_arrow_dataset.py — HuggingFace-arrow data loader for moshi-finetune.

Reads an annotated arrow dataset (output of MoshiAnnotate in arrow mode) and
yields moshi-finetune Sample objects compatible with the existing training loop.

We reuse the fork's classes entirely; this module just provides the alternative
data path — no fork files are edited.

Columns expected in the HF dataset:
    audio       dict {"array": float32 array (T, 2), "sampling_rate": int}
    alignments  list of [text, [start_s, end_s], "SPEAKER_MAIN"]
    duration    float  (seconds)
    (any extra columns are ignored)

The loader mirrors the signature of finetune.data.data_loader.build_data_loader
so it can be installed as a drop-in replacement.

Data/augmentation knobs (e.g. leading-silence jitter) come from an
``ArrowDataConfig`` installed via ``set_active_config`` by the launcher entry
point (moshi_finetune_launcher.py), so the fork's strict TrainArgs schema is
left untouched.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import torch
import torchaudio.functional as F_audio

from datasets import load_from_disk, Dataset
from finetune.data.interleaver import (
    InterleavedTokenizer,
    Sample,
    Batch,
    dicho,
)
from finetune.data.dataset import parse_data_sources

from i6_experiments.users.dorian_koch.speech_llm.moshi_arrow_config import ArrowDataConfig

_log = logging.getLogger(__name__)

_HF_SENTINEL = "dataset_info.json"  # marker file in every arrow dataset dir

# Active data/augmentation config, installed by the launcher before training
# (see moshi_finetune_launcher.py). Defaults to a no-op so legacy runs are
# unchanged when no sidecar is present.
_ACTIVE_CONFIG = ArrowDataConfig()


def set_active_config(cfg: ArrowDataConfig) -> None:
    """Install the data/augmentation config used by build_data_loader."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = cfg


def _is_arrow_source(path: str) -> bool:
    """True if path looks like a saved HF arrow dataset directory."""
    from pathlib import Path

    p = Path(path)
    return p.is_dir() and ((p / _HF_SENTINEL).exists() or (p / "state.json").exists())


# ---------------------------------------------------------------------------
# Tokenize one row in-memory (replaces InterleavedTokenizer.__call__ file I/O)
# ---------------------------------------------------------------------------


def _norm_aligns(raw) -> list:
    """Normalize an alignments column value to positional tuples
    ``(text, (start, end), speaker)`` expected by the fork's ``dicho`` /
    ``Interleaver``. Handles both the struct schema (list of dicts) written by
    the current annotate job and the legacy ``[text, [start, end], speaker]``
    layout stored via the Json feature (where numeric-looking words decode back
    to numbers, hence the ``str(...)``).
    """
    out = []
    for a in raw:
        if isinstance(a, dict):
            out.append((str(a["text"]), (a["start"], a["end"]), a["speaker"]))
        else:
            out.append((str(a[0]), (a[1][0], a[1][1]), a[2]))
    return out


def tokenize_arrow_row(
    row: dict,
    instruct_tokenizer: InterleavedTokenizer,
    start_sec: float,
    jitter_sec: float = 0.0,
) -> Sample:
    """Tokenize one (sliced) row from the arrow dataset.

    Mirrors InterleavedTokenizer.__call__ but reads audio/alignments from the
    dict rather than from disk.

    ``jitter_sec`` prepends that many seconds of leading silence to the audio
    (and shifts the alignments by the same amount), so the dialogue does not
    always start at frame 0. 0.0 reproduces the original behaviour exactly.
    """
    # New format (Option B): two mono HF Audio columns (audio_assistant/_user).
    # Legacy format: a single stereo "audio" column. Read both for back-compat.
    if row.get("audio_assistant") is not None:
        a, u = row["audio_assistant"], row["audio_user"]
        sr_native = a["sampling_rate"]
        assistant = np.asarray(a["array"], dtype=np.float32).reshape(-1)
        user = np.asarray(u["array"], dtype=np.float32).reshape(-1)
        n = max(assistant.shape[0], user.shape[0])
        if assistant.shape[0] < n:
            assistant = np.pad(assistant, (0, n - assistant.shape[0]))
        if user.shape[0] < n:
            user = np.pad(user, (0, n - user.shape[0]))
        audio_array = np.stack([assistant, user], axis=1)  # (T, 2): ch0=assistant, ch1=user
    else:
        sr_native = row["audio"]["sampling_rate"]
        audio_array = np.array(row["audio"]["array"], dtype=np.float32)  # (T, 2) or (2, T)
        if audio_array.ndim == 2 and audio_array.shape[0] == 2:
            audio_array = audio_array.T  # → (T, 2)

    # Convert to torch stereo tensor (2, T) expected by mimi.encode
    wav = torch.from_numpy(audio_array.T).float()  # (2, T)

    # Resample if needed
    if sr_native != instruct_tokenizer.mimi.sample_rate:
        wav = F_audio.resample(wav, sr_native, instruct_tokenizer.mimi.sample_rate)

    duration_sec = instruct_tokenizer.duration_sec
    num_audio_frames = instruct_tokenizer.num_audio_frames
    mimi_sr = instruct_tokenizer.mimi.sample_rate

    # Augmentation: prepend `jitter_sec` of leading silence so the dialogue does
    # not always start at frame 0. The alignments are shifted by the same amount
    # below so audio and text stay frame-synced.
    if jitter_sec > 0.0:
        pad_samples = int(jitter_sec * mimi_sr)
        if pad_samples > 0:
            wav = torch.nn.functional.pad(wav, (pad_samples, 0))

    # Slice to [start_sec, start_sec + duration_sec]
    start_sample = int(start_sec * mimi_sr)
    end_sample = start_sample + int(duration_sec * mimi_sr)
    wav_chunk = wav[:, start_sample:end_sample]

    with torch.no_grad():
        audio_tensor = wav_chunk.cuda()
        audio_tokens = instruct_tokenizer.mimi.encode(audio_tensor[:, None])
        audio_tokens = audio_tokens[..., :num_audio_frames]
        this_num = audio_tokens.shape[-1]
        audio_tokens = torch.nn.functional.pad(
            audio_tokens[..., :num_audio_frames],
            (0, num_audio_frames - this_num),
            value=instruct_tokenizer.interleaver.zero_padding,
        )
        audio_tokens = audio_tokens.view(1, -1, num_audio_frames)

        alignments = _norm_aligns(row.get("alignments", []))
        # Shift alignments to match the prepended leading silence.
        if jitter_sec > 0.0:
            alignments = [(a[0], (a[1][0] + jitter_sec, a[1][1] + jitter_sec), a[2]) for a in alignments]
        # Slice alignments to [start_sec, start_sec + duration_sec]
        start_align = dicho(alignments, start_sec)
        end_align = dicho(alignments, start_sec + duration_sec)
        sliced_aligns = [
            (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2]) for a in alignments[start_align:end_align]
        ]

        # this_num is in audio frames (not seconds). The upstream fork's own
        # InterleavedTokenizer.__call__ passes frames identically (interleaver.py:279).
        # build_token_stream produces T = frames * frame_rate tokens but word placement
        # is also in frame units, so the oversized stream is then truncated back to
        # num_audio_frames by the pad below. Accidentally correct; don't 'fix' to seconds.
        text_tokens = instruct_tokenizer.interleaver.prepare_item(sliced_aligns, this_num)
        text_tokens = torch.nn.functional.pad(
            text_tokens,
            (0, num_audio_frames - text_tokens.shape[-1]),
            value=instruct_tokenizer.interleaver.zero_padding,
        )

        codes = torch.cat([text_tokens, audio_tokens], dim=1)
        text_conds = row.get("text_conditions", None)
        return Sample(codes, text_conds)


# ---------------------------------------------------------------------------
# Per-source iterator over an arrow dataset
# ---------------------------------------------------------------------------


def _arrow_source_iterator(
    path: str,
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
    jitter_max_sec: float = 0.0,
) -> Iterator[Sample]:
    duration_sec = instruct_tokenizer.duration_sec
    epoch = 1
    while True:
        dataset: Dataset = load_from_disk(path)
        # Shard by rank
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        if shuffle_at_epoch and seed is not None:
            dataset = dataset.shuffle(seed=seed + epoch)

        # Per-epoch RNG for jitter; rank/epoch-distinct and reproducible.
        rng = np.random.RandomState(np.array((seed if seed is not None else 0, epoch, rank)))

        for row in dataset:
            total_dur = float(row.get("duration", 0.0))
            start_sec = 0.0
            while start_sec < total_dur:
                jitter = 0.0
                # Leading-silence jitter on the first window only, clamped to the
                # window headroom so no content is pushed past duration_sec.
                if jitter_max_sec > 0.0 and start_sec == 0.0:
                    headroom = max(0.0, duration_sec - total_dur)
                    jitter = float(rng.uniform(0.0, min(jitter_max_sec, headroom)))
                yield tokenize_arrow_row(row, instruct_tokenizer, start_sec, jitter)
                start_sec += duration_sec

        if is_finite:
            break
        _log.info("Rank %d finished arrow epoch %d", rank, epoch)
        epoch += 1
        if seed is not None:
            seed += 1  # advance seed each epoch (matches legacy behaviour)


# ---------------------------------------------------------------------------
# Multi-source build_dataset equivalent (handles mixed arrow + legacy)
# ---------------------------------------------------------------------------


def build_arrow_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
    jitter_max_sec: float = 0.0,
) -> Iterator[Sample]:
    """Build an iterator that mixes arrow and legacy sources.

    pretrain_data format: same comma-separated path:weight string as the
    original finetune.data.dataset.parse_data_sources.
    """
    from finetune.data.dataset import (
        get_dataset_iterator,
        interleave_iterators,
    )

    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)
    # Train shuffles per args; eval also shuffles (seeded) so each cycled 40-batch
    # eval subset is representative of the whole val set rather than a biased
    # contiguous chunk. Determinism comes from the fixed eval seed injected in
    # build_data_loader (the fork passes seed=None for eval).
    shuffle = shuffle_pretrain if not is_eval else True

    iterators = []
    for source in sources:
        path = str(source.path)
        if _is_arrow_source(path):
            iterators.append(
                _arrow_source_iterator(
                    path,
                    instruct_tokenizer,
                    rank=rank,
                    world_size=world_size,
                    # Eval loader must CYCLE, not run finite. The fork builds the
                    # eval loader once (train.py) and reuses that one generator for
                    # every eval, pulling ~40 batches each time. A finite val source
                    # (here: 3027 rows ~= 189 batches) drains after ~5 evals; the next
                    # eval then iterates zero batches and evaluate() divides loss by
                    # total_num_samples==0 -> NaN eval_loss/perplexity (train_loss is
                    # fine, separate loader). No shuffle/jitter on eval, so cycling
                    # just replays the same val windows -> a stable 40-batch estimate.
                    is_finite=False,
                    seed=seed,
                    shuffle_at_epoch=shuffle,
                    jitter_max_sec=jitter_max_sec,
                )
            )
        else:
            # Fall back to original file-based iterator for legacy sources
            iterators.append(
                get_dataset_iterator(
                    source,
                    instruct_tokenizer=instruct_tokenizer,
                    rank=rank,
                    world_size=world_size,
                    is_finite=is_eval,
                    seed=seed,
                    shuffle_at_epoch=shuffle,
                )
            )

    if is_eval:
        import itertools

        return itertools.chain.from_iterable(iterators)

    rng = np.random.RandomState(seed=np.array((seed, rank)))
    return interleave_iterators(iterators, probabilities=probabilities, rng=rng)


# ---------------------------------------------------------------------------
# Drop-in replacement for finetune.data.data_loader.build_data_loader
# ---------------------------------------------------------------------------


_EVAL_SHUFFLE_SEED = 12345  # fixed seed for the seeded, cycling eval loader shuffle


def build_data_loader(
    instruct_tokenizer,
    args,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
):
    """Drop-in for finetune.data.data_loader.build_data_loader.

    Used by moshi_finetune_launcher.py via monkeypatch.
    Supports mixed arrow + legacy sources.
    """
    if is_eval:
        assert args.eval_data != "", "No eval data provided."
    pretrain_data = args.train_data if not is_eval else args.eval_data

    # The fork builds the eval loader with seed=None. Supply a fixed eval seed so the
    # (now cycling) eval loader shuffles deterministically and reproducibly run-to-run.
    if is_eval and seed is None:
        seed = _EVAL_SHUFFLE_SEED

    # No jitter during eval; training jitter comes from the installed config.
    jitter_max_sec = 0.0 if is_eval else _ACTIVE_CONFIG.jitter_max_sec

    dataset = build_arrow_dataset(
        pretrain_data=pretrain_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=args.shuffle,
        jitter_max_sec=jitter_max_sec,
    )

    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)
        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []
