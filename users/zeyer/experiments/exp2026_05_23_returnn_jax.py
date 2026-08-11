"""
Loquacious AED+CTC on RETURNN's JAX backend: the backend comparison to ``base-small-v2``.

Companion recipe with its OWN sis manager, started with the torch-2.12 env
(``/home/az668407/work/py-envs/py3.12-torch2.12/bin/python``), because JAX is not installed in the
production env and cannot be added there without upgrading ``nvidia-cudnn-cu12`` underneath
torch 2.7.1 (measured with ``pip --dry-run``: 9.5.1.17 -> 9.24.0.43), which the other sessions
train from. The manager's interpreter IS the jobs' interpreter
(``RETURNN_PYTHON_EXE = sys.executable`` in ``settings.py``), so nothing else has to change --
same setup dir, same settings, same RETURNN checkout (which already carries ``returnn/jax/``).
Same pattern as the grad-align setup's ``*_p212`` companion.

Start it with (its own tmux window, `--inspect` first to see what would be submitted)::

    ./hpc-sis-m.py --inspect --target 0:<win> \
        --setup-dir /rwthfs/rz/cluster/home/az668407/setups/2026-05-23-returnn-paper \
        --py /home/az668407/work/py-envs/py3.12-torch2.12/bin/python \
        recipe/i6_experiments/users/zeyer/experiments/exp2026_05_23_returnn_jax.py

Only the backend differs from ``base-small-v2`` -- same model, data, LR schedule -- so the
training curves are directly comparable. The model config comes from
:func:`small_model_overrides` rather than a copy, so it cannot drift from the baseline.
"""

from __future__ import annotations

from .exp2026_05_23_returnn import loq_train, small_model_overrides


# Declared input shapes for the compiled step. The JAX step is compiled per input signature and a
# compile costs ~200 s, so the shapes are DECLARED rather than discovered: every batch is padded
# into the first bucket it fits, all buckets are compiled at startup, and nothing compiles during
# training (a batch fitting none is an error, not a compile).
#
# Derived from the PT baseline's own training log (job ReturnnTrainingJob.je6PefFx3gz2, log.run.1,
# ep 100: 3541 batches, 2722 distinct shapes -- so per-shape compilation is hopeless). Grouping
# those by padded length into 8 buckets costs 1.21x the batcher's own padding. Every bucket is
# ~one batch's worth of work (16-23M samples, vs batch_size 16M): a bucket must be, since
# max_seqs x max_seq_length would be 64M and OOMs (measured: 44.7 GiB allocation, fp32).
# Audio levels are MULTIPLES OF time_multiple (16000) and text of 8, because the engine rounds a
# batch up to those before matching a bucket -- buckets derived from raw observed lengths sit just
# below the rounded shapes and cover nothing (measured: batch (200, 96000) fit no bucket, run died).
#
# Audio every 16k samples, so a batch is padded at most 1 s beyond its own longest sequence.
# The sequence count per cell is FITTED to a replay of 114910 real batches (the first run's log),
# with a 15% margin, floored by what the batcher could produce at that length
# (n = min(max_seqs, 20M / audio)) so that cells the replay never visited are still covered.
#
# The previous grid derived n from a 1.5x budget headroom at 7 audio levels. Measured against the
# same replay, that computed 1.41x the volume PyTorch does (which pads only to the batch's own max),
# and the run was 1.40x slower -- the headroom WAS the slowdown. This grid computes 1.19x.
#
# 60 buckets is 60 compiles at ~35 s. That is ~35 min, and it would be paid again on every
# resubmission (the 11.9h limit splits the run), which is why jax_compilation_cache_dir is set
# below: a resubmitted run reloads the programs instead of rebuilding them.
_AUDIO_LEVELS = list(range(16_000, 320_001, 16_000))

# The text axis needs its OWN levels, not one bound per audio level: a run died after 1771 steps on
# audio (141, 128000) with text 192, where the audio and the sequence count fit and only the text
# did not. Transcript density has a long tail (192 SPM labels for 8 s of audio is ~23 labels/sec,
# far above the ~3.5 average), and a single generous bound instead would pad EVERY batch's decoder
# to it -- the decoder self-attention is quadratic in this axis.
#
# Unused buckets cost only their compile, never memory -- precompilation lowers and compiles,
# it does not execute. The replay never needed 768, but a different epoch order could.
_TEXT_LEVELS = [128, 384, 768]

# Per (audio, text) cell, the seq count the replayed batches actually needed, +15%, and never below
# what the batcher could produce at that audio length. See the comment on _AUDIO_LEVELS.
_FITTED_SEQ_COUNTS = {
    112_000: 191,
    128_000: 164,
    144_000: 143,
    160_000: 127,
    176_000: 114,
    192_000: 104,
    208_000: 96,
    224_000: 88,
    240_000: 82,
    256_000: 76,
    272_000: 72,
    288_000: 67,
    304_000: 64,
    320_000: 60,
}
_MAX_SEQS, _SEQ_BUDGET = 200, 20_000_000

# audio-major, text ascending: _bucket_for takes the first fit, so a batch lands in the smallest
# text bucket that holds it
_BUCKETS = [
    {
        "batch_dim": min(_MAX_SEQS, max(_SEQ_BUDGET // audio, _FITTED_SEQ_COUNTS.get(audio, 0))),
        "audio": audio,
        "text": text,
    }
    for audio in _AUDIO_LEVELS
    for text in _TEXT_LEVELS
]


def py():
    """Sisyphus entry point."""
    loq_train(
        "base-small-v2-jax",
        {},
        config_overrides={
            "model.behavior_version": 29,
            **small_model_overrides(),
            "train.backend": "jax",
            # bf16 compute, f32 parameters and optimizer (returnn.frontend.amp), the same split
            # torch_amp gives. Measured on an H100 at this batching: 1.4-1.7x faster than f32.
            "train.jax_amp": "bfloat16",
            "train.jax_jit": {
                # the padded time extent is rounded up per key, in that key's own unit
                # (samples for audio, labels for text), which bounds how many shapes exist at all
                "time_multiple": {"audio": 16_000, "text": 8},
                "buckets": _BUCKETS,
            },
            # 60 buckets is ~35 min of compiles at startup, and the 11.9h SLURM limit splits the
            # run into several submissions. Cached, a resubmitted run reloads them instead.
            "train.jax_compilation_cache_dir": "/work/az668407/jax_compilation_cache",
        },
        # torch-only, no JAX counterpart. The post-config torch_* entries are dropped by train()
        # itself; only the hashed train config needs an explicit delete, as for the TF variant.
        # The JAX engine REJECTS any option it would otherwise ignore silently, so if the first
        # run names further ones, they belong here.
        config_deletes=["train.torch_amp"],
    )
