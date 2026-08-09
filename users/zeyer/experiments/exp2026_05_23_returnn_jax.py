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
# The sequence count per level comes from the batch budget rather than from observed batches:
# n = min(max_seqs, 1.5 * batch_size / audio). Every bucket is then ~24M samples, i.e. 1.5x the 16M
# content budget, which is the headroom for a batch whose padding is worse than usual. Coverage is
# then structural, not observational -- fitting one epoch's shapes exactly would leave the next
# epoch's data order free to produce a batch that fits nothing and stops the run.
_BUCKETS = [
    {"batch_dim": 200, "audio": 96_000, "text": 128},  # 19M
    {"batch_dim": 188, "audio": 128_000, "text": 128},  # 24M
    {"batch_dim": 150, "audio": 160_000, "text": 128},  # 24M
    {"batch_dim": 125, "audio": 192_000, "text": 128},  # 24M
    {"batch_dim": 100, "audio": 240_000, "text": 128},  # 24M
    {"batch_dim": 84, "audio": 288_000, "text": 128},  # 24M
    {"batch_dim": 75, "audio": 320_000, "text": 128},  # 24M
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
        },
        # torch-only, no JAX counterpart. The post-config torch_* entries are dropped by train()
        # itself; only the hashed train config needs an explicit delete, as for the TF variant.
        # The JAX engine REJECTS any option it would otherwise ignore silently, so if the first
        # run names further ones, they belong here.
        config_deletes=["train.torch_amp"],
    )
