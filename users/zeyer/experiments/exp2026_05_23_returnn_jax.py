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
_BUCKETS = [
    {"batch_dim": 200, "audio": 89_920, "text": 73},  # 18M
    {"batch_dim": 177, "audio": 120_320, "text": 70},  # 21M
    {"batch_dim": 132, "audio": 176_640, "text": 110},  # 23M
    {"batch_dim": 90, "audio": 229_440, "text": 85},  # 21M
    {"batch_dim": 69, "audio": 236_320, "text": 89},  # 16M
    {"batch_dim": 67, "audio": 239_200, "text": 91},  # 16M
    {"batch_dim": 66, "audio": 279_664, "text": 91},  # 18M
    {"batch_dim": 57, "audio": 311_939, "text": 98},  # 18M
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
