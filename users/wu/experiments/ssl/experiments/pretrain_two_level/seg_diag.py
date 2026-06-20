"""Wire the reusable CIF-segmentation probe (``analysis/seg_diag``) onto the two-level pretraining arms.

One call -> for each rate arm, a forward pass over dev-clean that dumps per-utterance CIF diagnostics and an
analysis job that writes ``<arm>/seg_diag/ep<E>/{metrics.json,summary.txt,plots/}``. Defaults to the FINAL
pretrained checkpoint (the only one guaranteed to survive ``keep_last_n`` cleanup); pass ``epoch`` to probe a
different surviving checkpoint.
"""

from __future__ import annotations

from ...analysis.seg_diag import register_seg_diag
from ...data import datasets as ds
from ...data.spm import get_ls960_spm, SPM_VOCAB_SIZE
from .baseline import (
    two_level_assets,
    two_level_80ms_c128,
    two_level_120ms_c128,
    RATE_80MS,
    RATE_120MS,
)

PRETRAIN_NUM_EPOCHS = 50  # two_level_pretrain default => final checkpoint = ep50

_ARMS = [
    ("ssl/pretrain_two_level/ls960_cif80ms_k128", RATE_80MS, two_level_80ms_c128),
    ("ssl/pretrain_two_level/ls960_cif120ms_k128", RATE_120MS, two_level_120ms_c128),
]


def _arm_seg_diag(arm_prefix, target_rate_hz, train_job_fn, *, epoch, num_clusters=128):
    model_config, codebook = two_level_assets(target_rate_hz=target_rate_hz, num_clusters=num_clusters)
    train_job = train_job_fn()
    _spm_ds, spm_model = get_ls960_spm()
    return register_seg_diag(
        f"{arm_prefix}/seg_diag/ep{epoch}",
        checkpoint=train_job.out_checkpoints[epoch],
        model_config=model_config,
        codebook=codebook,
        num_clusters=num_clusters,
        target_rate_hz=target_rate_hz,
        vocab_size=SPM_VOCAB_SIZE,
        spm_model=spm_model,
        split=ds.DEV_CLEAN,
    )


def seg_diag_two_level_all(epoch: int = PRETRAIN_NUM_EPOCHS):
    """Register the CIF segmentation probe for both pretraining arms at ``epoch`` (default the final ckpt)."""
    return [_arm_seg_diag(p, r, fn, epoch=epoch) for p, r, fn in _ARMS]
