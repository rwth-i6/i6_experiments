"""
Like :mod:`.loquacious_aed_packed` but for the FZJ Jupiter cluster, i.e. 4-GPU (one GH200 node) training.

Multi-GPU recipe, following ``asr-base-mgpu-logmel-muon-lr5e3-wdbl-nep38-packed-graphc`` in
:mod:`i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj` (the best 4-GPU
optimizer setting found there), adapted to the Loquacious packed setup:

- ``__num_processes: 4`` in the TRAIN config (train_v4 pops it from there -- at the top level of the
  overrides it would trip ``train()``'s unhandled-keys assert).
- ``torch_distributed: {"reduce_type": "grad_explicit"}``: a compiled step returns the grads, so
  DDP's autograd hooks never fire; grad_explicit all-reduces them between step and optimizer
  (RETURNN refuses plain DDP under torch_cuda_graph). This rules out ``capture_optimizer`` (the
  optimizer must run after the all-reduce, i.e. outside the graph), and with it
  ``optimizer.capturable`` -- which Muon would not accept anyway.
- Dataset sharding: ``distrib_shard_files: True`` shards the arrow FILES across the 4 ranks, so the
  ranks cover a subepoch disjointly (each caches only its 1/4 into the node-local FileCache), and
  ``sharding_fix: True`` makes sure the sub-epoch dataset does not shard its seq order AGAIN on top
  (RETURNN issue 1738: every rank would silently consume 1/16). behavior_version 29 already
  implies the fix; setting it explicitly makes an older RETURNN fail loudly instead.
  Consequence: one subepoch is still 1000h of audio, so n_ep=100 keeps the 100k hours of the
  single-GPU runs -- at 4x the effective batch (4 x 28M packed content) and 1/4 the optimizer
  updates (~500 instead of ~2000 steps per subepoch).
- Muon (orthogonalized momentum on the 2-D hidden weights, AdamW on the rest, see
  ``optim_ext/muon.py``). LR: the reference swept the Muon peak LR {1e-3 ... 4e-2} and 5e-3 won,
  with base_lr 1.0 (vs the AdamW schedule's base_lr 0.5, peak 1e-3 -> eff. 5e-4). Weight decay
  stays 1e-2 with the AdamW module blacklist kept ("wdbl", better than without). ``optimizer.epsilon``
  is an AdamW kwarg Muon does not take -> deleted.
- SpecAugment step compensation: the schedule ``specaugment_steps`` (5k/15k/25k, from
  config_24gb_v6) is in STEPS, and 4-GPU sharding cuts the steps per subepoch by 4. Uncompensated,
  full SpecAugment strength would only be reached around subepoch 50 (of 100) instead of ~12, so the
  schedule is divided by 4 to land at the same point of the training. This is the "stepcomp" of the
  reference, which did the same for the packed 0.794 step ratio. The reference additionally found a
  stronger mask count (``specaugment_num_spatial_mask_factor`` 40-60 instead of 100) better -- but
  on LibriSpeech at ~100 passes; at 4 passes over 25k hours that regularization argument is weaker,
  so the factor is left at 100 here and exposed as a knob for a follow-up.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import returnn.frontend as rf

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs as _baseline_configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon

from .loquacious_aed_packed import (
    train,
    _base_config,
    _packed_overrides,
    text_seq_len_stats,
)

_NUM_GPUS = 4


def _scaled_steps(steps: Sequence[int], factor: float) -> Tuple[int, ...]:
    """A step schedule rescaled by ``factor`` (e.g. 1/num_gpus for sharded DDP), rounded to ints."""
    return tuple(int(round(s * factor)) for s in steps)


# Packed + compiled config for the multi-GPU DDP case, derived from the single-GPU _packed_overrides.
# Two keys go: capture_optimizer is incompatible with the grad_explicit reduce (see the module
# docstring), and optimizer.capturable is the matching AdamW option -- not applicable to Muon.
_packed_mgpu_overrides: Dict[str, Any] = {
    **{k: v for k, v in _packed_overrides.items() if k != "train.optimizer.capturable"},
    "train.torch_cuda_graph": {
        k: v for k, v in _packed_overrides["train.torch_cuda_graph"].items() if k != "capture_optimizer"
    },
    "train.__num_processes": _NUM_GPUS,
    "train.torch_distributed": {"reduce_type": "grad_explicit"},
    # The whole GH200 node: ReturnnTrainingJob multiplies cpu/mem by num_processes, so 72 -> 288
    # cores (the reference's setting), which the 4 x 25 dataset worker processes need.
    "train.__cpu_rqmt": 72,
    # File-level sharding across the ranks, see the module docstring.
    "train_dataset_opts": {"distrib_shard_files": True, "sharding_fix": True},
}


def _muon_overrides(*, base_lr: float = 1.0, peak_lr: float = 5e-3) -> Dict[str, Any]:
    """
    Muon in place of AdamW, with the reference's best LR (see the module docstring).

    Use together with ``config_deletes=["train.optimizer.epsilon"]``.
    ``optimizer.weight_decay`` (1e-2) and ``optimizer.weight_decay_modules_blacklist`` are inherited
    unchanged: weight_decay is a Muon kwarg, the blacklist is RETURNN updater logic that Muon never sees.
    """
    return {
        "train.optimizer.class": rf.build_dict(Muon)["class"],
        "train_update_func_from_n_ep": lambda n_ep: {
            "train": _baseline_configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=base_lr, peak_lr=peak_lr)
        },
    }


def _specaug_overrides(*, step_factor: float, num_spatial_mask_factor: Optional[int] = None) -> Dict[str, Any]:
    """
    SpecAugment adapted to a changed steps-per-epoch ratio.

    :param step_factor: steps per subepoch relative to the single-GPU base config, e.g. 1/4 for
        4-GPU sharded DDP. ``specaugment_steps`` is scaled by it so the ramp ends at the same
        point of the training (in epochs), see the module docstring.
    :param num_spatial_mask_factor: None keeps the default (100). Lower = more time masks
        (``max_num_masks = len // factor``); the reference's ladder found 40-60 best on LibriSpeech.
    """
    d: Dict[str, Any] = {
        "train.specaugment_steps": _scaled_steps(_base_config["train"]["specaugment_steps"], step_factor),
    }
    if num_spatial_mask_factor is not None:
        d["train.specaugment_num_spatial_mask_factor"] = num_spatial_mask_factor
    return d


def py():
    """Sisyphus entry point."""

    # base-v2-packed-qwenVocab (see loquacious_aed_packed.py) on 4 GPUs with Muon.
    # Same data (100k hours, n_ep 100), same model, same packed/compiled step per rank;
    # 4x the effective batch, 1/4 the updates, Muon at peak LR 5e-3, SpecAugment step-compensated.
    train(
        "base-v2-packed-qwenVocab-mgpu4-muon-lr5e3",
        config_overrides={
            **_packed_mgpu_overrides,
            **_muon_overrides(base_lr=1.0, peak_lr=5e-3),
            **_specaug_overrides(step_factor=1 / _NUM_GPUS),
            "train_seq_ordering": "random",
            "vocab": "qwen-restricted",
            "train_vocab_opts": None,
        },
        config_deletes=["train.optimizer.epsilon"],
    )
