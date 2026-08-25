"""
TTS-encoder project, FZJ (Juelich) variant -- larger-scale runs.

Role in the split: the FZJ 4-GPU recipe;
also owns the shared ``_train_tts_encoder`` builder (imported by the ``_rz`` recipe).
Imports ``train_ls_base`` + ``DbMelFeatureExtractor`` from ``exp2026_05_28_tts_encoder``.

- ``base-ls``: standard log-mel CTC+AED baseline. Same Job hash as RZ / the FZJ base setup -> imports.
- ``tts-enc-v1``: the TTS-encoder text-util training. Standard ``exp2024_04_23_baselines.aed.Model`` on the DbMel
  front-end + a frozen GlowTTS attached as ``model.tts`` (reusable wrapper in ``external_models/glow_tts.py``).
  Trained on paired LS audio + text-only LS-LM data via ``alternate_batching``:
    * audio batch:     raw audio -> model.encode -> aux-CTC + AED-CE.
    * text-only batch: phonemes  -> model.tts (frozen) -> log-mel -> model.encode_from_features -> same losses.

  ``aed_glowtts_model_def`` builds the model (frozen GlowTTS as model.tts); ``aed_glowtts_train_step`` is a custom
  RETURNN train_step doing its own extern_data extraction (incl. the 3rd "phonemes" stream the default train_v4
  step does not forward) + the dual-branch losses -- so no separate TrainDef is needed (passed as
  ``config["train_step"]``; train_v4/train_exp only fall back to the default TrainDef when no custom step is set).

  Data: a ``CombinedDataset`` interleaving
    "asr"  = LS OggZip (audio + spm, speed-perturbed); phonemes auto zero-filled (empty) by CombinedDataset.
    "text" = one ``LmDataset`` emitting the raw utf8 bytes of each LS-LM line, wrapped in a ``PostprocessingDataset``
             whose ``map_seq`` derives BOTH the spm target and the GlowTTS phonemes from that text (single corpus
             read; spm + phone_info opts passed via the config). spm[i] & phon[i] are trivially the same line.
  CV (dev/devtrain) stays the baseline audio-only sets, wrapped in a ``PostprocessingDataset`` that adds an empty
  ``phonemes`` stream so they satisfy the extern_data contract (every declared key required in every batch, incl.
  CV). At CV ``have_audio=True`` so the empty phonemes is never used.

NOTE: the data wiring (the text map_seq tokenization, CombinedDataset+alternate_batching, the CV PostprocessingDataset
map_seq, and the text-branch rf graph) is graph-buildable but NOT unit-testable -- validate with a 1-step smoke
run (``time_rqmt<=1``) before a full run. See projects/2026-05-28-tts-encoder.md.

Run from an FZJ setup (py7 is an RZ-only alias; on FZJ use the wrap.sh launcher, which does the module load):
``/e/project1/spell/zeyer1/py-envs/py3.13-torch2.12/wrap.sh python ./sis m recipe/i6_experiments/users/zeyer/experiments/exp2026_05_28_tts_encoder_fzj.py``
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Optional, Union, Dict, Any, Sequence, Tuple

from sisyphus import tk

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.model_interfaces import ModelDef
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import aed as _aed
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import Model
from i6_experiments.users.zeyer.external_models.glow_tts import GlowTtsLogMel, get_glow_tts_phoneme_vocab_size
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import train_ls_base, DbMelFeatureExtractor

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_fzj"

PHONEMES_DATA_KEY = "phonemes"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # WER comments above each experiment below are %, joint AED+CTC first-pass
    # (aed+ctc-batched/recog-1stpass), as {"dev-clean", "dev-other", "test-clean", "test-other"}.
    # Standard log-mel baseline. Same Job hash as RZ base-ls / the FZJ base setup -> imports, not re-trained.
    base_exp = train_ls_base("base-ls", prefix=prefix)
    # Test the full-node batched dev-phase scale tuning on the (ready) base-ls checkpoint.
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2 as _get_ls_task_v2
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_batched import (
        aed_ctc_timesync_recog_recomb_auto_scale_batched,
    )

    aed_ctc_timesync_recog_recomb_auto_scale_batched(
        prefix=prefix + "/aed/base-ls/aed+ctc-batched",
        task=_get_ls_task_v2(vocab="spm10k", train_epoch_split=1, train_epoch_wise_filter=None),
        aed_ctc_model=base_exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
        num_shards=8,
    )
    # No-TTS audio-only baseline trained with the SAME FZJ 4-GPU recipe as the TTS-encoder runs
    # (DbMel front-end, nep=25 x4 DDP, batched recog) -> directly comparable reference: the TTS-encoder
    # WERs differ from this only by the TTS/text augmentation.
    _train_asr_base_multigpu("asr-base-mgpu", prefix=prefix, feature_extraction=rf.build_dict(DbMelFeatureExtractor))
    # Multi-GPU sync-strategy ablation on the no-TTS DbMel baseline:
    # parameter averaging (local-SGD style) every N steps instead of per-step gradient all-reduce.
    # Mirrors the 2024 torch config config_11gb_..._mgpu4_pavg100.
    # Same model/data/base_lr as asr-base-mgpu; only the sync differs.
    for _ps_name, _ps in [("psync100", 100), ("psync10", 10)]:
        _train_asr_base_multigpu(
            "asr-base-mgpu-" + _ps_name,
            prefix=prefix,
            feature_extraction=rf.build_dict(DbMelFeatureExtractor),
            torch_distributed={"reduce_type": "param", "param_sync_step": _ps},
        )
    # Front-end isolation: same 4-GPU recipe but STANDARD log-mel (no DbMel), with a peak-LR sweep.
    #   logmel-mgpu vs asr-base-mgpu  -> log-mel-vs-DbMel front-end cost (mgpu held fixed)
    #   logmel-mgpu vs base-ls        -> mgpu-vs-single-GPU cost (front-end held fixed).
    # Peak-LR sweep: 0.5=no-scale, 1.0~sqrt, 2.0~linear (the usual large-batch UP-scaling).
    # But lr05 (no-scale) won, so 0.1/0.05 probe BELOW base-ls's LR:
    # the 64M effective batch + 1/4 the optimizer updates may want a gentler peak, not a larger one.
    for _lr_name, _lr in [("lr05", 0.5), ("lr10", 1.0), ("lr20", 2.0), ("lr01", 0.1), ("lr005", 0.05)]:
        _train_asr_base_multigpu(
            "asr-base-mgpu-logmel-" + _lr_name, prefix=prefix, feature_extraction=None, base_lr=_lr
        )
    # mgpu-gap experiments (close base-ls 4.06 vs mgpu-logmel 4.35), all on the no-TTS log-mel baseline:
    # (1) effective-batch match -- per-rank 4M (1/4) -> 16M effective = base-ls,
    #     restoring the update count (100*S vs the 64M run's 25*S). A diagnostic for the batch/updates cause.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-bs4m", prefix=prefix, feature_extraction=None, base_lr=0.5, batch_size_feat=25_000
    )
    # (2) SOAP -- STOPPED. Muon is the cheap spectral approximation of the same (Shampoo) preconditioning
    # at ~1/3 the per-step cost, so the eigh-based SOAP (~5x AdamW/step) is dominated here; not worth the budget.
    # from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.soap import SOAP
    # _train_asr_base_multigpu(
    #     "asr-base-mgpu-logmel-soap", prefix=prefix, feature_extraction=None, base_lr=1.0, peak_lr=3e-3,
    #     extra_config_updates={"optimizer.class": rf.build_dict(SOAP)["class"]},
    #     extra_config_deletes=["optimizer.epsilon", "optimizer.weight_decay_modules_blacklist"],
    # )
    # (3) Muon -- orthogonalized-momentum, as in the nanogpt speedruns / nanochat.
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon

    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=2e-2,  # Muon LR on the orthogonalized 2-D update; Adam-grouped params scaled down internally
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon", "optimizer.weight_decay_modules_blacklist"],
    )
    # Muon LR sweep (the most critical Muon knob) bracketing the 2e-2 run above.
    for _mlr_name, _mlr in [("lr1e2", 1e-2), ("lr4e2", 4e-2), ("lr5e3", 5e-3), ("lr2_5e3", 2.5e-3), ("lr1e3", 1e-3)]:
        _train_asr_base_multigpu(
            "asr-base-mgpu-logmel-muon-" + _mlr_name,
            prefix=prefix,
            feature_extraction=None,
            base_lr=1.0,
            peak_lr=_mlr,
            extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
            extra_config_deletes=["optimizer.epsilon", "optimizer.weight_decay_modules_blacklist"],
        )
    # Muon weight-decay check at the best LR (peak 5e-3):
    # the LR sweep held wd=0.01 (the AdamW default),
    # but Muon's spectrally-normalized update may pair with a different wd.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wd1e1",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"], "optimizer.weight_decay": 0.1},
        extra_config_deletes=["optimizer.epsilon", "optimizer.weight_decay_modules_blacklist"],
    )
    # Same best-LR Muon, but restoring the AdamW weight-decay blacklist
    # (rf.Embedding + rf.LearnedRelativePositionalEncoding excluded from wd).
    # weight_decay_modules_blacklist is RETURNN updater logic
    # (popped before the optimizer in returnn/torch/updater.py), so Muon never sees it --
    # keeping it just excludes those lookup-table params from weight decay.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # Train longer on the wdbl setting: nep 25 -> 38 (+50%), LR schedule stretched accordingly.
    # Tests whether the remaining 4-GPU gap (4.22 vs base-ls 4.06) is simply too few optimizer steps
    # (4x batch at fixed nep = 1/4 the updates); the final WER vs the nep=25 run (4.22) brackets
    # how much longer is needed.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-nep38",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # for the batch-size unit: the padded arm's batch_size is raw samples, not feature frames
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs

    # Packed tensors + whole-step CUDA graphs on that same baseline, layout the only difference.
    # Reproduces the single-GPU packed speedup on the standard FZJ 4-GPU setup.
    # packed_batch_size is the padded arm's batch_size in the same raw-sample unit,
    # so a step gets the same buffer, and packing turns the padding into content.
    # dim_capacity is max_seq_length_default_input, batch_size_bound is max_seqs;
    # both come from the baseline config, not from a tuned guess.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-nep38-packed-graphc",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        behavior_version=29,  # packed tensors need >= 29
        extra_config_updates={
            "optimizer.class": rf.build_dict(Muon)["class"],
            "packed_tensors": True,
            # a compiled step returns the grads, so DDP's autograd hooks never fire.
            # grad_explicit all-reduces them between step and optimizer, matching DDP;
            # 'param' would average params after per-GPU steps, a different optimization.
            # It also rules out capture_optimizer, which needs the optimizer inside the graph.
            "torch_distributed": {"reduce_type": "grad_explicit"},
            "batch_size": None,
            "packed_batch_size": {"data": 100_000 * configs._batch_size_factor, "classes": 5_000},
            # laplace sorting exists to reduce padding, which packing already removes
            "batching": "random",
            "torch_cuda_graph": {
                "batch_size_bound": 500,
                "dim_capacity": {"data": 312_000, "classes": 80},
                "warmup_steps": 0,
                "compile": True,
            },
        },
        extra_config_deletes=["optimizer.epsilon"],
    )
    # nep38 base_lr sweep with DEFAULT peak_lr (1e-3) -> default OCLR shape: eff. peak = base_lr*1e-3, floor =
    # base_lr*1e-5 (floor/peak ratio 1/100, vs the deeper 1/500 of the peak_lr=5e-3 baseline). base_lr in
    # {1,5,10} -> eff. peak {1e-3, 5e-3, 1e-2}. baselr5 (eff. peak 5e-3, shallow floor) vs lr5e3-wdbl-nep38
    # (same 5e-3 peak, deep floor 1e-5) isolates whether the deeper floor actually helped.
    for _base_lr in (1.0, 5.0, 10.0):
        _train_asr_base_multigpu(
            f"asr-base-mgpu-logmel-muon-baselr{_base_lr:g}-nep38",
            prefix=prefix,
            feature_extraction=None,
            base_lr=_base_lr,
            nep=38,
            extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
            extra_config_deletes=["optimizer.epsilon"],
        )
    # LR-floor check on the wdbl setting: base_lr 0.5 + peak_lr 1e-2 keeps the same effective peak
    # (base_lr * peak_lr = 5e-3) but halves the decay targets (low 1e-5 -> 5e-6, final 1e-6 -> 5e-7).
    # The Muon LR sweep held base_lr=1.0 throughout, so the floor is its one untested axis;
    # wsd-longdecay > wsd suggests deeper annealing helps.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-baselr05",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        peak_lr=1e-2,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # SyncBatchNorm ablation: does global (cross-rank) BatchNorm close the mgpu gap?
    # Same as lr05 (AdamW, base_lr 0.5), plus the new rf_batch_norm_distributed flag,
    # so all rf.BatchNorm (the 16 Conformer conv BNs + feature BN) use global stats.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-syncbn",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        # _meta_hash_trigger forces a fresh from-scratch run: the fp32 SyncBN-stats fix is a RETURNN
        # code change (not a config change), so the job hash would otherwise be unchanged. We want the
        # complete run under the fixed (fp32, numerically stable) distributed BatchNorm statistics.
        extra_config_updates={"rf_batch_norm_distributed": True, "_meta_hash_trigger": "syncbn-fp32-stats"},
    )
    # GroupNorm ablation: batch-independent normalization, removing BatchNorm from the mgpu-gap picture.
    # Three variants vs lr05 (AdamW base_lr 0.5): Conformer conv-block norm, feature front-end norm, and both.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-gn-conv",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        conv_norm=rf.build_dict(rf.GroupNorm, num_groups=32),
    )
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-gn-feat",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        feature_norm=rf.build_dict(rf.GroupNorm, num_groups=16),
    )
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-gn-both",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        conv_norm=rf.build_dict(rf.GroupNorm, num_groups=32),
        feature_norm=rf.build_dict(rf.GroupNorm, num_groups=16),
    )
    # gn-conv (Conformer conv-block GroupNorm) combined with the muon-lr5e3-wdbl variant: gn-conv gave
    # ~-0.2 dev-other on the plain AdamW baseline (4.49 vs 4.69), and muon-lr5e3-wdbl is at 4.22 dev-other,
    # so the combination tests whether it reaches the 1-GPU 4.06.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-gn-conv",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
        conv_norm=rf.build_dict(rf.GroupNorm, num_groups=32),
    )
    # Same, but with the standard time-pooled GroupNorm (GroupNormSpatial: reduces over the in-group
    # channels AND time, = torch.nn.GroupNorm) instead of the per-frame rf.GroupNorm above.
    # Direct standard-vs-per-frame comparison on the conv-block norm.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-gn-conv-tp",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
        conv_norm=rf.build_dict(rf.GroupNormSpatial, num_groups=32),
    )
    # gn-feat retry on muon-lr5e3-wdbl with the standard time-pooled GroupNorm (GroupNormSpatial), testing
    # whether time-pooling rescues the feature front-end norm (per-frame gn-feat was 5.91). Two group counts:
    # 16 (as the original) and 1 (a single group = per-sequence global feature norm over channels+time).
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-gn-feat-tp",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
        feature_norm=rf.build_dict(rf.GroupNormSpatial, num_groups=16),
    )
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-gn-feat-tp-g1",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
        feature_norm=rf.build_dict(rf.GroupNormSpatial, num_groups=1),
    )
    # Combine the two batch-independent norms that each track the muon-lr5e3-wdbl baseline:
    # per-frame GroupNorm conv norm (as in gn-conv) + per-seq global feature norm (as in gn-feat-tp-g1).
    # Retry of the failed gn-both with the feature side fixed (time-pooled).
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-gn-conv-feat-g1",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
        conv_norm=rf.build_dict(rf.GroupNorm, num_groups=32),
        feature_norm=rf.build_dict(rf.GroupNormSpatial, num_groups=1),
    )
    # torchaudio's Conformer conv-norm choice: standard time-pooled GroupNorm with a single group
    # (= per-seq global norm over channels+time in the conv block).
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-muon-lr5e3-wdbl-gn-conv-tp-g1",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
        conv_norm=rf.build_dict(rf.GroupNormSpatial, num_groups=1),
    )
    # WSD LR schedule (warmup ~2% / stable ~78% / decay ~20%) vs our OCLR (45% warmup / 45% decay),
    # same 5e-4 peak as lr05. Tests whether the long OCLR warmup wastes our scarce nep=25 updates.
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-wsd",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        extra_config_updates={
            "learning_rate_piecewise_steps": [0.5, 20, 25],  # epochs: warmup->peak, hold, decay
            "learning_rate_piecewise_values": [1e-05, 1e-03, 1e-03, 1e-06],
        },
    )
    # WSD with a longer decay (40% vs 20%): shorter stable phase, longer anneal.
    # Tests whether WSD's short decay was the problem (the anneal phase is where much of the gain is).
    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-wsd-longdecay",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        extra_config_updates={
            "learning_rate_piecewise_steps": [0.5, 15, 25],  # decay ep15->25 (40%)
            "learning_rate_piecewise_values": [1e-05, 1e-03, 1e-03, 1e-06],
        },
    )
    # LAMB -- the canonical large-batch optimizer (You et al., trust-ratio); the on-the-nose comparison.
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.lamb import LAMB

    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-lamb",
        prefix=prefix,
        feature_extraction=None,
        base_lr=1.0,
        peak_lr=2e-3,  # LAMB's trust-ratio normalizes the step; starts higher than AdamW's 5e-4
        extra_config_updates={"optimizer.class": rf.build_dict(LAMB)["class"]},
        extra_config_deletes=["optimizer.epsilon", "optimizer.weight_decay_modules_blacklist"],
    )
    # AdEMAMix -- AdamW + a slow gradient EMA (more progress per update); same 5e-4 peak as the AdamW best.
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.ademamix import AdEMAMixV2

    _train_asr_base_multigpu(
        "asr-base-mgpu-logmel-ademamix2",
        prefix=prefix,
        feature_extraction=None,
        base_lr=0.5,
        peak_lr=1e-3,
        extra_config_updates={
            "optimizer.class": rf.build_dict(AdEMAMixV2)["class"],
            "optimizer.betas": (0.9, 0.999, 0.9999),
            "optimizer.alpha": 5.0,
            # paper's alpha/beta3 warmup (official apple/ml-ademamix):
            # ramp alpha 0->5 and beta3 (in EMA-half-life space) over ~the full run (~100k steps),
            # so the slow EMA phases in gradually.
            # Required for stability -- fixed alpha=5 from step 0 diverged at ep8.
            "optimizer.alpha_warmup": 100_000,
            "optimizer.beta3_warmup": 100_000,
        },
        extra_config_deletes=["optimizer.epsilon", "optimizer.weight_decay_modules_blacklist"],
    )
    # tts-enc-v1: pseudo-speech-enc-style text usage (~5 effective text passes, 100 ASR).
    # {"dev-clean": 1.63, "dev-other": 4.13, "test-clean": 1.90, "test-other": 4.46}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder("tts-enc-v1", prefix=prefix)
    # tts-enc-v2: TTS-baseline-style text usage (~1.33 effective text passes).
    # Slower text fill rate (partition_epoch=75) lets the audio bucket grow large under max_seqs=500,
    # so cap audio batch + phoneme seq length explicitly.
    # {"dev-clean": 1.81, "dev-other": 4.50, "test-clean": 2.01, "test-other": 4.86}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-v2-textP75",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,  # ~baseline 75 SPM equivalent (1 SPM token ~3-4 phonemes on LS, measured from v1 logs)
    # )
    # v3 (phon cap 25k) and v3a (phon cap 50k) superseded by v3b (phon cap 75k): same scientific config
    # (lenscale ~0.1, 5 effective text passes), v3b is ~28% faster wall-time than v3 at same memory peak.
    # _train_tts_encoder("tts-enc-v3-lenscale-low", prefix=prefix, glow_tts_length_scale_range=(0.05, 0.15))
    # _train_tts_encoder(
    #     "tts-enc-v3a-lenscale-low-bigphon", prefix=prefix,
    #     glow_tts_length_scale_range=(0.05, 0.15), batch_size_phon=50_000,
    # )
    # tts-enc-v3b: highly compressed synth (length_scale ~0.1), phon cap 75k (max_seqs=500 sometimes binds).
    # {"dev-clean": 2.08, "dev-other": 5.03, "test-clean": 2.28, "test-other": 5.36}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-v3b-lenscale-low-biggerphon",
    #     prefix=prefix,
    #     glow_tts_length_scale_range=(0.05, 0.15),
    #     batch_size_phon=75_000,
    # )
    # GL-net / Griffin-Lim waveform variant of v2 (same data: textP75, ~1.33 text passes): the synthetic
    # text branch goes GlowTTS->GL-net->Griffin-Lim->waveform->ASR DbMel front-end (like the offline TTS
    # baseline), instead of feeding the GlowTTS log-mel directly. Tests whether the Griffin-Lim round-trip's
    # realistic distortion helps. GL-net ckpt (H9EByABag8UN/epoch.050.pt) synced from RZ to FZJ.
    # {"dev-clean": 1.85, "dev-other": 4.60, "test-clean": 2.08, "test-other": 4.85}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-v2-textP75-gl-wave",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    # )
    # Reference-match: replicate Nick's offline ls_lm_data synth EXACTLY --
    # noise_scale 0.7 fixed, length_scale 1.0 fixed, waveform path (GL-net + Griffin-Lim), on the textP75 data.
    # Closest possible replication of the 3.53 reference within our setup
    # (residual diff: our DbMel-80Hz re-extraction vs the reference's standard log-mel 100Hz,
    # ~+0.13 per base-ls-dbmel).
    # ~3.5 -> reproduced; ~4.1 -> the gap is our 4-GPU/nep regime, not the TTS.
    # See projects/2026-05-28-tts-encoder.md.
    # {"dev-clean": 1.79, "dev-other": 4.39, "test-clean": 1.97, "test-other": 4.69}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    # )
    # ref-match but with the reference's standard log-mel 100Hz ASR front-end (waveform path required):
    # closes the ~+0.13 DbMel-80Hz-vs-log-mel-100Hz re-extraction gap -> the actual reference ceiling.
    # {"dev-clean": 1.89, "dev-other": 4.48, "test-clean": 2.04, "test-other": 4.70}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    # )
    # Same, but length sampled (0.5,1.0) -- noise stays fixed at the reference 0.7:
    # tests whether shorter / variable synthetic durations keep the WER (the cheap-able axis).
    # {"dev-clean": 1.86, "dev-other": 4.54, "test-clean": 2.14, "test-other": 4.84}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-lensamp",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(0.5, 1.0),
    # )
    # Same as -lensamp, but with the SpecAugment time-mask width scaled per-seq by the sampled length_scale,
    # so compressed synthetic sequences are not over-masked (SpecAugment's absolute 20-frame width erases
    # short low-length_scale seqs). Tests whether length-aware augmentation recovers WER.
    # {"dev-clean": 1.96, "dev-other": 4.48, "test-clean": 2.09, "test-other": 4.80}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-lensamp-specaug",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     specaugment_length_scaled=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(0.5, 1.0),
    # )
    # Muon (lr5e3 + weight-decay blacklist, the best 4-GPU optimizer setting,
    # see asr-base-mgpu-logmel-muon-lr5e3-wdbl) on the ref-match-logmel TTS config:
    # tests whether the TTS text-utilization gain stacks with the optimizer gain.
    # {"dev-clean": 1.70, "dev-other": 4.18, "test-clean": 2.00, "test-other": 4.36}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-muon",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # better: -> tts-enc-logmel-refcfg-single-gumbel-muon-nep38
    # Same as ref-match-logmel-muon (= ref-match-logmel + muon-lr5e3-wdbl), but nep 25 -> 38 (+50% updates):
    # the training-matched point removing the 4-GPU step-count penalty (audio-only nep38 bought 4.22 -> 4.01).
    # Isolates whether the ~0 TTS text-util gain is the regime, not the TTS.
    # {"dev-clean": 1.74, "dev-other": 3.98, "test-clean": 1.92, "test-other": 4.22}  (1st TTS variant < 4.0)
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=100_000,  # 120k/5k OOM'd at the long-phon bucket (92.4GB GL spike)
    #     # post preload-fix: real durations are ~167ms audio/phoneme,
    #     # the old 25k phon default OOMs at step 0 even on 96GB (~66M samples/text batch).
    #     batch_size_phon=4_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # better: -> tts-enc-logmel-refcfg-single-gumbel-muon-nep38
    # Clean front-end x path comparison vs ref-match-logmel-muon-nep38: DbMel front-end + DIRECT injection
    # (no waveform/Griffin-Lim), everything else identical. The pair isolates DbMel+direct vs log-mel+waveform
    # at the best regime (log-mel is +0.13 better front-end but needs the ~0.10 GL waveform round-trip).
    # {"dev-clean": 1.62, "dev-other": 3.98, "test-clean": 1.91, "test-other": 4.37}  (== logmel 3.98)
    # _train_tts_encoder(
    #     "tts-enc-ref-match-dbmel-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     # post preload-fix: real durations are ~167ms audio/phoneme,
    #     # the old 25k phon default OOMs at step 0 even on 96GB (~66M samples/text batch).
    #     batch_size_phon=5_000,
    #     max_phon_len=300,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # better: -> tts-enc-logmel-refcfg-single-gumbel-muon-nep38
    # noise0-muon-nep38 (ref-match-logmel + noise 0 + muon + nep38) REMOVED 2026-06-17: NaN'd at ep12 (inf/nan
    # via debug_inf_nan). Cause = low/zero-variance synthesis x muon -- noise0 was stable under AdamW/nep25
    # (4.26) and muon+nep38 was stable at noise 0.7; only the combination NaN'd. See the noise01 variant below
    # (noise 0.1, a close-to-0 hedge).
    # Same best regime, noise 0.1 (close to 0, NOT exactly 0): tests whether tiny nonzero flow variance dodges
    # the muon+nep38 NaN that noise0 (exactly 0) hit, while keeping the low-noise benefit (noise0 was the best
    # synth at nep25, 4.26). If it NaNs too, the trigger is low-noise x muon broadly, not the exact zero.
    # {"dev-clean": 1.62, "dev-other": 3.98, "test-clean": 1.80, "test-other": 4.10}  (stable; best TTS variant)
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-noise01-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=100_000,  # 120k/5k OOM'd at the long-phon bucket (92.4GB GL spike)
    #     # post preload-fix: real durations are ~167ms audio/phoneme,
    #     # the old 25k phon default OOMs at step 0 even on 96GB (~66M samples/text batch).
    #     batch_size_phon=4_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.1, 0.1),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # ref-match-logmel-muon-nep38 + more text (P20 ~5 passes vs P75's 1.33): grafts v1's text-amount advantage
    # onto the best log-mel+waveform+muon+nep38 regime (more text helped: v1 P20 4.13 vs ref-match P75 4.48).
    _train_tts_encoder(
        "tts-enc-ref-match-logmel-textP20-muon-nep38",
        prefix=prefix,
        # x4 partition on BOTH streams + x4 subepochs (nep 38 -> 152) keeps the experiment identical
        # -- text stays 1.9 passes (152/80), audio stays 38 (152/4) -- but quarters each subepoch to ~5.5h
        # so it checkpoints within the 12h chunk. Original (20/1/38) never finished a subepoch in 12h
        # (54.8% at the time limit) -> no checkpoint -> retry_error. phon batch is pinned at 4000 (peaks
        # 78.7GB already; cannot grow without OOM), so repartitioning is the only lever.
        text_train_epoch_split=80,
        ls_train_epoch_split=4,
        batch_size_audio_frames=100_000,  # 120k/5k OOM'd at the long-phon bucket (92.4GB GL spike)
        # post preload-fix: real durations are ~167ms audio/phoneme,
        # the old 25k phon default OOMs at step 0 even on 96GB (~66M samples/text batch).
        batch_size_phon=4_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        base_lr=1.0,
        peak_lr=5e-3,
        nep=152,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # better: -> tts-enc-logmel-refcfg-rnddur-short-muon-nep38
    # ref-match-logmel-muon-nep38 + random durations (rnddur: i.i.d.-uniform durations renormalized to the
    # predictor total -- structure removed, length kept), to see if random durations hold up in the best regime.
    # {"dev-clean": 1.72, "dev-other": 3.95, "test-clean": 1.92, "test-other": 4.35}  (ties layer8 dev-other)
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-rnddur-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=100_000,  # 120k/5k OOM'd at the long-phon bucket (92.4GB GL spike)
    #     # post preload-fix: real durations are ~167ms audio/phoneme,
    #     # the old 25k phon default OOMs at step 0 even on 96GB (~66M samples/text batch).
    #     batch_size_phon=4_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     glow_tts_random_durations_jitter=(0.2, 1.8),
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # Stronger duration compression: length 0.3-0.7 (mean 0.5 -> ~2x cheaper synthetic speech than
    # ref-match length 1.0), with the length-scaled SpecAugment.
    # Cost-vs-WER axis point below lensamp (0.5-1.0).
    # {"dev-clean": 1.99, "dev-other": 4.61, "test-clean": 2.10, "test-other": 4.94}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-lensamp-low-specaug",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     specaugment_length_scaled=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(0.3, 0.7),
    # )
    # better: -> pseudo-enc-logmel-muon-nep38
    # Speech-likeness 2x2, cell A (acoustics=embedding, durations=random): pseudo-speech-encoder --
    # TRAINABLE phoneme embedding, blank-interleaved, random durations (labels 1 frame, blanks 0-3;
    # the earlier study's winning setting), no TTS at all. SpecAugment time-mask width scaled to the
    # ~3x-compressed pseudo sequences (20 -> 6).
    # Same text/audio regime as ref-match-logmel (cell D: acoustics=TTS, durations=learned).
    # {"dev-clean": 1.73, "dev-other": 4.30, "test-clean": 1.98, "test-other": 4.58}
    # _train_tts_encoder(
    #     "pseudo-enc-logmel",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_specaug_max_width=6,
    # )
    # pseudo-enc-logmel (best pseudo-enc at 4.30) in the best 4-GPU regime: muon-lr5e3-wdbl + nep38.
    # {"dev-clean": 1.64, "dev-other": 4.05, "test-clean": 1.81, "test-other": 4.27}
    _train_tts_encoder(
        "pseudo-enc-logmel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_specaug_max_width=6,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # better: -> pseudo-enc-logmel-muon-nep38
    # Consistency-ladder (b): NO blank insertion -- blanks get duration 0, so the feature sequence is
    # pure label embeddings (labels still 1 frame each). Isolates the contribution of the random blank
    # frames; the phoneme seq itself already carries [space] silence tokens between words.
    # {"dev-clean": 2.11, "dev-other": 4.97, "test-clean": 2.32, "test-other": 5.32}
    # _train_tts_encoder(
    #     "pseudo-enc-logmel-noblank",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_blank_duration_range=(0, 0),
    #     pseudo_enc_specaug_max_width=6,
    # )
    # Consistency-ladder (d): like (b) no-blank, but the embedding is the FROZEN per-phoneme mean
    # log-mel table from real audio + MFA alignments ("table-lookup TTS"): acoustics from data, zero
    # trained text-encoder params. [space] keeps its real-silence row; [start]/[end] = global mean.
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import get_mfa_phone_mean_logmel_table

    # better: -> pseudo-enc-logmel-mfatable-muon-nep38
    # {"dev-clean": 2.19, "dev-other": 4.90, "test-clean": 2.24, "test-other": 5.21}
    # _train_tts_encoder(
    #     "pseudo-enc-logmel-noblank-mfatable",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_blank_duration_range=(0, 0),
    #     pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
    #     pseudo_enc_specaug_max_width=6,
    # )
    # better: -> pseudo-enc-logmel-mfatable-muon-nep38
    # Rung (d) WITH blank: the frozen MFA mean-logmel table, but blanks restored (0-3 frames) like the
    # phoneme winner -- the apples-to-apples mfatable comparison (blank gave -0.67 abs dev-other for the
    # trained embedding). Blank row = [space] silence acoustics (frozen-table loader, by design).
    # {"dev-clean": 1.70, "dev-other": 4.28, "test-clean": 1.85, "test-other": 4.70}
    # _train_tts_encoder(
    #     "pseudo-enc-logmel-mfatable",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
    #     pseudo_enc_specaug_max_width=6,
    # )
    # pseudo-enc-logmel-mfatable (frozen MFA table + blank) in the best 4-GPU regime: muon-lr5e3-wdbl + nep38.
    # {"dev-clean": 1.59, "dev-other": 3.95, "test-clean": 1.78, "test-other": 4.44}
    _train_tts_encoder(
        "pseudo-enc-logmel-mfatable-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_specaug_max_width=6,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 1.69, "dev-other": 4.21, "test-clean": 1.88, "test-other": 4.39}
    # mfatable with realistic durations: NO blanks (the phoneme seq already carries [space] silence) + label
    # duration 4-8 frames at the 100Hz front-end (avg ~60ms/phon, like real speech; ~1 enc-frame/phon after
    # 6x down, vs ~25ms / 0.4-frame at the old label-1 + blank-0-3). Tests whether the frozen mean-log-mel
    # table needs realistic durations to be usable. muon-lr5e3-wdbl + nep38.
    _train_tts_encoder(
        "pseudo-enc-logmel-mfatable-realdur-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        # label dur 4-8 makes the 100Hz front-end seqs ~2.4x longer than label-1 -> shrink the phon batch
        # (default 25k OOMed at step 1; 8k matches the old front-end-frame volume).
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_range=(4, 8),
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )

    # {"dev-clean": 2.06, "dev-other": 4.92, "test-clean": 2.17, "test-other": 5.18}
    # better: -> tts-enc-logmel-refcfg-single-gumbel-muon-nep38
    # ============ Fixed-era wave (2026-07-16, post preload-fix), muon-nep38 regime ============
    # Counterparts of the running RZ fixed experiments (see exp2026_05_28_tts_encoder_rz.py),
    # naming per the new scheme: "refcfg" = the 4 reference knobs (SamplingBPE/auxShared/auxNoBias/padRnd100).
    # batch_size_phon is EXPLICIT everywhere: with the preload fixed, real durations are ~167ms
    # audio/phoneme, so the old 25k default text batches would OOM (the RZ lesson).
    # The full-match dual-stream base (waveform + log-mel + the 4 knobs).
    # _train_tts_encoder(
    #     "tts-enc-logmel-refcfg-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=100_000,  # 120k/5k OOM'd at the long-phon bucket (92.4GB GL spike)
    #     batch_size_phon=4_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     enc_aux_logits_share_weights=True,
    #     enc_aux_logits_with_bias=False,
    #     pad_audio_rnd=100,
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # {"dev-clean": 1.53, "dev-other": 3.55, "test-clean": 1.64, "test-other": 3.69}
    # refcfg + single-stream: audio + text mixed in ONE batch, one loss set.
    # 90k/4k, not 120k/5k: merged batches stack audio AND text in one step.
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-single-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=90_000,
        batch_size_phon=4_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        single_stream=True,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 1.59, "dev-other": 3.56, "test-clean": 1.69, "test-other": 3.76}
    # + gumbel interleave: random union ordering (sampling-without-replacement), the offline ordering.
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-single-gumbel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=90_000,
        batch_size_phon=4_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # gumbel-single (the mixing winner) + short random durations (w = w_pred * U(0.2, 0.5)):
    # efficiency axis (shrunk durations) stacked on the FIXED mixing winner, waveform path kept.
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-single-gumbel-rnddur-short-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=90_000,
        batch_size_phon=4_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        glow_tts_random_durations_jitter_mult=(0.2, 0.5),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 2.56, "dev-other": 4.96, "test-clean": 2.99, "test-other": 4.83}
    # DbMel DIRECT injection (no GL/Griffin-Lim/waveform) on the refcfg base.
    _train_tts_encoder(
        "tts-enc-dbmel-direct-refcfg-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        batch_size_phon=5_000,
        max_phon_len=300,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 1.97, "dev-other": 4.56, "test-clean": 2.27, "test-other": 4.85}
    # better: -> tts-enc-logmel-refcfg-rnddur-short-muon-nep38
    # Degenerate durations: fixed 1 mel frame per phoneme (waveform path).
    # _train_tts_encoder(
    #     "tts-enc-logmel-refcfg-dur1-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     batch_size_phon=15_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     glow_tts_fixed_duration=1,
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     enc_aux_logits_share_weights=True,
    #     enc_aux_logits_with_bias=False,
    #     pad_audio_rnd=100,
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # {"dev-clean": 1.79, "dev-other": 3.98, "test-clean": 1.96, "test-other": 4.23}
    # Short random durations: w = w_pred * U(0.2, 0.5) per phoneme (waveform path).
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-rnddur-short-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        batch_size_phon=15_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        glow_tts_random_durations_jitter_mult=(0.2, 0.5),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # Full efficiency combo on the FIXED mixing winner: gumbel-single + DIRECT dbmel injection
    # (no GL/waveform) + short random durations. The one config that stacks the quality winner with
    # both cheap moves -- tests whether the mixing win survives the efficiency shortcuts, or they interact.
    _train_tts_encoder(
        "tts-enc-dbmel-direct-single-gumbel-rnddur-short-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=90_000,
        batch_size_phon=4_000,
        max_phon_len=300,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        glow_tts_random_durations_jitter_mult=(0.2, 0.5),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 2.03, "dev-other": 4.76, "test-clean": 2.09, "test-other": 5.25}
    # DbMel-direct + dur1: the cheapest online-TTS variant (RZ shows a duration x path interaction).
    _train_tts_encoder(
        "tts-enc-dbmel-direct-refcfg-dur1-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        batch_size_phon=15_000,
        max_phon_len=300,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        glow_tts_fixed_duration=1,
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 1.66, "dev-other": 3.70, "test-clean": 1.89, "test-other": 4.08}
    # The pseudo-enc winner (RZ 1-GPU nep100: 3.74 joint) in the muon-nep38 regime:
    # layer-4 injection, NO blanks (the June fleet only had layer4 WITH blanks / layer8 noblank).
    _train_tts_encoder(
        "pseudo-enc-layer4-noblank-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        batch_size_phon=8_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 14.56, "dev-other": 6.86, "test-clean": 13.66, "test-other": 7.67}
    # better: -> pseudo-enc-layer4-noblank-muon-nep38
    # Single-stream version of the pseudo-enc winner (merged at the layer-4 boundary, one loss set).
    # 90k/7k: merged batches stack audio + text (RZ: 100k/8k OOM'd at 80GB; here 96GB).
    # _train_tts_encoder(
    #     "pseudo-enc-layer4-noblank-single-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     # 90k/7k OOM'd mid-ep1 (16.5GB spike at 77GB); 70k/6k = the proven RZ caps.
    #     batch_size_audio_frames=70_000,
    #     batch_size_phon=6_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_start_layer=4,
    #     pseudo_enc_blank_duration_range=(0, 0),
    #     pseudo_enc_specaug_max_width=6,
    #     single_stream=True,
    #     pseudo_enc_single_stream_version=3,  # v1/v2 were broken (see notes)
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # gumbel-single on the pseudo-enc winner: single-stream v3 merge (layer-4, noblank)
    # + gumbel interleave ordering -- the mandated mixing on the cheapest injection mechanism.
    _train_tts_encoder(
        "pseudo-enc-layer4-noblank-single-gumbel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        pseudo_enc_single_stream_version=3,
        interleave_gumbel_scale=1.0,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 1.83, "dev-other": 4.39, "test-clean": 1.97, "test-other": 4.56}
    # Frozen MFA mean-logmel table, realistic durations, box smoothing, under gumbel-single.
    # Box width 9 == linear interpolation: upsampling is already a box of width d, a second gives a triangle.
    # Odd width stays centered, and 9 >= the max duration 8, which is what removes the plateaus.
    _train_tts_encoder(
        "pseudo-enc-logmel-mfatable-realdur-smoothbox9-single-gumbel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_range=(4, 8),
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_smooth_box_width=9,
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import get_mfa_phone_duration_table

    # Frozen MFA table, per-phone durations, linear interpolation, under gumbel-single.
    # Real durations are right-skewed and span 3.4x across phones, so the uniform (4,8) fit only 43% of them.
    # Sampled as median[phone] * exp(0.45 * N(0,1)); lerp because no box width fits 5..17-frame segments.
    _train_tts_encoder(
        "pseudo-enc-logmel-mfatable-realdur2-lerp-single-gumbel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
        pseudo_enc_duration_sigma=0.45,
        pseudo_enc_lerp=True,
        # Real LibriSpeech has 4.76 pauses/utterance, the 0.95 default gives ~30.
        # The [space] duration comes from the table's gap-derived row, so count and length both match.
        glow_tts_add_silence_between_words=0.15,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # Same, but durations scaled to 0.7, so ~30% shorter sequences at ~1 encoder frame per phoneme.
    # Differs from the run above in exactly this one knob.
    _train_tts_encoder(
        "pseudo-enc-logmel-mfatable-realdur2-lerp-dur07-single-gumbel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
        pseudo_enc_duration_sigma=0.45,
        pseudo_enc_duration_scale=0.7,
        pseudo_enc_lerp=True,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        glow_tts_add_silence_between_words=0.15,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # Same as the dur07 run, but with packed tensors + Inductor compile, to compare speed directly.
    # Bounds come from that run's own config. `capture` defaults to True, so the whole step
    # is captured; only capture_optimizer is off, since grad_explicit reduces between
    # the step and the optimizer.
    _train_tts_encoder(
        "pseudo-enc-logmel-mfatable-realdur2-lerp-dur07-packed-single-gumbel-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
        pseudo_enc_duration_sigma=0.45,
        pseudo_enc_duration_scale=0.7,
        # bounds the upsampling per sequence, so the packed buffer can be sized.
        # measured natural ratio is mean 7.5, and at most 9.8 above 50 phonemes,
        # so 10 guards the tail without reshaping the duration distribution
        pseudo_enc_max_len_factor=10,
        # packing removes padding, so laplace only makes per-step CTC cost climb
        train_seq_ordering="random",
        pseudo_enc_lerp=True,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        glow_tts_add_silence_between_words=0.15,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        behavior_version=29,  # packed tensors need >= 29
        # join right after the pseudo encoder, so no batch-dim select/scatter and no host reads
        pseudo_enc_frontend_concat=True,
        extra_config_updates={
            "optimizer.class": rf.build_dict(Muon)["class"],
            "packed_tensors": True,
            # a compiled step returns the grads, so DDP's autograd hooks never fire.
            # grad_explicit all-reduces them between step and optimizer, matching DDP;
            # 'param' would average params after per-GPU steps, a different optimization.
            "torch_distributed": {"reduce_type": "grad_explicit"},
            "batch_size": None,
            # classes is ~3-4k tokens per batch here, so this never binds on batch size,
            # but it DOES size the CTC FSA edge list (5 * classes bound + 5 * batch bound),
            # which the BW kernels walk every frame -- so slack here is paid every step
            "packed_batch_size": {"data": 11_200_000, "classes": 5_000, "phonemes": 6_000},
            # laplace sorting exists to reduce padding, which packing already removes
            "batching": "random",
            # dead key since phase_timing was removed from RETURNN, kept because it is in the hash
            # and this arm is already trained (ReturnnTrainingJob.A4pDyOmirv1N)
            "phase_timing": True,
            # batch_size_bound is the seq-count capacity every packed buffer bound is derived from,
            # so it must match max_seqs (500 above); the packed batcher enforces that cap.
            # classes gets 80 (cap 75) for the BOS/EOS shifts.
            "torch_cuda_graph": {
                "batch_size_bound": 500,
                "dim_capacity": {"data": 312_000, "classes": 80, "phonemes": 300},
                "warmup_steps": 0,
                "compile": True,
            },
        },
        extra_config_deletes=["optimizer.epsilon"],
    )
    # Packing fits ~26% more seqs per step, so an epoch is 3156 steps instead of 3971 (0.794x),
    # and everything keyed to the step count lands weaker over a run.
    # The mask count: bv28 draws it from each seq's own length, where bv25 used the batch max,
    # which on this data is 252k samples against a true mean of 206k, i.e. 1.227x more;
    # 100/1.227 = 82 restores that strength, 70 goes past it to test whether it was the optimum.
    # specaugment_steps is a STEP schedule, so its ramp finished at ep 7.9 instead of ep 6.3.
    # Decoupled weight decay shrinks by lr*wd per step, so its total is 0.794x as well.
    for _sa_factor in (82, 70):
        _train_asr_base_multigpu(
            f"asr-base-mgpu-logmel-muon-lr5e3-wdbl-nep38-packed-graphc-specaug{_sa_factor}-stepcomp",
            prefix=prefix,
            feature_extraction=None,
            base_lr=1.0,
            peak_lr=5e-3,
            nep=38,
            behavior_version=29,  # packed tensors need >= 29
            extra_config_updates={
                "optimizer.class": rf.build_dict(Muon)["class"],
                "optimizer.weight_decay": 0.0126,  # 0.01 / 0.794
                "packed_tensors": True,
                "torch_distributed": {"reduce_type": "grad_explicit"},
                "batch_size": None,
                # 16_640 = 128 * 130 encoder frames, times the /6 subsample and /160 log-mel,
                # so the bound stays exact through the pipeline instead of landing at 16666.67
                # classes sizes the CTC FSA edge list (5*bound + 5*batch), walked every frame;
                # the finished run peaked at 4270 tokens, so 4352 = 34*128 covers it with 12% fewer edges
                "packed_batch_size": {"data": 16_640 * 6 * configs._batch_size_factor, "classes": 4_352},
                "batching": "random",
                "specaugment_num_spatial_mask_factor": _sa_factor,
                "specaugment_steps": (4000, 12000, 20000),  # (5000, 15000, 25000) * 0.794
                "torch_cuda_graph": {
                    "batch_size_bound": 500,
                    "dim_capacity": {"data": 312_000, "classes": 80},
                    "warmup_steps": 0,
                    "compile": True,
                },
            },
            extra_config_deletes=["optimizer.epsilon"],
        )
    # Front-end injection with RATE-MATCHED durations on the LEARNED embedding
    # (label dur 4-8 at 100Hz ~ 1 enc-frame/phon; completes the mfatable-realdur cell with
    # trainable embeddings -- separates injection depth from effective frames-per-phoneme).
    _train_tts_encoder(
        "pseudo-enc-logmel-noblank-realdur-muon-nep38",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        batch_size_phon=8_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_duration_range=(4, 8),
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # {"dev-clean": 2.00, "dev-other": 4.35, "test-clean": 2.30, "test-other": 4.85}
    # better: -> pseudo-enc-layer4-noblank-muon-nep38
    # SpecAugment width sweep on the pseudo output (6 = the scaled default; 3 / 12 bracket it).
    # _train_tts_encoder(
    #     "pseudo-enc-layer4-noblank-specaug3-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     batch_size_phon=8_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_start_layer=4,
    #     pseudo_enc_blank_duration_range=(0, 0),
    #     pseudo_enc_specaug_max_width=3,
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # {"dev-clean": 1.62, "dev-other": 3.74, "test-clean": 1.82, "test-other": 4.04}
    # better: -> pseudo-enc-layer4-noblank-muon-nep38
    # _train_tts_encoder(
    #     "pseudo-enc-layer4-noblank-specaug12-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     batch_size_phon=8_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_start_layer=4,
    #     pseudo_enc_blank_duration_range=(0, 0),
    #     pseudo_enc_specaug_max_width=12,
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # Encoder-share / layer-split injection (earlier study's insight #1), at the ENC frame rate:
    # the pseudo features live in the encoder model space and enter the Conformer at layer N directly
    # (no conv front-end on the text branch -- this matches the earlier study's injection point;
    # the pseudo-enc-logmel variants instead pass the full front-end incl. 6x subsampling).
    # N=0 = faithful reproduction of the earlier study's setting; N=4/8 share fewer encoder layers.
    # Same duration setting (label 1, blank 0-3), now counted in encoder frames like there.
    # layer0 dropped 2026-06-17: its text branch runs all 16 encoder layers at the un-subsampled enc-rate
    # and OOMs (measured 89.7 GB peak on a long text batch; it is also the worst injection point). See notes.
    # pseudo-enc-layer8: {"dev-clean": 1.68, "dev-other": 3.95, "test-clean": 1.85, "test-other": 4.23}; layer4: {"dev-clean": 1.69, "dev-other": 4.24, "test-clean": 1.92, "test-other": 4.45}
    for _start_layer in (4, 8):
        _train_tts_encoder(
            f"pseudo-enc-layer{_start_layer}",
            prefix=prefix,
            text_train_epoch_split=75,
            batch_size_audio_frames=120_000,
            max_phon_len=300,
            # At the enc rate there is no 6x subsampling on the text branch, so a phoneme costs ~6x
            # more than in the pseudo-enc-logmel variants; 25k phon OOMed at step 1. 8k phon
            # (~20k enc frames at ~2.5 frames/phoneme) matches the audio batch's enc-frame volume.
            batch_size_phon=8_000,
            asr_logmel=True,
            pseudo_speech_enc=True,
            pseudo_enc_start_layer=_start_layer,
            pseudo_enc_specaug_max_width=6,
        )
    # Layer-split depth sweep in the best regime (muon-lr5e3-wdbl + nep38): inject the pseudo-speech at
    # encoder layer N (= share the top 16-N encoder layers). layer8 won at nep25 (3.95, first text-util
    # variant below the no-text baseline). 0-indexed layers "0".."15", so N=16 = share ZERO encoder layers
    # (degenerate endpoint; the collected_outputs guard skips all aux losses there).
    # (layer0 in-flight)
    # layer4: {"dev-clean": 1.58, "dev-other": 3.79, "test-clean": 1.83, "test-other": 4.10}  (best overall)
    # layer8: {"dev-clean": 1.68, "dev-other": 4.11, "test-clean": 1.85, "test-other": 4.23}
    # layer12: {"dev-clean": 1.85, "dev-other": 4.38, "test-clean": 2.03, "test-other": 4.63}
    # layer16: {"dev-clean": 1.95, "dev-other": 4.30, "test-clean": 2.16, "test-other": 4.52}
    for _start_layer in (0, 4, 8, 12, 16):
        _train_tts_encoder(
            f"pseudo-enc-layer{_start_layer}-muon-nep38",
            prefix=prefix,
            text_train_epoch_split=75,
            batch_size_audio_frames=120_000,
            max_phon_len=300,
            batch_size_phon=8_000,
            asr_logmel=True,
            pseudo_speech_enc=True,
            pseudo_enc_start_layer=_start_layer,
            pseudo_enc_specaug_max_width=6,
            base_lr=1.0,
            peak_lr=5e-3,
            nep=38,
            extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
            extra_config_deletes=["optimizer.epsilon"],
        )
    # Reduced-blank layer-split: max 1 blank (uniform 0-1, avg 0.5 -> ~1.5 enc-frames/phoneme vs 2.5 at
    # blank 0-3), so the un-subsampled text sequence is ~40% shorter -> fixes the layer0 OOM and is closer to
    # real speech length. layer0 (OOM-prone) + layer8 (the winner), best regime (muon-lr5e3-wdbl + nep38).
    # (layer0 in-flight)
    # layer8: {"dev-clean": 1.63, "dev-other": 3.93, "test-clean": 1.77, "test-other": 4.13}
    for _start_layer in (0, 8):
        _train_tts_encoder(
            f"pseudo-enc-layer{_start_layer}-blank1-muon-nep38",
            prefix=prefix,
            text_train_epoch_split=75,
            batch_size_audio_frames=120_000,
            max_phon_len=300,
            batch_size_phon=8_000,
            asr_logmel=True,
            pseudo_speech_enc=True,
            pseudo_enc_start_layer=_start_layer,
            pseudo_enc_blank_duration_range=(0, 1),
            pseudo_enc_specaug_max_width=6,
            base_lr=1.0,
            peak_lr=5e-3,
            nep=38,
            extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
            extra_config_deletes=["optimizer.epsilon"],
        )
    # better: -> pseudo-enc-layer4-noblank-muon-nep38
    # layer8 with NO blanks (the phoneme seq already carries [space] silence): label 1 frame at the enc-rate
    # = ~1 enc-frame/phon (still feasible, unlike the front-end noblank which collapsed to ~0.17 enc-frame).
    # Tests whether blanks help layer8 or just bloat it, and whether the noblank regression transfers here.
    # {"dev-clean": 1.62, "dev-other": 3.95, "test-clean": 1.78, "test-other": 4.09}  (best test-other)
    # _train_tts_encoder(
    #     "pseudo-enc-layer8-noblank-muon-nep38",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     batch_size_phon=8_000,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_start_layer=8,
    #     pseudo_enc_blank_duration_range=(0, 0),
    #     pseudo_enc_specaug_max_width=6,
    #     base_lr=1.0,
    #     peak_lr=5e-3,
    #     nep=38,
    #     extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
    #     extra_config_deletes=["optimizer.epsilon"],
    # )
    # better: -> pseudo-enc-logmel-muon-nep38
    # Unit-granularity ablation (CJST-style): the ASR's own spm10k subword units as pseudo-enc input
    # instead of phonemes (identity output mapping; no lexicon/phoneme step needed; the earlier study
    # also used the ASR target units). Same duration setting -> ~3.2x shorter sequences than the
    # phoneme variant (cheapness bonus of coarser units).
    # {"dev-clean": 2.30, "dev-other": 5.11, "test-clean": 2.47, "test-other": 5.51}
    # _train_tts_encoder(
    #     "pseudo-enc-logmel-spm",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     asr_logmel=True,
    #     pseudo_speech_enc=True,
    #     pseudo_enc_units="spm",
    #     pseudo_enc_specaug_max_width=6,
    # )
    # Speech-likeness 2x2, cell B (acoustics=TTS, durations=random): frozen GlowTTS acoustics, but the
    # per-phoneme durations replaced by i.i.d. uniform samples renormalized to the predictor total
    # (same length/cost as ref-match-logmel; only the learned alignment structure is removed).
    # {"dev-clean": 1.83, "dev-other": 4.46, "test-clean": 2.06, "test-other": 4.58}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-rnddur",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     glow_tts_random_durations_jitter=(0.2, 1.8),
    # )
    # Multiplicative duration jitter (online analogue of the offline pipeline's variability finding):
    # w = w_pred * U(0.7, 1.3) per phoneme, NOT renormalized -- keeps the learned alignment structure,
    # varies per-phoneme speaking rate and total length (rnddur removes the structure instead).
    # {"dev-clean": 1.80, "dev-other": 4.40, "test-clean": 1.99, "test-other": 4.69}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-rnddur-mult",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     glow_tts_random_durations_jitter_mult=(0.7, 1.3),
    # )
    # Wide length-scale range 0.1-1.0 (+ per-seq length-scaled SpecAugment): pushes the compression
    # axis below lensamp-low (0.3-0.7) toward the v3b-lenscale-low regime, now with length-aware masking.
    # {"dev-clean": 1.94, "dev-other": 4.54, "test-clean": 2.19, "test-other": 4.89}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-lensamp-wide-specaug",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     specaugment_length_scaled=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(0.1, 1.0),
    # )
    # Synthetic-variability axes on ref-match-logmel (one knob each):
    # noise0 -> deterministic acoustics (no flow sampling noise; ref uses 0.7);
    # 1spk -> single fixed speaker (no voice diversity; ref samples one of 1172 per seq).
    # {"dev-clean": 1.76, "dev-other": 4.26, "test-clean": 1.93, "test-other": 4.66}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-noise0",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.0, 0.0),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    # )
    # {"dev-clean": 1.83, "dev-other": 4.52, "test-clean": 2.10, "test-other": 4.89}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-1spk",
    #     prefix=prefix,
    #     text_train_epoch_split=75,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    #     glow_tts_fixed_speaker=0,
    # )
    # Text-amount axis: text_train_epoch_split 37 / 150 = ~2x / ~0.5x text per audio epoch vs the 75 default.
    # {"dev-clean": 1.76, "dev-other": 4.33, "test-clean": 1.99, "test-other": 4.56}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-textP37",
    #     prefix=prefix,
    #     text_train_epoch_split=37,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    # )
    # {"dev-clean": 1.86, "dev-other": 4.54, "test-clean": 2.08, "test-other": 4.81}
    # DISABLED 2026-07-16: pre-fix-era cell, not rerun after the preload fix (see notes).
    # _train_tts_encoder(
    #     "tts-enc-ref-match-logmel-textP150",
    #     prefix=prefix,
    #     text_train_epoch_split=150,
    #     batch_size_audio_frames=120_000,
    #     max_phon_len=300,
    #     tts_waveform=True,
    #     asr_logmel=True,
    #     tts_waveform_peak_norm=True,
    #     glow_tts_noise_scale_range=(0.7, 0.7),
    #     glow_tts_length_scale_range=(1.0, 1.0),
    # )

    # MFA per-phone mean log-mel table ("table-lookup TTS"), for the upcoming frozen pseudo-enc variant
    # (consistency ladder (d), see projects notes). CPU-only jobs: HF dataset download + mean computation.
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import get_mfa_phone_mean_logmel_table

    get_mfa_phone_mean_logmel_table()

    # TODO: import the finished RZ base-ls-dbmel (ReturnnTrainingJob.8mdaueLDfiGP); do NOT re-train on FZJ.


def _train_asr_base_multigpu(
    name: str,
    *,
    prefix: str,
    feature_extraction: Optional[Dict[str, Any]] = None,
    base_lr: float = 0.5,
    peak_lr: float = 1e-3,
    nep: int = 25,
    behavior_version: int = 25,
    torch_distributed: Optional[Dict[str, Any]] = None,
    batch_size_feat: int = 100_000,
    conv_norm: Optional[Dict[str, Any]] = None,
    feature_norm: Optional[Dict[str, Any]] = None,
    extra_config_updates: Optional[Dict[str, Any]] = None,
    extra_config_deletes: Optional[Sequence[str]] = None,
):
    """
    No-TTS audio-only ASR baseline trained with the FZJ 4-GPU DDP recipe.

    Same architecture + 4-GPU / nep=25 (x4 DDP -> ~100 effective passes) / batched recog as the
    TTS-encoder runs, but trained on LibriSpeech-960 audio only (no TTS, no text data, default AED
    model def, standard train step).

    :param feature_extraction: None -> the default standard log-mel front-end (like base-ls); pass
        ``rf.build_dict(DbMelFeatureExtractor)`` for the DbMel front-end (like the TTS-encoder runs).
        log-mel here vs DbMel here isolates the front-end (mgpu held fixed); log-mel here vs base-ls
        isolates mgpu-vs-single-GPU (front-end held fixed).
    :param base_lr: peak-LR scale of the lrlin/OCLR schedule (for the DDP peak-LR sweep).
    """
    from returnn.frontend.decoder.transformer import TransformerDecoder
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
        train_exp as aed_train_exp,
        _raw_sample_rate,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
    from i6_experiments.users.zeyer.recog_batched import recog_training_exp_batched
    from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_batched import (
        aed_ctc_timesync_recog_recomb_auto_scale_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        model_recog_with_recomb,
    )

    vocab = "spm10k"
    task = get_librispeech_task_raw_v2(vocab=vocab, train_epoch_split=1, train_epoch_wise_filter=None)

    # Same model as the TTS-encoder runs (Conformer L16 D1024 + Transformer dec L6 D1024, DbMel),
    # minus the GlowTTS attachment (default AED model def, no custom train step).
    model_config: Dict[str, Any] = {
        "behavior_version": behavior_version,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            num_layers=16,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "feature_batch_norm": True,
    }
    if feature_extraction is not None:
        model_config["feature_extraction"] = feature_extraction
    if conv_norm is not None:
        # Replace the Conformer conv-block BatchNorm with a batch-independent norm (e.g. GroupNorm).
        model_config["enc_build_dict"]["encoder_layer"]["conv_norm"] = conv_norm
    if feature_norm is not None:
        # Replace the feature BatchNorm front-end with a batch-independent norm (e.g. GroupNorm).
        # NB: key is feature_norm_module, NOT the existing boolean feature_norm option in aed.py.
        model_config["feature_batch_norm"] = False
        model_config["feature_norm_module"] = feature_norm

    exp = aed_train_exp(
        name,
        configs.config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config=model_config,
        config_updates={
            # nep=25 default; with num_processes=4 DDP the data is iterated ~4x/epoch -> ~100 effective
            # passes, matching base-ls (nep=100) and the TTS-encoder runs.
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(nep, base_lr=base_lr, peak_lr=peak_lr),
            # Match base-ls (imported single-GPU LibriSpeech AED baseline, identical model):
            # batch_size 16M + max_seqs 200. The ONLY intended difference here is multi-GPU DDP.
            # Earlier 400k/500, copied from the TTS runs' alternate audio/text batching
            # (where those numbers don't translate), OOM'd this plain-ASR model on 96GB;
            # even 200k/500 still OOM'd at step 0 (~203 seqs -> 91GB).
            "batch_size": batch_size_feat * configs._batch_size_factor,  # = base-ls 16M at the 100k default
            "max_seqs": 200,  # = base-ls
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            # AED BPE-target cap, matching base-ls/v1 (spm10k).
            # This is NOT the frame-sync per-frame "75 trap":
            # for this plain-AED model 75 is the standard cap;
            # None lets very-long-target utterances through
            # and OOMs the decoder at batch_size 64M / max_seqs 500.
            "max_seq_length_default_target": 75,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            # torch DDP across the 4 GH200 (NCCL all-reduce). Required: without it, num_processes=4
            # falls back to horovod and asserts. No unused params here (plain AED, all used every
            # step), so find_unused_parameters can stay off.
            # Default: per-step gradient all-reduce (DDP).
            # Override via torch_distributed=, e.g. {"reduce_type": "param", "param_sync_step": N}
            # for local-SGD-style parameter averaging.
            "torch_distributed": (
                torch_distributed if torch_distributed is not None else {"options": {"find_unused_parameters": False}}
            ),
            "__mem_rqmt": 100,
            "__cpu_rqmt": 72,
            **(extra_config_updates or {}),
        },
        config_deletes=extra_config_deletes,
        post_config_updates={"log_grad_norm": True, "stop_for_resubmission_when_low_time_left": True},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        vocab=vocab,
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        num_processes=4,
        gpu_mem=96,
        recog_training_func=recog_training_exp_batched,
        recog_def=model_recog_with_recomb,  # joint AED+CTC per-epoch recog, fixed scales 1/1
        search_config={"aux_loss_layers": [16]},  # the recog model needs the layer-16 CTC head built
    )
    # Headline AED+CTC first-pass recog (sharded, tuned scales), same as base-ls.
    aed_ctc_timesync_recog_recomb_auto_scale_batched(
        prefix=prefix + "/aed/" + name + "/aed+ctc-batched",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
        num_shards=8,
    )
    return exp


def _train_tts_encoder(
    name: str,
    *,
    prefix: str,
    text_train_epoch_split: int = 20,
    ls_train_epoch_split: int = 1,
    # None = keep the dataset defaults (laplace), so other experiments keep their hashes.
    # "random" for packing: there is no padding to minimise, and length-sorted batches
    # make per-step CTC cost climb (O(frames x targets) at a constant frame budget).
    train_seq_ordering: Optional[str] = None,
    txt_only_loss_scale: float = 1.0,
    glow_tts_length_scale_range=(0.7, 1.1),
    glow_tts_noise_scale_range=(0.3, 0.9),
    batch_size_audio_frames: int = 400_000,
    batch_size_phon: int = 25_000,
    max_phon_len: Optional[int] = None,
    tts_waveform: bool = False,
    asr_logmel: bool = False,
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    enc_aux_logits_share_weights: bool = False,
    enc_aux_logits_with_bias: bool = True,
    pad_audio_rnd: Optional[int] = None,
    single_stream: bool = False,
    pseudo_enc_single_stream_version: int = 1,
    interleave_gumbel_scale: Union[None, float, str] = None,
    specaugment_length_scaled: bool = False,
    tts_waveform_peak_norm: bool = False,
    pseudo_speech_enc: bool = False,
    pseudo_enc_duration_range: Tuple[int, int] = (1, 1),
    pseudo_enc_units: str = "phonemes",
    pseudo_enc_blank_duration_range: Tuple[int, int] = (0, 3),
    pseudo_enc_frozen_table: Optional[tk.Path] = None,
    pseudo_enc_smooth_sigma: Optional[float] = None,
    pseudo_enc_smooth_box_width: Optional[int] = None,
    pseudo_enc_duration_table: Optional[tk.Path] = None,
    pseudo_enc_duration_sigma: Optional[float] = None,
    pseudo_enc_duration_scale: Optional[float] = None,
    pseudo_enc_max_len_factor: Optional[int] = None,
    behavior_version: int = 25,
    pseudo_enc_frontend_concat: bool = False,
    pseudo_enc_lerp: bool = False,
    glow_tts_add_silence_between_words: Optional[float] = None,
    pseudo_enc_start_layer: Optional[int] = None,
    pseudo_enc_specaug_max_width: Optional[int] = None,
    glow_tts_random_durations_jitter: Optional[Tuple[float, float]] = None,
    glow_tts_random_durations_jitter_mult: Optional[Tuple[float, float]] = None,
    glow_tts_fixed_duration: Optional[float] = None,
    glow_tts_fixed_speaker: Optional[int] = None,
    num_processes: int = 4,
    gpu_mem: int = 96,
    nep: int = 25,
    base_lr: float = 0.5,
    peak_lr: Optional[float] = None,
    extra_config_updates: Optional[Dict[str, Any]] = None,
    extra_config_deletes: Optional[Sequence[str]] = None,
):
    from returnn.frontend.decoder.transformer import TransformerDecoder
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
        get_vocab_by_str,
        get_train_corpus_text,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
        train_exp as aed_train_exp,
        _raw_sample_rate,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_batched import (
        aed_ctc_timesync_recog_recomb_auto_scale_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        model_recog_with_recomb,
    )
    from i6_experiments.users.zeyer.recog_batched import recog_training_exp_batched
    from i6_experiments.users.zeyer.returnn.alternate_batching import alternate_batching
    from i6_experiments.users.zeyer.datasets.utils.multi_proc import multi_proc_dataset_opts
    from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
    from i6_experiments.users.zeyer.external_models.glow_tts import (
        get_glow_tts_phone_info,
        get_glow_tts_phoneme_extern_data,
        get_glow_tts_preload_from_files,
        get_glow_tts_gl_preload_from_files,
    )

    # multi-GPU: full-node batched recog; single-GPU (RZ): the standard non-batched recog.
    if num_processes > 1:
        recog_training_func = recog_training_exp_batched
    else:
        from i6_experiments.users.zeyer.recog import recog_training_exp as recog_training_func

    # Every injection mechanism supports single_stream now.
    # Waveform TTS and layer-split pseudo-enc have their own steps;
    # direct-mel TTS and front-end pseudo-enc both merge with the featurized audio rows in model.in_dim.

    vocab = "spm10k"
    task = get_librispeech_task_raw_v2(
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        train_epoch_split=ls_train_epoch_split,
        train_epoch_wise_filter=None,
    )
    task = dataclasses.replace(task)
    base_train = task.train_dataset
    in_key = base_train.get_default_input()  # "data"
    tgt_key = base_train.get_default_target()  # "classes"
    base_extern = dict(base_train.get_extern_data())
    base_eval = base_train.get_eval_datasets()

    # paired ASR audio sub-dataset, with speed perturbation (applied here -- do NOT pass __train_audio_preprocess,
    # else aed_train_exp would try to set it on the DatasetConfigStatic below).
    asr_ds = copy.deepcopy(base_train.get_train_dataset())
    assert asr_ds["class"] == "OggZipDataset", asr_ds["class"]
    if train_seq_ordering:
        asr_ds["seq_ordering"] = train_seq_ordering
    asr_ds["audio"] = dict(asr_ds["audio"])
    asr_ds["audio"]["pre_process"] = speed_pert_librosa_config
    # OggZip decode + speed_pert is the heavy data-loading work; wrap *only* it in MPD.
    # LmDataset is cheap and CombinedDataset does not support sharding, so we keep MPD off the outer levels.
    asr_ds = multi_proc_dataset_opts(asr_ds, num_workers=4)

    # text-only sub-dataset: one LmDataset emits the raw utf8 bytes of each LM line; a PostprocessingDataset
    # derives BOTH the spm target and the GlowTTS phonemes from that text in its map_seq (single corpus read,
    # no second tokenizing LmDataset). The spm + phone_info opts travel via the config (tk.Paths resolved there).
    phon_extern = get_glow_tts_phoneme_extern_data()
    corpus_files = [get_librispeech_normalized_lm_data(), get_train_corpus_text()]
    spm_dim = base_extern[tgt_key]["sparse_dim"]  # spm vocab dim (same object as the audio target)
    phon_dim = phon_extern["sparse_dim"]  # GlowTTS phoneme vocab dim
    text_ds = {
        "class": "PostprocessingDataset",
        # real ordering stays on the inner LmDataset; "default" here (see notes_postprocessing_train_dataset).
        "seq_ordering": "default",
        "dataset": {
            "class": "LmDataset",
            "corpus_file": corpus_files,
            "orth_vocab": {"class": "Utf8ByteTargets"},  # data = raw utf8 bytes of the line
            "use_cache_manager": True,
            "seq_end_symbol": None,
            "unknown_symbol": None,
            "partition_epoch": text_train_epoch_split,
            "seq_ordering": train_seq_ordering or "laplace:.1000",
        },
        "map_seq": functools.partial(_glowtts_text_map_seq, target_key=tgt_key, spm_dim=spm_dim, phon_dim=phon_dim),
        "map_outputs": {
            tgt_key: {"dims": [Dim(None, name="spm_seq")], "sparse_dim": spm_dim, "dtype": "int32"},
            PHONEMES_DATA_KEY: {"dims": [Dim(None, name="phon_seq")], "sparse_dim": phon_dim, "dtype": "int32"},
        },
    }

    # interleave audio + text-only; CombinedDataset zero-fills the missing stream per branch (empty length-0).
    combined = {
        "class": "CombinedDataset",
        "datasets": {"asr": asr_ds, "text": text_ds},
        "data_map": {
            ("asr", in_key): in_key,
            ("asr", tgt_key): tgt_key,
            ("text", tgt_key): tgt_key,
            ("text", PHONEMES_DATA_KEY): PHONEMES_DATA_KEY,
        },
        "seq_ordering": "interleave",
        # Gumbel-max softened interleave (see RETURNN CombinedDataset); only emitted when set (hash-safe)
        **({"interleave_gumbel_scale": interleave_gumbel_scale} if interleave_gumbel_scale is not None else {}),
    }

    extern_data = {**base_extern, PHONEMES_DATA_KEY: phon_extern}
    eval_datasets = {
        k: _wrap_eval_with_empty_phonemes(v, base_extern=base_extern, phon_extern=phon_extern)
        for k, v in base_eval.items()
    }

    task.train_dataset = DatasetConfigStatic(
        main_name="LS ASR + Text(spm+phon)",
        train_dataset=combined,
        default_input=in_key,
        default_target=tgt_key,
        extern_data=extern_data,
        eval_datasets=eval_datasets,
        use_deep_copy=True,
    )

    model_config: Dict[str, Any] = {
        "behavior_version": behavior_version,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            num_layers=16,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "feature_batch_norm": True,
        "feature_extraction": rf.build_dict(DbMelFeatureExtractor),
    }
    # Reference-match knobs (offline reference: auxShared / auxNoBias / AudioPadRnd100).
    # Opt-in only, so existing hashes stay; in model_config so recog rebuilds the same params.
    if enc_aux_logits_share_weights:
        model_config["enc_aux_logits_share_weights"] = True
    if not enc_aux_logits_with_bias:
        model_config["enc_aux_logits_with_bias"] = False
    if pad_audio_rnd:
        model_config["pad_audio"] = {"train": ((0, pad_audio_rnd), (0, pad_audio_rnd))}
    if asr_logmel:
        # Standard log-mel (100Hz) ASR front-end instead of DbMel (80Hz), matching the reference.
        # Requires the waveform path: the frozen-TTS DbMel log-mel can be injected directly only when
        # the ASR front-end is also DbMel; with log-mel the ASR must re-extract from the waveform.
        assert tts_waveform or pseudo_speech_enc, (
            "asr_logmel requires the waveform path (tts_waveform=True), "
            "unless pseudo_speech_enc (whose features live in the front-end space by construction)"
        )
        del model_config["feature_extraction"]  # -> default log_mel_filterbank_from_raw (100Hz)

    # Recog-only model config: the pseudo-speech-encoder flags live in config_updates (training only)
    # and do NOT travel into recog via ModelDefWithCfg (only model_config does), so without this the
    # recog rebuilds the default TTS model (model.tts) and fails to load the pseudo-enc checkpoint
    # (missing tts.* keys). Pass the construction-relevant flags through search_config (per-epoch recog)
    # + extra_config (headline recog) -- recog-only, so the training hash is unchanged.
    # aux_loss_layers: the per-epoch recog is joint AED+CTC (model_recog_with_recomb, fixed scales 1/1),
    # since AED-only per-epoch WERs are unreliable;
    # the recog model needs the layer-16 CTC head built (config_updates is training-only).
    recog_model_cfg = {
        "aux_loss_layers": [16],
        **(
            {
                "pseudo_speech_enc": True,
                "pseudo_enc_duration_range": pseudo_enc_duration_range,
                **(
                    {"pseudo_enc_single_stream_version": pseudo_enc_single_stream_version}
                    if pseudo_enc_single_stream_version != 1
                    else {}
                ),
                "pseudo_enc_blank_duration_range": pseudo_enc_blank_duration_range,
                **({"pseudo_enc_units": pseudo_enc_units} if pseudo_enc_units != "phonemes" else {}),
                **({"pseudo_enc_start_layer": pseudo_enc_start_layer} if pseudo_enc_start_layer is not None else {}),
                **({"pseudo_enc_smooth_sigma": pseudo_enc_smooth_sigma} if pseudo_enc_smooth_sigma is not None else {}),
                **(
                    {"pseudo_enc_smooth_box_width": pseudo_enc_smooth_box_width}
                    if pseudo_enc_smooth_box_width is not None
                    else {}
                ),
                **(
                    {"pseudo_enc_duration_table": pseudo_enc_duration_table}
                    if pseudo_enc_duration_table is not None
                    else {}
                ),
                **(
                    {"pseudo_enc_duration_sigma": pseudo_enc_duration_sigma}
                    if pseudo_enc_duration_sigma is not None
                    else {}
                ),
                **(
                    {"pseudo_enc_duration_scale": pseudo_enc_duration_scale}
                    if pseudo_enc_duration_scale is not None
                    else {}
                ),
                **(
                    {"pseudo_enc_max_len_factor": pseudo_enc_max_len_factor}
                    if pseudo_enc_max_len_factor is not None
                    else {}
                ),
                **({"pseudo_enc_lerp": True} if pseudo_enc_lerp else {}),
            }
            if pseudo_speech_enc
            else {}
        ),
    }
    exp = aed_train_exp(
        name,
        configs.config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        task=task,
        model_def=aed_glowtts_model_def,
        model_config=model_config,
        recog_def=model_recog_with_recomb,  # joint AED+CTC per-epoch recog, fixed scales 1/1
        search_config=recog_model_cfg,
        config_updates={
            # DDP per-rank random_seed_offset iterates the data N=4x per epoch, so divide nep by N
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(
                nep, base_lr=base_lr, **({"peak_lr": peak_lr} if peak_lr is not None else {})
            ),
            # batch_size dict is keyed by ACTUAL data keys; caps are per-variant.
            # NB: under alternate_batching the audio cap (default 400k*factor = 64M samples)
            # is NOT the binding constraint -- these runs peak ~67GB.
            # A full 64M batch of REAL audio (as the plain audio-only asr-base-mgpu baseline packs)
            # costs ~91GB and OOMs the GPU.
            # So 64M here is a loose upper bound under the alternating audio/text batching,
            # NOT a validated plain-audio batch size -- do not copy it into a non-TTS / audio-only config.
            "batch_size": {
                in_key: batch_size_audio_frames * configs._batch_size_factor,
                PHONEMES_DATA_KEY: batch_size_phon,
            },
            "max_seqs": 500,  # let phon cap bind (was 200, capped text bucket too early)
            # max_seq_length: cap outlier phoneme seqs (variant-specific; keeps v1/v3 hashes stable when not set)
            **(
                {
                    "max_seq_length": {
                        in_key: 19.5 * _raw_sample_rate,
                        tgt_key: 75,
                        PHONEMES_DATA_KEY: max_phon_len,
                    }
                }
                if max_phon_len is not None
                else {}
            ),
            "optimizer.weight_decay": 1e-2,
            # single_stream: default torch batching mixes audio + text seqs in ONE batch
            # (offline-reference style); otherwise alternate_batching (pure audio / pure text batches).
            **({} if single_stream else {"torch_batching": functools.partial(alternate_batching, asr_key=in_key)}),
            # custom step; no TrainDef needed
            "train_step": (
                (
                    # layer-split pseudo merges in encoder space and has its own step;
                    # front-end pseudo and direct-mel TTS merge in feature space, so they share this one.
                    aed_pseudo_enc_single_stream_train_step
                    if pseudo_speech_enc and pseudo_enc_start_layer is not None and pseudo_enc_start_layer >= 0
                    else (
                        aed_pseudo_enc_frontend_single_stream_train_step
                        if pseudo_enc_frontend_concat
                        else aed_glowtts_single_stream_train_step
                    )
                )
                if single_stream
                else aed_glowtts_train_step
            ),
            "learning_rate_control_error_measure": "ce",  # set explicitly since there is no TrainDef
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": 75,  # text batches have no audio length cap
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "preload_from_files": {
                # No GlowTTS params for the pseudo-speech-encoder (its embedding is trainable, not preloaded).
                **(get_glow_tts_preload_from_files() if not pseudo_speech_enc else {}),
                **(get_glow_tts_gl_preload_from_files() if tts_waveform else {}),  # GL-net for the waveform path
            },
            # Only emit tts_waveform when True,
            # so the non-waveform models (v1/v2/v3b) keep their original hash
            # (this key is otherwise a no-op for them; model_def reads it via config.bool("tts_waveform", False)).
            # Adding it unconditionally re-hashed finished trainings.
            **({"tts_waveform": True} if tts_waveform else {}),
            # Peak-normalize the synthetic waveform per-seq to match the real-audio path
            # (OggZipDataset peak_normalization=True). Opt-in: the raw-log-mel ASR front-end
            # needs it (else feature_batch_norm NaNs on the arbitrary synthetic scale);
            # the DbMel front-end has its own fixed norm and is left unchanged.
            **({"tts_waveform_peak_norm": True} if tts_waveform_peak_norm else {}),
            "glow_tts_length_scale_range": glow_tts_length_scale_range,
            "glow_tts_noise_scale_range": glow_tts_noise_scale_range,
            **({"specaugment_length_scaled": True} if specaugment_length_scaled else {}),
            **(
                {
                    "pseudo_speech_enc": True,
                    "pseudo_enc_duration_range": pseudo_enc_duration_range,
                    **(
                        {"pseudo_enc_single_stream_version": pseudo_enc_single_stream_version}
                        if pseudo_enc_single_stream_version != 1
                        else {}
                    ),
                    "pseudo_enc_blank_duration_range": pseudo_enc_blank_duration_range,
                }
                if pseudo_speech_enc
                else {}
            ),
            **({"pseudo_enc_units": pseudo_enc_units} if pseudo_speech_enc and pseudo_enc_units != "phonemes" else {}),
            **({"pseudo_enc_frozen_table": pseudo_enc_frozen_table} if pseudo_enc_frozen_table is not None else {}),
            **({"pseudo_enc_start_layer": pseudo_enc_start_layer} if pseudo_enc_start_layer is not None else {}),
            **({"pseudo_enc_smooth_sigma": pseudo_enc_smooth_sigma} if pseudo_enc_smooth_sigma is not None else {}),
            **(
                {"pseudo_enc_smooth_box_width": pseudo_enc_smooth_box_width}
                if pseudo_enc_smooth_box_width is not None
                else {}
            ),
            **(
                {"pseudo_enc_duration_table": pseudo_enc_duration_table}
                if pseudo_enc_duration_table is not None
                else {}
            ),
            **(
                {"pseudo_enc_duration_sigma": pseudo_enc_duration_sigma}
                if pseudo_enc_duration_sigma is not None
                else {}
            ),
            **(
                {"pseudo_enc_duration_scale": pseudo_enc_duration_scale}
                if pseudo_enc_duration_scale is not None
                else {}
            ),
            **(
                {"pseudo_enc_max_len_factor": pseudo_enc_max_len_factor}
                if pseudo_enc_max_len_factor is not None
                else {}
            ),
            **({"pseudo_enc_lerp": True} if pseudo_enc_lerp else {}),
            **(
                {"pseudo_enc_specaug_max_width": pseudo_enc_specaug_max_width}
                if pseudo_enc_specaug_max_width is not None
                else {}
            ),
            **(
                {"glow_tts_random_durations_jitter": glow_tts_random_durations_jitter}
                if glow_tts_random_durations_jitter is not None
                else {}
            ),
            **(
                {"glow_tts_random_durations_jitter_mult": glow_tts_random_durations_jitter_mult}
                if glow_tts_random_durations_jitter_mult is not None
                else {}
            ),
            **({"glow_tts_fixed_duration": glow_tts_fixed_duration} if glow_tts_fixed_duration is not None else {}),
            **({"glow_tts_fixed_speaker": glow_tts_fixed_speaker} if glow_tts_fixed_speaker is not None else {}),
            # dual-stream only: the single-stream step has ONE loss set (no txt_ prefix, no extra scale)
            **({} if single_stream else {"txt_only_loss_scale": txt_only_loss_scale, "separate_txt_only_losses": True}),
            # text-only data pipeline: the PostprocessingDataset map_seq reads these to tokenize raw text into
            # spm targets + GlowTTS phonemes (nested tk.Paths are resolved as config values).
            # train-targets vocab; with train_vocab_opts (e.g. SamplingBPE) the text stream samples too
            "glow_tts_text_spm_opts": (
                get_vocab_by_str(vocab).copy(**train_vocab_opts) if train_vocab_opts else get_vocab_by_str(vocab)
            ).get_opts(),
            "glow_tts_phone_info": get_glow_tts_phone_info(
                train=True, add_silence_between_words=glow_tts_add_silence_between_words
            ),
            # DDP across 4 GH200: each runs a model replica + its data shard, gradients all-reduced via NCCL.
            # ~400 M trainable params (Enc L16 D1024 + Dec L6 D1024 + spm10k) fits one GH200 (95 GB) easily,
            # so DDP is the right tool here -- FSDP would only add sharding overhead for no memory benefit.
            # alternate_batching alternates audio-batch vs text-only-batch:
            # GlowTTS only fires on text, so its params are unused on audio steps
            **({"torch_distributed": {"options": {"find_unused_parameters": True}}} if num_processes > 1 else {}),
            # Use most of the JUPITER node (~480 GiB total, ~440 usable): 4 DDP procs * (1 main + MPD workers)
            # each load the ~4 GB LM corpus, plus model + activations. ~400 GiB gives comfortable headroom.
            # per-DDP-proc; sis multiplies by num_processes=4 -> 400 GiB total (node has ~480 GiB usable)
            "__mem_rqmt": 100 if num_processes > 1 else 60,
            # 4 GPUs * 72 cores/Grace-Hopper = full node (288 cores). Helps MPD workers / data loading.
            "__cpu_rqmt": 72 if num_processes > 1 else 24,
            **(extra_config_updates or {}),
        },
        config_deletes=extra_config_deletes,
        post_config_updates={
            "log_grad_norm": True,
            # opt out of train_v4's default outer-MPD; CombinedDataset can't shard. MPD is wrapped around OggZip above.
            "__multi_proc_dataset": False,
            "stop_for_resubmission_when_low_time_left": True,
        },
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        # single-GPU must pass None, NOT 1:
        # any num_processes sets horovod_num_processes and picks a distributed launcher,
        # and without torch_distributed returnn falls back to horovod -> TF assert.
        num_processes=num_processes if num_processes > 1 else None,
        gpu_mem=gpu_mem,
        recog_training_func=recog_training_func,
    )
    # Joint AED+CTC first-pass recog with tuned scales. multi-GPU: sharded over the full node
    # (batched); single-GPU (RZ): the standard non-batched recog. This is the headline WER.
    if num_processes > 1:
        aed_ctc_timesync_recog_recomb_auto_scale_batched(
            prefix=prefix + "/aed/" + name + "/aed+ctc-batched",
            task=task,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
            num_shards=8,
            extra_config=recog_model_cfg,
        )
    else:
        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
            aed_ctc_timesync_recog_recomb_auto_scale,
        )

        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
            # construction-relevant flags (pseudo-enc etc.), like the batched branch above;
            # without this, pseudo-enc recogs rebuild the default TTS model and fail to load.
            extra_config=recog_model_cfg,
        )
    return exp


def aed_glowtts_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Standard aed.Model + frozen GlowTTS attached as model.tts (log-mel out_dim == encoder in_dim).
    With config pseudo_speech_enc, a trainable PseudoSpeechEncoder is attached as model.pseudo_enc
    instead (no TTS)."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    model = _aed.aed_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    model.tts = None
    model.pseudo_enc = None
    if config.bool("pseudo_speech_enc", False):
        # Pseudo-speech-encoder instead of the frozen TTS: trainable token embedding + random durations,
        # the simple/cheap end of the speech-likeness axis.
        # Units: "phonemes" (default), or "spm" = the ASR's own subword units (CJST-style; no lexicon).
        units = config.typed_value("pseudo_enc_units", "phonemes")
        if units == "spm":
            vocab_dim = target_dim
        else:
            assert units == "phonemes", f"unknown pseudo_enc_units {units!r}"
            vocab_dim = Dim(get_glow_tts_phoneme_vocab_size(), name="glowtts_phonemes")
        # Layer-split injection (pseudo_enc_start_layer >= 0): the pseudo features live in the encoder
        # model space at the subsampled encoder frame rate, entering the Conformer at that layer.
        start_layer = config.int("pseudo_enc_start_layer", -1)
        model.pseudo_enc = PseudoSpeechEncoder(
            vocab_dim=vocab_dim,
            out_dim=model.encoder.out_dim if start_layer >= 0 else model.in_dim,
            label_duration_range=tuple(config.typed_value("pseudo_enc_duration_range", (1, 1))),
            blank_duration_range=tuple(config.typed_value("pseudo_enc_blank_duration_range", (0, 3))),
            smooth_sigma=config.typed_value("pseudo_enc_smooth_sigma", None),
            smooth_box_width=config.typed_value("pseudo_enc_smooth_box_width", None),
            duration_table=config.typed_value("pseudo_enc_duration_table", None),
            duration_sigma=config.float("pseudo_enc_duration_sigma", 0.45),
            duration_scale=config.float("pseudo_enc_duration_scale", 1.0),
            max_len_factor=config.typed_value("pseudo_enc_max_len_factor", None),
            lerp=config.bool("pseudo_enc_lerp", False),
        )
        frozen_table = config.typed_value("pseudo_enc_frozen_table", None)
        if frozen_table:
            import numpy

            assert units == "phonemes", "pseudo_enc_frozen_table is a phoneme-vocab table"
            assert start_layer < 0, "the frozen table lives in the front-end feature space"
            npz = numpy.load(frozen_table, allow_pickle=True)
            means, table_labels = npz["means"], list(npz["labels"])
            assert means.shape == (vocab_dim.dimension, model.in_dim.dimension)
            # Blank row := [space] (real silence acoustics; silence and blank share one row by design;
            # with blank duration 0 the blank row is never emitted anyway).
            table = numpy.concatenate([means, means[table_labels.index("[space]")][None]], axis=0)
            model.pseudo_enc.embedding.weight.initial = table
            model.pseudo_enc.embedding.weight.trainable = False
        return model
    noise_scale_range = config.typed_value("glow_tts_noise_scale_range", (0.3, 0.9))
    length_scale_range = config.typed_value("glow_tts_length_scale_range", (0.7, 1.1))
    rnd_dur_jitter = config.typed_value("glow_tts_random_durations_jitter", None)
    rnd_dur_jitter_mult = config.typed_value("glow_tts_random_durations_jitter_mult", None)
    model.tts = GlowTtsLogMel(
        phoneme_vocab_dim=Dim(get_glow_tts_phoneme_vocab_size(), name="glowtts_phonemes"),
        out_dim=model.in_dim,  # GlowTTS log-mel feeds straight into the encoder (same DbMel space)
        glow_tts_noise_scale_range=tuple(noise_scale_range),
        glow_tts_length_scale_range=tuple(length_scale_range),
        return_waveform=config.bool("tts_waveform", False),  # waveform mode: GL-net + Griffin-Lim -> ASR front-end
        peak_normalize_waveform=config.bool("tts_waveform_peak_norm", False),
        random_durations_jitter=tuple(rnd_dur_jitter) if rnd_dur_jitter is not None else None,
        random_durations_jitter_mult=tuple(rnd_dur_jitter_mult) if rnd_dur_jitter_mult is not None else None,
        fixed_duration=config.typed_value("glow_tts_fixed_duration", None),
        fixed_speaker=config.typed_value("glow_tts_fixed_speaker", None),
    )
    # GlowTTS is frozen: imported params, never updated.
    for p in model.tts.parameters():
        p.trainable = False
    return model


aed_glowtts_model_def: ModelDef[Model]
aed_glowtts_model_def.behavior_version = 25
aed_glowtts_model_def.backend = "torch"
aed_glowtts_model_def.batch_size_factor = _aed.aed_model_def.batch_size_factor


def _tts_specaug_max_spatial_dims(config, sampled_scales):
    """Per-seq SpecAugment time-mask width, scaled by the sampled length_scale, so compressed synthetic
    sequences are not over-masked (SpecAugment's absolute width erases short low-length_scale seqs).
    Returns None (-> baseline absolute width) unless specaugment_length_scaled is set."""
    if not config.bool("specaugment_length_scaled", False):
        return None
    base = config.typed_value("specaugment_max_consecutive_spatial_dims") or 20
    return rf.maximum(rf.cast(sampled_scales["length_scale"] * float(base), "int32"), 1)


class PseudoSpeechEncoder(rf.Module):
    """Trainable pseudo-speech encoder for text injection (no TTS),
    following the earlier pseudo-speech-encoder study's simple winning design
    (cf. RandomDurationPredictor + interleaver_mode="targets" there):
    the label sequence is interleaved with a blank index, CTC-alignment-style
    (blank, l0, blank, l1, ..., blank), per-position random durations are sampled
    separately for labels and blanks (blanks may get duration 0),
    then a single embedding lookup over the with-blank vocab and upsampling via rf.repeat.
    Produces pseudo feature frames in the ASR front-end feature space (in_dim),
    fed through Model.encode_from_features.
    Unlike the frozen TTS path, this is trained jointly with the ASR model (no stop_gradient)."""

    def __init__(
        self,
        *,
        vocab_dim: Dim,
        out_dim: Dim,
        label_duration_range: Tuple[int, int] = (1, 1),
        blank_duration_range: Tuple[int, int] = (0, 3),
        smooth_sigma: Optional[float] = None,
        smooth_box_width: Optional[int] = None,
        duration_table: Optional[str] = None,
        duration_sigma: float = 0.45,
        duration_scale: float = 1.0,
        max_len_factor: Optional[int] = None,
        lerp: bool = False,
    ):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.out_dim = out_dim
        # Optional temporal smoothing of the upsampled step function (None = off).
        # A Gaussian rounds the corners but keeps a plateau inside long segments;
        # a box of width d is exact linear interpolation instead. Frozen either way, so no trained params.
        assert not (smooth_sigma and smooth_box_width), "smooth_sigma and smooth_box_width are exclusive"
        self.smooth_sigma = smooth_sigma
        self.smooth_box_width = smooth_box_width
        self.smooth = None
        if smooth_sigma or smooth_box_width:
            import numpy
        if smooth_sigma:
            half = max(1, int(numpy.ceil(3.0 * smooth_sigma)))
            kernel = numpy.exp(-0.5 * ((numpy.arange(2 * half + 1) - half) / smooth_sigma) ** 2)
            self.smooth = self._make_smoother(out_dim, kernel)
        elif smooth_box_width:
            self.smooth = self._make_smoother(out_dim, numpy.ones(smooth_box_width))
        # Durations in output frames, sampled uniform int [lo, hi] per position.
        # Defaults follow the earlier study's winning setting:
        # labels always exactly 1 frame, blanks 0-3 frames (mean total ~2.5 frames per label).
        assert label_duration_range[0] >= 1
        self.label_duration_range = label_duration_range
        self.blank_duration_range = blank_duration_range
        self.blank_idx = vocab_dim.dimension
        self.wb_vocab_dim = vocab_dim + 1
        self.embedding = rf.Embedding(self.wb_vocab_dim, out_dim)
        # Per-phone durations from the MFA alignments (None = the uniform label_duration_range).
        # Real durations are right-skewed and span ~3.4x across phones, which a uniform range cannot represent.
        # Data, not a weight, so it stays out of the checkpoint and is reloaded from the path at recog.
        self.duration_sigma = duration_sigma
        # Shortens the sequences without touching the shape: lognormal is scale-invariant,
        # so the skew and the per-phone ratios survive. Below ~0.7 the phones fall under one
        # encoder frame each (6x subsampling), which is what made the old label-1 setup unusable.
        self.duration_scale = duration_scale
        # Bounds the whole sequence, not the single label: the upsampled length stays within
        # max_len_factor * the input length. Without such a bound the packed buffer for the
        # repeat cannot be sized. Long phones keep their length as long as the total fits.
        self.max_len_factor = max_len_factor
        self.duration_medians = None
        if duration_table:
            import numpy

            npz = numpy.load(duration_table, allow_pickle=True)
            med = npz["medians"].astype("float32")
            assert med.shape == (vocab_dim.dimension,), f"duration table {med.shape} vs vocab {vocab_dim}"
            # Blank row: 1 frame. Blank is not a phone, and with blank_duration_range=(0,0)
            # (the standard here) it is never emitted anyway.
            self.duration_medians = numpy.concatenate([med, numpy.ones((1,), dtype="float32")])
        # Linear interpolation between successive phoneme targets instead of a step function.
        # Real log-mel has essentially no plateaus, which lerp reproduces and a Gaussian does not.
        # With per-phone durations the box shortcut no longer applies: one width cannot fit 5..17 frames.
        self.lerp = lerp

    def __call__(self, labels: Tensor, *, spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        assert labels.sparse_dim.dimension == self.vocab_dim.dimension, (
            f"vocab size mismatch: {labels.sparse_dim} vs {self.vocab_dim}"
        )
        # Fast path: exactly one frame per label and no blanks -> nothing to interleave or repeat.
        # Only valid when durations are uniform-1 and no interpolation is requested.
        if (
            self.duration_medians is None
            and not self.lerp
            and self.blank_duration_range == (0, 0)
            and self.label_duration_range == (1, 1)
        ):
            # The embedding is over the with-blank vocab, so re-tag the sparse dim.
            # The values are all < blank_idx here; only the dim label differs.
            labels_wb = labels.copy()
            labels_wb.sparse_dim = self.wb_vocab_dim
            emb = self.embedding(labels_wb)
            emb.feature_dim = self.out_dim
            out_spatial_dim = spatial_dim
        elif self.blank_duration_range == (0, 0):
            import numpy

            # No blanks, so the interleave would only add zero-duration positions.
            # Using the labels directly keeps this rf-only: packed-friendly and traceable.
            labels_wb = labels.copy()
            labels_wb.sparse_dim = self.wb_vocab_dim
            # Durations are sampled only in training:
            # in eval the jitter drops to 1 and the range to its middle,
            # so a dev score does not depend on the draw.
            # rf.where, not a Python if, since train_flag can be a Tensor.
            train = rf.get_run_ctx().train_flag
            if self.duration_medians is not None:
                med = rf.convert_to_tensor(numpy.asarray(self.duration_medians), dims=[self.wb_vocab_dim])
                base = rf.gather(med, indices=labels_wb, axis=self.wb_vocab_dim) * self.duration_scale
                jitter = rf.exp(rf.random_normal(base.dims, dtype="float32") * self.duration_sigma)
                jitter = rf.where(train, jitter, 1.0)
                durations = rf.maximum(rf.cast(base * jitter + 0.5, "int32"), 1)
            else:
                l_lo, l_hi = self.label_duration_range
                durations = rf.where(
                    train,
                    rf.random_uniform(labels.dims, minval=l_lo, maxval=l_hi + 1, dtype="int32"),
                    rf.constant((l_lo + l_hi) // 2, dims=labels.dims, dtype="int32"),
                )
            durations = durations.copy_masked(0, dims=[spatial_dim])
            if self.max_len_factor:
                # Scale here, not inside rf.repeat alone, so that seg_start below is derived
                # from exactly the durations the repeat expands by.
                durations = rf.limit_repeats(durations, in_spatial_dim=spatial_dim, max_len_factor=self.max_len_factor)
            if not self.lerp:
                rep, out_spatial_dim = rf.repeat(
                    labels_wb,
                    in_spatial_dim=spatial_dim,
                    repeats=durations,
                    max_len_factor=self.max_len_factor,
                )
                emb = self.embedding(rep)
                emb.feature_dim = self.out_dim
            else:
                # Each frame glides from its own target toward the next, by position in the segment.
                # Without blanks the successor is simply the next label; the last one keeps its own.
                nxt = rf.gather(
                    labels_wb, indices=rf.range_over_dim(spatial_dim) + 1, axis=spatial_dim, clip_to_valid=True
                )
                seg_start = rf.cumsum(durations, spatial_dim=spatial_dim) - durations
                cur_rep, out_spatial_dim = rf.repeat(
                    labels_wb,
                    in_spatial_dim=spatial_dim,
                    repeats=durations,
                    max_len_factor=self.max_len_factor,
                )
                nxt_rep, _ = rf.repeat(
                    nxt, in_spatial_dim=spatial_dim, repeats=durations, out_spatial_dim=out_spatial_dim
                )
                start_rep, _ = rf.repeat(
                    seg_start, in_spatial_dim=spatial_dim, repeats=durations, out_spatial_dim=out_spatial_dim
                )
                dur_rep, _ = rf.repeat(
                    durations, in_spatial_dim=spatial_dim, repeats=durations, out_spatial_dim=out_spatial_dim
                )
                pos = rf.range_over_dim(out_spatial_dim)
                frac = rf.cast(pos - start_rep, "float32") / rf.cast(rf.maximum(dur_rep, 1), "float32")
                emb_cur = self.embedding(cur_rep)
                emb_nxt = self.embedding(nxt_rep)
                frac = rf.cast(frac, emb_cur.dtype)
                emb = emb_cur * (1.0 - frac) + emb_nxt * frac
                emb.feature_dim = self.out_dim
        else:
            import torch

            (batch_dim,) = labels.remaining_dims(spatial_dim)
            ids_raw = labels.copy_compatible_to_dims_raw([batch_dim, spatial_dim])  # [B, T] int
            bs, t = ids_raw.shape
            dev = ids_raw.device
            # Interleave with the blank index: [blank, l0, blank, l1, ..., l_{T-1}, blank],
            # per-seq length 2*len+1 (padding positions are blank too; masked via the dyn dim).
            inter_raw = torch.full((bs, 2 * t + 1), self.blank_idx, dtype=ids_raw.dtype, device=dev)
            inter_raw[:, 1::2] = ids_raw
            inter_dim = spatial_dim * 2 + 1
            interleaved = rf.convert_to_tensor(inter_raw, dims=[batch_dim, inter_dim], sparse_dim=self.wb_vocab_dim)
            # Durations: blanks (even positions) always from blank_duration_range; labels (odd
            # positions) either per-phone lognormal from the MFA table, or the uniform range.
            train = rf.get_run_ctx().train_flag
            b_lo, b_hi = self.blank_duration_range
            blank_dur = rf.where(
                train,
                rf.random_uniform(interleaved.dims, minval=b_lo, maxval=b_hi + 1, dtype="int32"),
                rf.constant((b_lo + b_hi) // 2, dims=interleaved.dims, dtype="int32"),
            )
            if self.duration_medians is not None:
                med = torch.tensor(self.duration_medians, device=dev)  # [wb_vocab], tiny
                base = med[inter_raw.long()] * self.duration_scale  # [B, S] per-position median duration
                jitter = torch.exp(self.duration_sigma * torch.randn(base.shape, device=dev))
                sampled = torch.clamp(torch.round(base * jitter), min=1.0).to(torch.int32)
                median = torch.clamp(torch.round(base), min=1.0).to(torch.int32)
                label_dur = rf.where(
                    train,
                    rf.convert_to_tensor(sampled, dims=[batch_dim, inter_dim]),
                    rf.convert_to_tensor(median, dims=[batch_dim, inter_dim]),
                )
            else:
                l_lo, l_hi = self.label_duration_range
                label_dur = rf.where(
                    train,
                    rf.random_uniform(interleaved.dims, minval=l_lo, maxval=l_hi + 1, dtype="int32"),
                    rf.constant((l_lo + l_hi) // 2, dims=interleaved.dims, dtype="int32"),
                )
            durations = rf.where(rf.range_over_dim(inter_dim) % 2 == 0, blank_dur, label_dur)
            durations = durations.copy_masked(0, dims=[inter_dim])
            if not self.lerp:
                interleaved, out_spatial_dim = rf.repeat(interleaved, in_spatial_dim=inter_dim, repeats=durations)
                emb = self.embedding(interleaved)
                emb.feature_dim = self.out_dim
            else:
                # Each frame glides from its own phoneme target toward the next, by position in the segment.
                # Frame 0 is exactly the phoneme's own embedding, and no duration leaves a flat middle.
                # Shift by two: the sequence is interleaved, so the successor sits 2 positions away.
                assert self.blank_duration_range == (0, 0), "lerp assumes no blanks (glide target skips them)"
                nxt_raw = torch.cat([inter_raw[:, 2:], inter_raw[:, -2:]], dim=1)
                nxt = rf.convert_to_tensor(nxt_raw, dims=[batch_dim, inter_dim], sparse_dim=self.wb_vocab_dim)
                dur_raw = durations.copy_compatible_to_dims_raw([batch_dim, inter_dim]).to(torch.int64)
                start_raw = (torch.cumsum(dur_raw, dim=1) - dur_raw).to(torch.int32)  # exclusive cumsum
                seg_start = rf.convert_to_tensor(start_raw, dims=[batch_dim, inter_dim])
                # All four repeats share one out_spatial_dim, else they would not align in time.
                cur_rep, out_spatial_dim = rf.repeat(interleaved, in_spatial_dim=inter_dim, repeats=durations)
                nxt_rep, _ = rf.repeat(
                    nxt, in_spatial_dim=inter_dim, repeats=durations, out_spatial_dim=out_spatial_dim
                )
                start_rep, _ = rf.repeat(
                    seg_start, in_spatial_dim=inter_dim, repeats=durations, out_spatial_dim=out_spatial_dim
                )
                dur_rep, _ = rf.repeat(
                    durations, in_spatial_dim=inter_dim, repeats=durations, out_spatial_dim=out_spatial_dim
                )
                pos = rf.range_over_dim(out_spatial_dim)
                # clamp the denominator: padded positions carry no meaningful duration
                frac = rf.cast(pos - start_rep, "float32") / rf.cast(rf.maximum(dur_rep, 1), "float32")
                emb_cur = self.embedding(cur_rep)
                emb_nxt = self.embedding(nxt_rep)
                frac = rf.cast(frac, emb_cur.dtype)
                emb = emb_cur * (1.0 - frac) + emb_nxt * frac
                emb.feature_dim = self.out_dim
        if self.smooth is not None:
            # out_spatial_dim passed explicitly so "same" padding reuses that dim
            # instead of minting a new one (downstream dims must keep matching).
            emb, _ = self.smooth(emb, in_spatial_dim=out_spatial_dim, out_spatial_dim=out_spatial_dim)
            emb.feature_dim = self.out_dim
        return emb, out_spatial_dim

    @staticmethod
    def _make_smoother(out_dim: Dim, kernel) -> rf.Conv1d:
        """Depthwise (groups = feature dim) 1D conv with a frozen kernel: a pure temporal blur,
        identical per channel, no cross-feature mixing and no trained parameters.
        The kernel is renormalized to sum 1 (DC gain 1), so feature magnitudes are preserved."""
        import numpy

        kernel = numpy.asarray(kernel, dtype="float64")
        kernel = kernel / kernel.sum()
        width = int(kernel.shape[0])
        conv = rf.Conv1d(out_dim, out_dim, width, padding="same", with_bias=False, groups=out_dim.dimension)
        # filter is [out_dim, in_dim // groups = 1, width]; broadcast the one kernel over all channels.
        conv.filter.initial = numpy.broadcast_to(kernel[None, None, :], (out_dim.dimension, 1, width)).astype("float32")
        conv.filter.trainable = False
        return conv


def aed_glowtts_train_step(*, model: Model, extern_data, **_kwargs_unused):
    """Custom RETURNN train_step: dual-branch AED+CTC (paired audio + text-only via GlowTTS).

    Does its own extern_data extraction -- including the 3rd ``phonemes`` stream that the default train_v4 step
    does not forward -- and the loss computation, so no separate TrainDef is needed. Pass as config["train_step"].

    alternate_batching makes each batch pure audio or pure text-only; we branch on whether the audio stream is
    empty. Audio batch -> model.encode. Text batch -> model.tts(phonemes) (frozen) -> model.encode_from_features.
    Then the standard aux-CTC + AED-CE losses (a "txt_" prefix + txt_only_loss_scale for the text branch).
    """
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict
    import torch

    config = get_global_config()  # noqa
    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[config.typed_value("target")]
    targets_spatial_dim = targets.get_time_dim_tag()
    phonemes = extern_data[PHONEMES_DATA_KEY]
    phonemes_spatial_dim = phonemes.get_time_dim_tag()

    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert isinstance(use_normalized_loss, str) and use_normalized_loss in ("none", "frames", "seqs")
    label_smoothing = config.float("label_smoothing", 0.1)
    aux_ctc_label_smoothing = config.float("aux_ctc_label_smoothing", 0.0)
    text_augment = config.typed_value("text_augment", None)
    separate_txt_only_losses = config.bool("separate_txt_only_losses", True)
    txt_only_loss_scale = config.float("txt_only_loss_scale", 1.0)

    ctc_loss = rf.ctc_loss
    if aux_ctc_label_smoothing:
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        ctc_loss = ctc_loss_fixed_grad

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)

    # alternate_batching => each batch is pure audio or pure text-only. Branch on whether audio is empty.
    audio_sizes_cpu = data_spatial_dim.get_size_tensor()  # [B] on CPU
    have_audio = bool(rf.reduce_max(audio_sizes_cpu, axis=audio_sizes_cpu.dims).raw_tensor.item() > 0)
    if have_audio:
        loss_prefix = ""
        global_loss_scale = 1.0
    else:
        loss_prefix = "txt_" if separate_txt_only_losses else ""
        global_loss_scale = txt_only_loss_scale

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    collected_outputs = CollectOutputsDict(allowed_key_patterns=[str(layer_idx - 1) for layer_idx in aux_loss_layers])
    if have_audio:
        enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    elif model.pseudo_enc is not None:
        # Pseudo-speech-encoder text branch: TRAINABLE embedding upsampled by random durations
        # (no TTS, no stop_gradient -- the pseudo encoder learns jointly with the ASR model).
        if config.typed_value("pseudo_enc_units", "phonemes") == "spm":
            # CJST-style: the ASR's own subword units as injection input (identity output mapping).
            pseudo_src, pseudo_src_spatial_dim = targets, targets_spatial_dim
        else:
            pseudo_src, pseudo_src_spatial_dim = phonemes, phonemes_spatial_dim
        feats, feats_spatial_dim = model.pseudo_enc(pseudo_src, spatial_dim=pseudo_src_spatial_dim)
        start_layer = config.int("pseudo_enc_start_layer", -1)
        if start_layer >= 0:
            # Layer-split injection: the pseudo features live in the encoder model space and enter the
            # Conformer at layer start_layer directly (no conv front-end on the text branch -- the
            # earlier study's injection point). Aux losses below start_layer are skipped via the
            # collected_outputs guard in the loop below.
            enc_raw, enc_spatial_dim = model.encode_from_enc_space(
                feats,
                spatial_dim=feats_spatial_dim,
                start_layer=start_layer,
                collected_outputs=collected_outputs,
                specaugment_max_spatial_dims=config.typed_value("pseudo_enc_specaug_max_width", None),
            )
        else:
            enc_raw, enc_spatial_dim = model.encode_from_features(
                feats,
                in_spatial_dim=feats_spatial_dim,
                collected_outputs=collected_outputs,
                # Pseudo sequences are much shorter than real speech (~3x), so the absolute SpecAugment
                # time-mask width over-masks them; scale it down (cf. specaugment_length_scaled for TTS).
                specaugment_max_spatial_dims=config.typed_value("pseudo_enc_specaug_max_width", None),
            )
        enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)
    elif config.bool("tts_waveform", False):
        # waveform mode: model.tts returns a synthetic WAVEFORM (GlowTTS->GL-net->Griffin-Lim) that
        # re-enters the ASR through its own feature front-end -- identical path to real audio (line above).
        sampled_scales = {}
        # no_grad: the TTS is frozen;
        # without it, all TTS activations are kept for a backward that never happens (GPU OOM).
        with torch.no_grad():
            wave, wave_spatial_dim = model.tts(
                phonemes, spatial_dim=phonemes_spatial_dim, out_sampled_scales=sampled_scales
            )
        wave = rf.stop_gradient(wave)  # TTS is frozen
        enc, enc_spatial_dim = model.encode(
            wave,
            in_spatial_dim=wave_spatial_dim,
            collected_outputs=collected_outputs,
            specaugment_max_spatial_dims=_tts_specaug_max_spatial_dims(config, sampled_scales),
        )
    else:
        sampled_scales = {}
        # no_grad: see the waveform branch above.
        with torch.no_grad():
            log_mel, log_mel_spatial_dim = model.tts(
                phonemes, spatial_dim=phonemes_spatial_dim, out_sampled_scales=sampled_scales
            )
        log_mel = rf.stop_gradient(log_mel)  # TTS is frozen
        enc_raw, enc_spatial_dim = model.encode_from_features(
            log_mel,
            in_spatial_dim=log_mel_spatial_dim,
            collected_outputs=collected_outputs,
            specaugment_max_spatial_dims=_tts_specaug_max_spatial_dims(config, sampled_scales),
        )
        enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)

    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
            continue
        if str(layer_idx - 1) not in collected_outputs:
            continue  # text branch entering above this layer (layer-split injection)
        linear = getattr(model, f"enc_aux_logits_{layer_idx}")
        aux_logits = linear(collected_outputs[str(layer_idx - 1)])
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        if aux_ctc_label_smoothing:
            aux_ctc_log_probs = rf.label_smoothed_log_prob_gradient(
                aux_ctc_log_probs, smoothing=aux_ctc_label_smoothing, axis=model.wb_target_dim
            )
        aux_loss = ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=ctc_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=ctc_targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        if use_normalized_loss in ("none", "frames"):
            aux_loss.mark_as_loss(
                f"{loss_prefix}ctc_{layer_idx}",
                scale=aux_loss_scales[i] * global_loss_scale,
                custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(),
                use_normalized_loss={"none": False, "frames": True}[use_normalized_loss],
            )
        elif use_normalized_loss == "seqs":
            aux_loss.mark_as_loss(
                f"{loss_prefix}ctc_{layer_idx}",
                scale=0,
                custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(),
            )
            aux_loss.mark_as_loss(
                f"{loss_prefix}seq_ctc_{layer_idx}",
                scale=aux_loss_scales[i] * global_loss_scale,
                use_normalized_loss=True,
            )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )
    if text_augment:
        input_labels, targets_w_eos, targets_w_eos_spatial_dim = rf.cond(
            rf.get_run_ctx().train_flag,
            lambda: text_augment(
                input_labels=input_labels,
                targets_w_eos=targets_w_eos,
                spatial_dim=targets_w_eos_spatial_dim,
                exclude_labels={model.bos_idx, model.eos_idx},
            ),
            lambda: (input_labels, targets_w_eos, targets_w_eos_spatial_dim),
        )

    collected_outputs = CollectOutputsDict(
        allowed_key_patterns=[str(layer_idx - 1) for layer_idx in dec_aux_loss_layers]
    )
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=collected_outputs,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        out = collected_outputs[str(layer_idx - 1)]
        dec_aux_logits[layer_idx] = linear(norm(out))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )
        if not model.out_eos_separated:
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:
            log_prob = _aed.log_probs_with_eos_separated(
                logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx
            )
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        if use_normalized_loss in ("none", "frames"):
            loss.mark_as_loss(
                f"{loss_prefix}ce{postfix}",
                scale=scale * global_loss_scale,
                use_normalized_loss={"none": False, "frames": True}[use_normalized_loss],
            )
        elif use_normalized_loss == "seqs":
            loss.mark_as_loss(f"{loss_prefix}ce{postfix}", scale=0)
            loss_ = rf.pad_packed(loss, dims=batch_dims + [targets_w_eos_spatial_dim], in_dim=pack_dim)
            seq_loss = rf.reduce_sum(loss_, axis=targets_w_eos_spatial_dim)
            seq_loss.mark_as_loss(
                f"{loss_prefix}seq_ce{postfix}", scale=scale * global_loss_scale, use_normalized_loss=True
            )

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"{loss_prefix}fer{postfix}", as_error=True)


def aed_glowtts_single_stream_train_step(*, model: Model, extern_data, **_kwargs_unused):
    """Custom RETURNN train_step: single-stream AED+CTC (real audio + TTS-synth mixed in one batch).

    Unlike aed_glowtts_train_step (alternate_batching -- each batch pure audio OR pure text, separate "txt_" losses),
    here real-audio and text seqs arrive mixed in one batch, like the offline-TTS reference.
    We split the batch by a per-seq audio/text mask,
    run the frozen GlowTTS only on the text sub-batch (peak-normalized synthetic waveforms),
    scatter those back into their batch positions next to the real waveforms
    (rf.nested handles the ragged batch-dependent time dims),
    then a single model.encode + one loss set (no "txt_" prefix, no global_loss_scale).
    With tts_waveform=True the text rows carry synth WAVEFORMS (merged next to real audio, one model.encode).
    With tts_waveform=False (direct-mel) the text rows carry TTS log-mel directly: the audio rows are
    featurized to log-mel and the two are merged in FEATURE space, then model.encode_from_features.
    With pseudo_speech_enc (front-end, start_layer < 0) the text rows come from the trainable
    pseudo encoder instead of the frozen TTS, merged the same way.
    The merged batch gets one uniform SpecAugment width, so audio and text rows are augmented alike.
    """
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict
    import torch

    config = get_global_config()  # noqa
    direct_mel = not config.bool("tts_waveform", False)  # direct DbMel injection (no GL/waveform) vs waveform TTS
    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[config.typed_value("target")]
    targets_spatial_dim = targets.get_time_dim_tag()
    phonemes = extern_data[PHONEMES_DATA_KEY]
    phonemes_spatial_dim = phonemes.get_time_dim_tag()

    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert use_normalized_loss in ("none", "frames"), f"single_stream: unsupported {use_normalized_loss!r}"
    normed = {"none": False, "frames": True}[use_normalized_loss]
    label_smoothing = config.float("label_smoothing", 0.1)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)

    (batch_dim,) = data.remaining_dims(data_spatial_dim)
    # per-seq split: text seqs carry empty audio (size 0); real-audio seqs carry empty phonemes.
    text_mask_cpu = data_spatial_dim.get_size_tensor() <= 0
    text_mask = rf.copy_to_device(text_mask_cpu, data.device)

    num_text = int(text_mask_cpu.raw_tensor.sum())
    num_total = int(batch_dim.get_dim_value())
    # split off the text sub-batch and run the frozen (peak-normalizing) TTS only there.
    # Guard the pure batches (possible with the gumbel-random interleave):
    # masked_select/TTS on an empty sub-batch crashes.
    if num_text == 0:
        phon_t = phon_t_spatial_dim = text_bdim = sel_dim_map = None
        wave_t = wave_t_spatial_dim = None
    elif num_text == num_total:
        phon_t, phon_t_spatial_dim = phonemes, phonemes_spatial_dim
        text_bdim = sel_dim_map = None
    else:
        (phon_t, phon_t_spatial_dim), text_bdim, sel_dim_map = rf.nested.masked_select_nested(
            (phonemes, phonemes_spatial_dim), mask=text_mask, mask_cpu=text_mask_cpu, dims=[batch_dim]
        )
    if phon_t is not None:
        if model.pseudo_enc is not None:
            # Front-end pseudo-enc features are already in model.in_dim,
            # so they merge with the featurized audio rows like direct-mel TTS below.
            # No stop_gradient here: unlike the frozen TTS, the pseudo encoder trains.
            wave_t, wave_t_spatial_dim = model.pseudo_enc(phon_t, spatial_dim=phon_t_spatial_dim)
        else:
            # no_grad: the TTS is frozen;
            # without it, all TTS activations are kept for a backward that never happens (GPU OOM).
            with torch.no_grad():
                wave_t, wave_t_spatial_dim = model.tts(phon_t, spatial_dim=phon_t_spatial_dim)
            wave_t = rf.stop_gradient(wave_t)  # TTS is frozen
    if config.bool("debug_single_stream_stats", False) and not direct_mel and wave_t is not None:
        _lens = wave_t_spatial_dim.dyn_size_ext.raw_tensor
        _n_text = int(_lens.numel())
        _n_tot = int(batch_dim.get_dim_value())
        _phon_lens = phon_t_spatial_dim.dyn_size_ext.raw_tensor
        _tgt_lens = targets_spatial_dim.dyn_size_ext.raw_tensor
        print(
            f"single_stream: {_n_text}/{_n_tot} text seqs,"
            f" synth {float(_lens.sum()) / 16000.0:.1f} s"
            f" (mean {float(_lens.float().mean()) / 16000.0 if _n_text else 0.0:.2f} s/seq),"
            f" phon/seq mean {float(_phon_lens.float().mean()) if _n_text else 0.0:.1f}"
            f" max {int(_phon_lens.max()) if _n_text else 0},"
            f" tgt/seq(all) mean {float(_tgt_lens.float().mean()):.1f}"
        )

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    collected_outputs = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in aux_loss_layers])
    if not direct_mel:
        # waveform path: scatter synth WAVEFORMS (wave_t) into the text positions, real waveforms are the
        # backup, then one model.encode (uniform feature-extraction + feature-BN + SpecAugment over the batch).
        if num_text == 0:
            wave, wave_spatial_dim = data, data_spatial_dim
        elif num_text == num_total:
            wave, wave_spatial_dim = wave_t, wave_t_spatial_dim
        else:
            wave, wave_spatial_dim = rf.nested.masked_scatter_nested(
                (wave_t, wave_t_spatial_dim),
                (data, data_spatial_dim),
                mask=text_mask,
                mask_cpu=text_mask_cpu,
                dims=[batch_dim],
                in_dim=text_bdim,
                masked_select_dim_map=sel_dim_map,
            )
        enc, enc_spatial_dim = model.encode(wave, in_spatial_dim=wave_spatial_dim, collected_outputs=collected_outputs)
    else:
        # direct-mel: wave_t holds the TTS log-mel, so featurize the audio rows and merge in feature space.
        # Audio rows skip pad_audio here, symmetric with the text rows, which never get it.
        if num_text == num_total:
            feats, feats_spatial_dim = wave_t, wave_t_spatial_dim
        else:
            feat_audio, feat_audio_spatial_dim = model.feature_extraction(data, in_spatial_dim=data_spatial_dim)
            if num_text == 0:
                feats, feats_spatial_dim = feat_audio, feat_audio_spatial_dim
            else:
                # The front-end yields float32, but the embedding runs under autocast and yields bfloat16,
                # and masked_scatter_ rejects mixed dtypes.
                if wave_t.dtype != feat_audio.dtype:
                    wave_t = rf.cast(wave_t, feat_audio.dtype)
                feats, feats_spatial_dim = rf.nested.masked_scatter_nested(
                    (wave_t, wave_t_spatial_dim),
                    (feat_audio, feat_audio_spatial_dim),
                    mask=text_mask,
                    mask_cpu=text_mask_cpu,
                    dims=[batch_dim],
                    in_dim=text_bdim,
                    masked_select_dim_map=sel_dim_map,
                )
        enc_raw, enc_spatial_dim = model.encode_from_features(
            feats, in_spatial_dim=feats_spatial_dim, collected_outputs=collected_outputs
        )
        enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)

    # aux CTC losses (single stream)
    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
            continue
        aux_logits = getattr(model, f"enc_aux_logits_{layer_idx}")(collected_outputs[str(layer_idx - 1)])
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        aux_loss = rf.ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=ctc_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=ctc_targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_{layer_idx}",
            scale=aux_loss_scales[i],
            custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=normed,
        )

    # AED CE + FER (single stream)
    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=model.eos_idx,
        out_dims=[targets_w_eos_spatial_dim],
    )
    dec_collected = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in dec_aux_loss_layers])
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=dec_collected,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        dec_aux_logits[layer_idx] = linear(norm(dec_collected[str(layer_idx - 1)]))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )
        if not model.out_eos_separated:
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:
            log_prob = _aed.log_probs_with_eos_separated(
                logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx
            )
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss(f"ce{postfix}", scale=scale, use_normalized_loss=normed)
        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"fer{postfix}", as_error=True)


def aed_pseudo_enc_frontend_single_stream_train_step(*, model: Model, extern_data, **_kwargs_unused):
    """Single-stream AED+CTC with the FRONT-END pseudo-speech encoder, without splitting the batch.

    Audio rows carry no phonemes and text rows no audio, so both branches can run on the full
    batch with the other rows empty, and a per-row concat is already the merge.
    That keeps it free of masks, batch-dim masked_select/scatter and host reads,
    so it stays statically traceable (needed for compile/capture) and packing-friendly.
    One encode over the joined batch follows, as in a normal step.
    """
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict

    config = get_global_config()  # noqa
    assert not config.bool("tts_waveform", False), "front-end pseudo-enc single-stream: no waveform path"
    assert config.bool("pseudo_speech_enc", False), "this step is for the pseudo-speech encoder"
    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[config.typed_value("target")]
    targets_spatial_dim = targets.get_time_dim_tag()
    phonemes = extern_data[PHONEMES_DATA_KEY]
    phonemes_spatial_dim = phonemes.get_time_dim_tag()

    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert use_normalized_loss in ("none", "frames"), f"single_stream: unsupported {use_normalized_loss!r}"
    normed = {"none": False, "frames": True}[use_normalized_loss]
    label_smoothing = config.float("label_smoothing", 0.1)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)

    # Both branches over the full batch; each row is empty in exactly one of them,
    # so nothing is wasted and no split is needed.
    # The pseudo encoder is trainable, so no stop_gradient here.
    text_feats, text_spatial_dim = model.pseudo_enc(phonemes, spatial_dim=phonemes_spatial_dim)
    audio_feats, audio_spatial_dim = model.feature_extraction(data, in_spatial_dim=data_spatial_dim)
    if text_feats.dtype != audio_feats.dtype:
        # the embedding runs under autocast while the front-end does not
        text_feats = rf.cast(text_feats, audio_feats.dtype)
    feats, feats_spatial_dim = rf.concat(
        (audio_feats, audio_spatial_dim), (text_feats, text_spatial_dim), handle_dynamic_dims=True
    )
    feats.feature_dim = model.in_dim
    # Each row is audio-only or text-only (see the docstring), so no row is ever as long
    # as both together. The concat dim is a Dim SUM, whose capacity is audio + text; under
    # capture that is what EVERY row costs, and it drives the CTC frame loop. The true
    # per-row max is the larger of the two. An overrun would trip the packed bound check.
    _a_cap = _dim_capacity(audio_spatial_dim)
    _t_cap = _dim_capacity(text_spatial_dim)
    if _a_cap and _t_cap:
        feats_spatial_dim.capacity = max(_a_cap, _t_cap)

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    collected_outputs = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in aux_loss_layers])
    enc_raw, enc_spatial_dim = model.encode_from_features(
        feats, in_spatial_dim=feats_spatial_dim, collected_outputs=collected_outputs
    )
    enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)

    # aux CTC losses (single stream)
    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
            continue
        aux_logits = getattr(model, f"enc_aux_logits_{layer_idx}")(collected_outputs[str(layer_idx - 1)])
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        aux_loss = rf.ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=ctc_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=ctc_targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_{layer_idx}",
            scale=aux_loss_scales[i],
            # explicit device: get_size_tensor defaults to CPU, a sync under tracing
            custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(device=aux_ctc_log_probs.device),
            use_normalized_loss=normed,
        )

    # AED CE + FER (single stream)
    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=model.eos_idx,
        out_dims=[targets_w_eos_spatial_dim],
    )
    dec_collected = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in dec_aux_loss_layers])
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=dec_collected,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        dec_aux_logits[layer_idx] = linear(norm(dec_collected[str(layer_idx - 1)]))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )
        if not model.out_eos_separated:
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:
            log_prob = _aed.log_probs_with_eos_separated(
                logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx
            )
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss(f"ce{postfix}", scale=scale, use_normalized_loss=normed)
        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"fer{postfix}", as_error=True)


def _dim_capacity(dim: Dim) -> Optional[int]:
    """:return: the dim's static upper bound, declared or derived, None if unknown"""
    if dim.dimension is not None:
        return dim.dimension
    # noinspection PyProtectedMember
    return dim.capacity or dim._derived_capacity()


def aed_pseudo_enc_single_stream_train_step(*, model: Model, extern_data, **_kwargs_unused):
    """Custom RETURNN train_step: single-stream AED+CTC with the pseudo-speech-encoder
    (real audio + pseudo-enc text seqs mixed in one batch), VERSION 2.

    v1 merged at the layer-start_layer INPUT
    and ran the upper Conformer blocks once on the merged batch;
    it broke audio-conditioned learning (dev stuck at decoder-LM level;
    the exact defect was never isolated --
    the scatter itself tests OK for row alignment and gradient flow).
    v2 composes the step from the PROVEN dual-stream building blocks:
    audio rows run the STANDARD full encoder path (as dual-stream audio steps),
    text rows run pseudo_enc + the STANDARD layer-split path (as dual-stream text steps),
    and the per-layer aux taps + final outputs are scatter-merged into the full batch
    (per-row processing in the upper blocks is mathematically identical either way;
    batching them together is purely computational).
    ONE decoder pass + ONE loss set.
    Aux CTC tapped below start_layer (ctc_4 at start_layer=4): audio rows only
    (same coverage as dual-stream, where the text branch skips those aux losses).
    """
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict

    config = get_global_config()  # noqa
    assert config.bool("pseudo_speech_enc", False)
    assert config.int("pseudo_enc_single_stream_version", 1) == 3, (
        "v1/v2 are broken, see notes"
        " (v1: merge at the layer input;"
        " v2: ctc_4 targets selected separately -> own batch dim -> audio x targets cross-product CTC);"
        " only version=3 is supported"
    )
    start_layer = config.int("pseudo_enc_start_layer", -1)
    assert start_layer >= 0, "pseudo-enc single-stream: layer-split injection only"
    assert config.typed_value("pseudo_enc_units", "phonemes") == "phonemes", "pseudo-enc single-stream: phonemes only"
    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[config.typed_value("target")]
    targets_spatial_dim = targets.get_time_dim_tag()
    phonemes = extern_data[PHONEMES_DATA_KEY]
    phonemes_spatial_dim = phonemes.get_time_dim_tag()

    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert use_normalized_loss in ("none", "frames"), f"single_stream: unsupported {use_normalized_loss!r}"
    normed = {"none": False, "frames": True}[use_normalized_loss]
    label_smoothing = config.float("label_smoothing", 0.1)
    pseudo_specaug_width = config.typed_value("pseudo_enc_specaug_max_width", None)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    (batch_dim,) = data.remaining_dims(data_spatial_dim)
    # per-seq split: text seqs carry empty audio (size 0); real-audio seqs carry empty phonemes.
    text_mask_cpu = data_spatial_dim.get_size_tensor() <= 0
    audio_mask_cpu = rf.logical_not(text_mask_cpu)
    audio_mask = rf.copy_to_device(audio_mask_cpu, data.device)
    text_mask = rf.copy_to_device(text_mask_cpu, data.device)
    num_text = int(text_mask_cpu.raw_tensor.sum())
    num_total = int(batch_dim.get_dim_value())

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    n_layers = len(model.encoder.layers)
    aux_src = {}  # layer_idx -> (out, out_spatial_dim, targets, targets_spatial_dim)
    if num_text == 0:
        # pure real-audio batch: the standard full encode.
        collected = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in aux_loss_layers])
        enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected)
        for i in aux_loss_layers:
            if i <= n_layers:
                aux_src[i] = (collected[str(i - 1)], enc_spatial_dim, ctc_targets, ctc_targets_spatial_dim)
    elif num_text == num_total:
        # pure text batch: the plain layer-split path (aux below start_layer skipped, as in dual-stream).
        collected = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in aux_loss_layers])
        feats, feats_spatial_dim = model.pseudo_enc(phonemes, spatial_dim=phonemes_spatial_dim)
        enc_raw, enc_spatial_dim = model.encode_from_enc_space(
            feats,
            spatial_dim=feats_spatial_dim,
            start_layer=start_layer,
            collected_outputs=collected,
            specaugment_max_spatial_dims=pseudo_specaug_width,
        )
        enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)
        for i in aux_loss_layers:
            if i <= n_layers and i - 1 >= start_layer:
                aux_src[i] = (collected[str(i - 1)], enc_spatial_dim, ctc_targets, ctc_targets_spatial_dim)
    else:
        # mixed batch: each branch runs its PROVEN dual-stream path on its selected rows.
        # ONE joint select for the audio and its ctc targets:
        # separate selects create separate selected batch dims,
        # and rf.ctc_loss then broadcasts them into an audio x targets cross-product
        # (the v2 bug -- ctc_4 was ~20 * num_audio_rows, gradient noise into layers 1..start_layer).
        ((wave_a, wave_a_spatial_dim), (audio_ct, audio_ct_sp)), audio_bdim, sel_map_a = rf.nested.masked_select_nested(
            ((data, data_spatial_dim), (ctc_targets, ctc_targets_spatial_dim)),
            mask=audio_mask,
            mask_cpu=audio_mask_cpu,
            dims=[batch_dim],
        )
        coll_a = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in aux_loss_layers])
        enc_a_raw, enc_a_spatial_dim = model.encode_no_transform(
            wave_a, in_spatial_dim=wave_a_spatial_dim, collected_outputs=coll_a
        )
        (phon_t, phon_t_spatial_dim), text_bdim, sel_map_t = rf.nested.masked_select_nested(
            (phonemes, phonemes_spatial_dim), mask=text_mask, mask_cpu=text_mask_cpu, dims=[batch_dim]
        )
        feats_t, feats_t_spatial_dim = model.pseudo_enc(phon_t, spatial_dim=phon_t_spatial_dim)
        coll_t = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in aux_loss_layers])
        enc_t_raw, enc_t_spatial_dim = model.encode_from_enc_space(
            feats_t,
            spatial_dim=feats_t_spatial_dim,
            start_layer=start_layer,
            collected_outputs=coll_t,
            specaugment_max_spatial_dims=pseudo_specaug_width,
        )

        def _merge(t_a, dim_a, t_t, dim_t):
            """Merge the two selected sub-batches into the full batch:
            scatter the audio rows into a zero-width full-batch stub,
            then the text rows into the result."""
            # dim sizes are CPU by convention (in-step rf default device is cuda)
            stub_dim = Dim(rf.zeros([batch_dim], dtype="int32", device="cpu"), name="merge_stub")
            stub = rf.zeros([batch_dim, stub_dim, model.encoder.out_dim], dtype=t_a.dtype, device=t_a.device)
            half, half_dim = rf.nested.masked_scatter_nested(
                (t_a, dim_a),
                (stub, stub_dim),
                mask=audio_mask,
                mask_cpu=audio_mask_cpu,
                dims=[batch_dim],
                in_dim=audio_bdim,
                masked_select_dim_map=sel_map_a,
            )
            if t_t.dtype != t_a.dtype:
                t_t = rf.cast(t_t, t_a.dtype)
            return rf.nested.masked_scatter_nested(
                (t_t, dim_t),
                (half, half_dim),
                mask=text_mask,
                mask_cpu=text_mask_cpu,
                dims=[batch_dim],
                in_dim=text_bdim,
                masked_select_dim_map=sel_map_t,
            )

        # aux tapped below the injection layer: audio rows only.
        aux_below = [i for i in aux_loss_layers if i - 1 < start_layer and i <= n_layers]
        for i in aux_below:
            aux_src[i] = (coll_a[str(i - 1)], enc_a_spatial_dim, audio_ct, audio_ct_sp)
        # shared aux taps: merged over the full batch, against the full-batch targets.
        for i in aux_loss_layers:
            if i - 1 >= start_layer and i <= n_layers:
                m_out, m_dim = _merge(coll_a[str(i - 1)], enc_a_spatial_dim, coll_t[str(i - 1)], enc_t_spatial_dim)
                aux_src[i] = (m_out, m_dim, ctc_targets, ctc_targets_spatial_dim)
        # final encoder output: merged, then the decoder transform.
        enc_raw, enc_spatial_dim = _merge(enc_a_raw, enc_a_spatial_dim, enc_t_raw, enc_t_spatial_dim)
        enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)

    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx not in aux_src:
            continue
        aux_out, aux_sp, aux_tgt, aux_tgt_sp = aux_src[layer_idx]
        aux_logits = getattr(model, f"enc_aux_logits_{layer_idx}")(aux_out)
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        aux_loss = rf.ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=aux_tgt,
            input_spatial_dim=aux_sp,
            targets_spatial_dim=aux_tgt_sp,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_{layer_idx}",
            scale=aux_loss_scales[i],
            custom_inv_norm_factor=aux_tgt_sp.get_size_tensor(),
            use_normalized_loss=normed,
        )

    # AED CE + FER (single stream), identical to aed_glowtts_single_stream_train_step.
    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=model.eos_idx,
        out_dims=[targets_w_eos_spatial_dim],
    )
    dec_collected = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in dec_aux_loss_layers])
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=dec_collected,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        dec_aux_logits[layer_idx] = linear(norm(dec_collected[str(layer_idx - 1)]))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )
        if not model.out_eos_separated:
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:
            log_prob = _aed.log_probs_with_eos_separated(
                logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx
            )
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss(f"ce{postfix}", scale=scale, use_normalized_loss=normed)
        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"fer{postfix}", as_error=True)
        if postfix == "" and 0 < num_text < num_total:
            # diagnostics: the same CE split by stream (audio rows vs text rows), reported only.
            loss_padded = rf.pad_packed(loss, dims=batch_dims + [targets_w_eos_spatial_dim], in_dim=pack_dim)
            for diag_name, diag_mask in [("ce_audio_rows", audio_mask), ("ce_text_rows", text_mask)]:
                masked = rf.where(diag_mask, loss_padded, rf.zeros_like(loss_padded))
                denom = rf.where(
                    diag_mask,
                    rf.cast(targets_w_eos_spatial_dim.get_size_tensor(), "float32"),
                    rf.zeros([batch_dim], dtype="float32", device="cpu"),
                )
                diag = rf.reduce_sum(masked, axis=masked.dims) / rf.maximum(
                    rf.copy_to_device(rf.reduce_sum(denom, axis=denom.dims), masked.device), 1.0
                )
                diag.mark_as_loss(name=diag_name, as_error=True)


_glowtts_text_tok_cache = None


def _glowtts_text_tokenizers():
    """Lazily build + cache (per worker process) the spm vocab + GlowTTS PhoneSeqGenerator from the config."""
    global _glowtts_text_tok_cache
    if _glowtts_text_tok_cache is None:
        from returnn.config import get_global_config
        from returnn.datasets.util.vocabulary import Vocabulary
        from returnn.datasets.lm import PhoneSeqGenerator

        config = get_global_config()
        spm = Vocabulary.create_vocab(**config.typed_value("glow_tts_text_spm_opts"))
        seq_gen = PhoneSeqGenerator(**config.typed_value("glow_tts_phone_info"))
        _glowtts_text_tok_cache = (spm, seq_gen)
    return _glowtts_text_tok_cache


def _glowtts_text_map_seq(seq, *, target_key, spm_dim, phon_dim, rng, **_kwargs):
    """PostprocessingDataset map_seq: raw utf8 bytes -> (spm target, GlowTTS phonemes), both from the same text."""
    import numpy as np
    from returnn.tensor import Tensor, TensorDict, Dim as _Dim

    spm, seq_gen = _glowtts_text_tokenizers()
    orth = bytes(np.asarray(seq["data"].raw_tensor).astype("uint8").tolist()).decode("utf8")
    spm_ids = np.array(spm.get_seq(orth), dtype="int32")
    # per-seq silence/pronunciation-variant randomization, seeded from the epoch-seeded rng.
    seq_gen.random_seed(int(rng.randint(0, 2**31 - 1)))
    phon_ids = seq_gen.seq_to_class_idxs(seq_gen.generate_seq(orth), dtype="int32")

    out = TensorDict()
    out.data[target_key] = Tensor(
        target_key, dims=[_Dim(None, name="spm_seq")], dtype="int32", sparse_dim=spm_dim, raw_tensor=spm_ids
    )
    out.data[PHONEMES_DATA_KEY] = Tensor(
        PHONEMES_DATA_KEY, dims=[_Dim(None, name="phon_seq")], dtype="int32", sparse_dim=phon_dim, raw_tensor=phon_ids
    )
    return out


def _add_empty_phonemes_map_seq(seq, *, phonemes_sparse_dim, **_kwargs):
    """PostprocessingDataset map_seq: pass everything through, add an empty ``phonemes`` stream.

    Used only for the audio-only CV/eval datasets so they carry the (unused-at-CV) phonemes key.
    """
    import numpy as np
    from returnn.tensor import Tensor, TensorDict, Dim as _Dim

    out = TensorDict()
    for k, v in seq.data.items():
        out.data[k] = v
    # Dynamic per-seq dim (length 0 here); PostprocessingDataset requires it dynamic to match map_outputs.
    spatial = _Dim(None, name="phon_seq")
    out.data[PHONEMES_DATA_KEY] = Tensor(
        PHONEMES_DATA_KEY,
        dims=[spatial],
        dtype="int32",
        sparse_dim=phonemes_sparse_dim,
        raw_tensor=np.zeros([0], "int32"),
    )
    return out


def _extern_template_to_map_output(tmpl: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an extern_data template ({dim_tags:[batch,...], [sparse_dim]}) into a PostprocessingDataset
    ``map_outputs`` entry ({dims:[non-batch dims], dtype, [sparse_dim]})."""
    from returnn.tensor import batch_dim

    dims = [d for d in tmpl["dim_tags"] if d != batch_dim]
    out: Dict[str, Any] = {"dims": dims, "dtype": "int32" if "sparse_dim" in tmpl else "float32"}
    if "sparse_dim" in tmpl:
        out["sparse_dim"] = tmpl["sparse_dim"]
    return out


def _wrap_eval_with_empty_phonemes(
    eval_ds: Dict[str, Any], *, base_extern: Dict[str, Any], phon_extern: Dict[str, Any]
) -> Dict[str, Any]:
    """Wrap an audio-only eval dataset so it also emits an empty ``phonemes`` stream (extern_data contract)."""
    map_outputs = {
        k: _extern_template_to_map_output(v) for k, v in {**base_extern, PHONEMES_DATA_KEY: phon_extern}.items()
    }
    return {
        "class": "PostprocessingDataset",
        # Explicit "default": PostprocessingDataset rejects a non-default seq_ordering on itself, and RETURNN
        # would otherwise inject one. The inner eval dataset keeps its own "sorted_reverse".
        "seq_ordering": "default",
        "dataset": eval_ds,
        "map_seq": functools.partial(_add_empty_phonemes_map_seq, phonemes_sparse_dim=phon_extern["sparse_dim"]),
        "map_outputs": map_outputs,
    }
