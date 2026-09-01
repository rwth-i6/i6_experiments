"""
Loquacious AED (~500M: 16L Conformer 1024d + 6L Transformer decoder 1024d, spm10k),
trained with RETURNN's packed-tensor path.

Reproduces ``base-v2-large-nFullEp4.0-nEp100-totalHours100k`` from
:mod:`i6_experiments.users.zeyer.experiments.exp2025_10_04_loquacious`
(subset ``large``, 100k hours = 4.0 full epochs = 100 subepochs at train_epoch_split 25),
using the packed-tensor configuration developed in
:mod:`i6_experiments.users.zeyer.experiments.exp2026_05_23_returnn`.

Two experiments are registered:

- ``base-v2``        -- the PADDED reference, behaviour version 29. Not optional: a packed
                        training curve is uninterpretable without the same-behaviour padded
                        control to compare it against.
- ``base-v2-packed`` -- the packed run, at Albert's measured throughput optimum
                        (28M packed content per step, random ordering, unpartitioned).

Measured on this exact model (561M) on a 96GB H100, from the benchmark jobs of the
2026-05-23-returnn-paper setup:

    padded_eager, laplace, bs100k (its own tuned optimum)   206.1 seqs/s   64.4 GB
    packed_graphc, same batching                            336.6 seqs/s   46.2 GB
    packed_graphc, packed_batch_size 16.19M                 377.0 seqs/s   45.6 GB
    packed_graphc, packed_batch_size 28M (this config)      408.5 seqs/s   71.6 GB

i.e. ~2.0x throughput end-to-end: 1.63x from the packed implementation at an identical batch,
x1.21 from spending the freed memory on a bigger batch.

WHAT "PACKED" IS HERE: purely an engine/config concern. The model, train step and recog defs in
``..model`` are unmodified AED code -- exactly as in ``exp2026_05_23_returnn.loq_train``, which
runs the stock ``aed_model_def`` / ``aed_training``. Three independent knobs are involved:

- ``packed_tensors`` / ``packed_batch_size`` -- ragged storage + content-based batching.
  On its own this does NOT speed up the step (under laplace there is only ~9.5% padding to
  recover, and the ragged layout costs more than that); what it buys is memory, and the ability
  to compile at all.
- ``torch_cuda_graph: {compile: True}`` -- Inductor-codegen of the whole step. THIS is the speed
  win (0.648 -> 0.347 s/step). It is only possible on the packed path: the padded CTC loss is
  ``aten._ctc_loss``, untraceable under fake tensors, which is why the packed native op exists.
- ``torch_cuda_graph: {capture: True}`` -- CUDA-graph capture on top. Measured within noise at
  this model size (0.341 vs 0.347); it matters when steps are short, not at 0.35 s/step.

DIFFERENCES FROM THE REFERENCE, deliberate (see also the ``__init__`` of this package):

1. SPM sampling is actually applied. ``aed_train_exp`` silently ignores ``train_vocab_opts``
   whenever a ``task`` is passed, which both the reference and ``loq_train`` do -- so the
   reference ran WITHOUT the SamplingBytePairEncoding it configures. Here it is passed to
   ``get_loquacious_task_raw_v2`` directly, so it takes effect. See ``_TEXT_DIM_CAP`` below for
   the one consequence that needs watching.
2. The dataset actually gets its worker processes. The reference sets
   ``__multi_proc_dataset_opts`` (the train_v3 key) while train_v4 reads ``__multi_proc_dataset``,
   so it never applied and every run sat at the task default of 2 workers -- which is why
   Albert's scale-ladder trainings hit a loader wall. Passed via the task's ``multi_proc`` arg,
   the mechanism the Loquacious dataset actually uses.

NOT included here (out of scope for reproducing this one training): the Transformer LM and the
``ctc_recog_recomb_labelwise_prior_auto_scale`` / 4-gram LM decoding of the reference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from sisyphus import tk

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_task_raw_v2
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs as _baseline_configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as _aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

from ..model import model_def, train_def
from ..model.recognition.aed_beam_search import recog_def as aed_recog_def
from ..model.recognition.ctc_beam_search import recog_def as ctc_recog_def

__setup_root_prefix__ = "exp2026_08_27_aed_llm_vocab"


# --- packed layout / capacity constants -------------------------------------------------------

# Encoder frontend downsampling: 6 frames of 10ms at 16kHz = 960 raw samples per encoder frame.
_ALIGN = 960

# Audio dim capacity, in raw samples: max_seq_length_default_input is 19.5s = 312_000 samples and
# there is NO speed perturbation in this config, so 312_960 (the next multiple of 960) is provably
# sufficient -- no sequence can exceed it.
_AUDIO_DIM_CAP = 312_960

# Text dim capacity, in labels. This is a HARD per-sequence cap: graph capture raises loudly in
# _copy_in if a sequence exceeds it (it does not silently truncate), so a too-small value fails
# fast rather than corrupting a training.
#
# Albert measured, on the unsampled spm10k train shards and CONDITIONED on the 19.5s audio filter
# (8.96M of 9.49M seqs pass): max 246, p99.99 86, p99 66 -- and used 256.
# We enable SamplingBytePairEncoding (see the module docstring), which lengthens targets, so 256
# would be cutting it fine. 384 = 1.56x the unsampled conditional max.
# :func:`text_seq_len_stats` below registers the measurement to replace this estimate with a fact.
#
# NOTE: max_seq_length_default_target stays None (as in the reference), so RETURNN does NOT filter
# over-long targets -- a sequence above this cap would abort the training rather than be skipped.
# If that ever happens, raise this value (and the packed_total_bound text budget with it).
_TEXT_DIM_CAP = 384

# Throughput optimum from Albert's (batch size x activation memory budget) sweep under random
# ordering: 28M packed audio content, unpartitioned, measured 409 seqs/s at 71.6 GB of 79.2.
# 30M buys nothing (+4.7 GB, -2 seqs/s) and the frontier sits between 28M and 30M.
_AUDIO_BUDGET = 28_000_000

# Text budget and seq bound scale with the audio budget, at Albert's derivation
# scale = 28M / 16_192_320 = 1.7292: round(4_000 * scale) = 6_917, round(200 * scale) = 346.
# The 4_000 base came from a laplace batch simulation at the 16.19M budget (text sums:
# mean 2_910 / p99.9 3_303 / max 3_429), i.e. max + 17%.
# packed_batch_size is enforced BY THE BATCHER, so this can never overflow the buffer -- it can
# only close batches early, which would show up as audio-bound slack in the smoke run's
# log_batch_size output. SPM sampling eats into the 17% headroom; watch that number.
_TEXT_BUDGET = 6_917

# max_seqs MUST track batch_size_bound. Albert's first attempt left max_seqs at 200 while the
# bound was 346, so the batcher closed every batch at 200 seqs and the 28M budget was never
# reached (measured 24.3-27.2M content per step, i.e. 3-13% pure waste: the buffer was still
# sized and computed over for 28M).
_BATCH_SIZE_BOUND = 346


def _get_task(**kwargs) -> Task:
    """
    :func:`get_loquacious_task_raw_v2`, bypassing its ``@cache``.

    The cache is a plain ``functools.cache``, so any dict-valued argument raises
    ``TypeError: unhashable type: 'dict'`` -- which makes ``train_vocab_opts`` (a dict)
    impossible to pass through it. That is the second reason the SPM sampling never applied
    anywhere: even if ``aed_train_exp`` had forwarded the option, this call would have crashed.
    Indeed no caller in the tree passes it.

    Bypassing the cache only costs a rebuild of the (cheap) Task wrapper; the Jobs it creates are
    hash-deduplicated by Sisyphus as usual, so repeated calls converge on the same graph nodes.
    """
    return get_loquacious_task_raw_v2.__wrapped__(**kwargs)


_base_config: Dict[str, Any] = {
    # base-v2-large-nFullEp4.0-nEp100-totalHours100k:
    # large subset, 100k hours -> train_epoch_split 25, 4.0 full epochs, 100 subepochs.
    "subset": "large",
    "total_k_hours": 100,
    "vocab": "spm10k",
    "model": {
        # 29 (up from the reference's 24) is REQUIRED for packed training: the conv-block
        # BatchNorm masks its statistics, which otherwise run over the raw packed storage and
        # count the packing gap frames. It also brings 25 (scatter masking), 26 (DistributeFiles
        # sharding, a no-op unsharded), 27 (module output keeps input dtype under autocast) and
        # 28 (per-seq specaugment masks).
        # The padded reference runs at 29 too, else a packed-vs-padded comparison would also
        # carry 24 -> 29 and mean nothing.
        "behavior_version": 29,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
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
    },
    "train_update_func_from_n_ep": lambda n_ep: {
        "train": _baseline_configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5)
    },
    "train": dict_update_deep(
        _baseline_configs.config_96gb_bf16_accgrad1,
        {
            # 100k frames at 10ms level ~ 1000 sec of audio per batch; max_seqs 200 by default.
            "batch_size": 100_000 * _baseline_configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            # No speed perturbation, as in the reference (this is what makes _AUDIO_DIM_CAP provable).
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            # No target-len filter, as in the reference. See the _TEXT_DIM_CAP note.
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    ),
    # NB: no "__multi_proc_dataset_opts" here -- that key is train_v3's and train_v4 ignores it
    # (see the module docstring). Worker count goes through "multi_proc" below.
    "train_post": dict_update_deep(_baseline_configs.post_config, {"log_grad_norm": True}),
    "train_vocab_opts": {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    "multi_proc": 25,
    "env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}


def train(
    name: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    *,
    config_deletes: Optional[Sequence[str]] = None,
    recog_def_ctc_only: bool = False,
    final_aed_ctc_recog: bool = True,
    prefix: Optional[str] = None,
) -> Tuple[ModelWithCheckpoints, Task, int]:
    """
    Register one Loquacious AED training + its recogs.

    :param name: experiment name, used in the alias/output prefix
    :param config_overrides: deep-merged onto :data:`_base_config` (dotted keys, e.g. "train.batch_size")
    :param config_deletes: applied BEFORE the overrides, so a variant can drop an option that does
        not apply to it instead of having to set it to a no-op value
    :param recog_def_ctc_only: use the cheap CTC time-sync search for the per-epoch recog instead
        of the AED beam search (the reference uses AED, which is the default here)
    :param final_aed_ctc_recog: also run the joint AED+CTC time-sync recog with auto-tuned scales
        on the final fixed epoch
    :param prefix: alias/output prefix; defaults to this module's setup prefix
    :return: (exp, task, aux_ctc_layer)
    """
    if prefix is None:
        prefix = get_setup_prefix_for_module(__name__)

    config = dict_update_deep(_base_config.copy(), config_overrides, config_deletes, dict_value_merge=False)

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    subset = config.pop("subset")
    total_k_hours = config.pop("total_k_hours")
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)  # 100 subepochs

    train_update_func_from_n_ep = config.pop("train_update_func_from_n_ep")
    if train_update_func_from_n_ep:
        config = dict_update_deep(config, train_update_func_from_n_ep(n_ep))

    model_config: Dict[str, Any] = config.pop("model")
    train_config: Dict[str, Any] = config.pop("train")
    post_config: Dict[str, Any] = config.pop("train_post")
    vocab: str = config.pop("vocab")
    train_vocab_opts = config.pop("train_vocab_opts")
    multi_proc = config.pop("multi_proc")
    env_updates = config.pop("env_updates")
    # Only passed on when set, so the default call stays as-is.
    train_seq_ordering = config.pop("train_seq_ordering", None)

    assert not config, f"unhandled config keys: {sorted(config)}"

    if vocab == "qwen-restricted":
        # The restricted Qwen2 LLM vocab (see sis_recipe.llm_vocab): the point of this setup.
        # Its own task builder, because get_loquacious_task_raw_v2 resolves `vocab` via
        # get_vocab_by_str (SPM only) and hardcodes the SPM recog post-processing.
        from .llm_vocab import get_loquacious_task_qwen_restricted

        # train_vocab_opts here means RETURNN vocab opts for the TRAIN split only, e.g.
        # {"bpe_dropout": 0.1} -- the byte-level-BPE analogue of the spm10k run's
        # SamplingBytePairEncoding. Verified to work with the pruned tokenizer; see
        # RestrictedQwenVocab.copy for the measured caveats.
        task = get_loquacious_task_qwen_restricted(
            subset_name=subset,
            train_epoch_split=train_epoch_split,
            train_vocab_opts=train_vocab_opts,
            multi_proc=multi_proc,
            **({"train_seq_ordering": train_seq_ordering} if train_seq_ordering is not None else {}),
        )
    else:
        task = _get_task(
            vocab=vocab,
            subset_name=subset,
            train_epoch_split=train_epoch_split,
            # Unlike aed_train_exp's ignored kwarg, this one actually reaches the dataset.
            train_vocab_opts=train_vocab_opts,
            multi_proc=multi_proc,
            **({"train_seq_ordering": train_seq_ordering} if train_seq_ordering is not None else {}),
        )

    aux_ctc_layer = max(
        i for i in train_config["aux_loss_layers"] if i <= model_config["enc_build_dict"]["num_layers"]
    )

    exp = _aed_train_exp(
        name,
        train_config,
        prefix=prefix + "/aed/",
        task=task,
        model_def=model_def,
        model_config=model_config,
        train_def=train_def,
        recog_def=ctc_recog_def if recog_def_ctc_only else aed_recog_def,
        # The CTC search reads the top aux head, which the recog-side model only builds
        # if aux_loss_layers is in the (recog) config.
        search_config={"aux_loss_layers": [aux_ctc_layer]} if recog_def_ctc_only else None,
        post_config_updates=post_config,
        vocab=vocab,
        env_updates=env_updates,
    )

    if final_aed_ctc_recog:
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=f"{prefix}/aed/{name}/aed+ctc",
            task=task,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=aux_ctc_layer,
        )

    return exp, task, aux_ctc_layer


# The packed configuration. Kept as a module-level dict so that the step-4 smoke/ablation cells
# (TrainStepBenchmarkJob at packed_graphc / packed_compiled / padded_eager) can be added later
# against exactly the regime the training runs in, rather than a hand-copied approximation of it.
_packed_overrides: Dict[str, Any] = {
    # Adam must be capturable for the optimizer step to live inside the CUDA graph.
    "train.optimizer.capturable": True,
    # Bare `True`: gap 0, align 1, no manual layout tuning. The frontend stride then does not
    # divide the align, so the conv auto-realigns (align 1 -> 960) on every step -- and Albert
    # measured that as FASTER than the hand-tuned per_key layout (0.484 vs 0.499-0.545 s/step).
    "train.packed_tensors": True,
    # Batch on PACKED content sums. batch_size MUST be None: otherwise the padded accounting
    # always binds first and the packed budget is a no-op.
    "train.batch_size": None,
    "train.packed_batch_size": {"audio": _AUDIO_BUDGET, "text": _TEXT_BUDGET},
    "train.max_seqs": _BATCH_SIZE_BOUND,
    "train.torch_cuda_graph": {
        "batch_size_bound": _BATCH_SIZE_BOUND,
        # Loquacious extern-data keys are "audio"/"text", NOT "data"/"classes" like the LS baselines.
        "dim_capacity": {"audio": _AUDIO_DIM_CAP, "text": _TEXT_DIM_CAP},
        # Budget == buffer bound by construction, for both keys.
        "packed_total_bound": {"audio": _AUDIO_BUDGET, "text": _TEXT_BUDGET},
        # 0, not the default 2. The EAGER warmup step is the memory peak (61.6 GB at the 16.19M
        # budget), so it -- not the captured graph -- sets the batch-size ceiling; with 2 warmup
        # steps every budget above 16.19M died with CUDA OOM at 79.1 of 79.2 GiB, inside the
        # warmup, before the compile ever ran. warmup_steps 0 removes it entirely (explicit
        # optimizer-state init + host constants outside the trace) and is what makes this budget
        # reachable at all.
        "warmup_steps": 0,
        "capture_optimizer": True,
        "compile": True,
    },
}


def py():
    """Sisyphus entry point."""
    # The padded control. Same behaviour version, same data, same LR schedule; only the batching
    # regime differs from base-v2-packed.
    # train("base-v2")

    # The packed run.
    train(
        "base-v2-packed",
        config_overrides={
            **_packed_overrides,
            # A packed_batch_size bounds CONTENT, so it is ordering-independent -- which is the
            # configuration it is actually for. A normal batch_size bounds the padded rectangle
            # and only means anything under laplace.
            "train_seq_ordering": "random",
        },
    )

    # THE experiment this setup exists for: the same AED, from scratch, on the restricted Qwen2
    # LLM vocab (39_922) instead of spm10k -- so the encoder is trained to produce features
    # decodable into the LLM's own token inventory, and can later seed a speech LLM.
    #
    # Only the vocab differs from base-v2-packed. Model size 713M vs 711M at spm10k (the vocab
    # dimension appears 5x in this model, but 39_922 is close enough to spm10k's 10_025 that it
    # costs little; the full 151_646 vocab would have been 1_285M).
    #
    # Target lengths measured on train with this tokenizer: mean 26.97, p99 87, MAX 229 -- so the
    # inherited _TEXT_DIM_CAP of 384 has ample headroom over a hard measured maximum (there is no
    # SPM-style sampling here to inflate it). Tokens per second of audio is 2.84 vs spm10k's
    # ~2.88, so the packed text budget carries over unchanged.
    # train(
    #     "base-v2-packed-qwenVocab",
    #     config_overrides={
    #         **_packed_overrides,
    #         "train_seq_ordering": "random",
    #         "vocab": "qwen-restricted",
    #         "train_vocab_opts": None,
    #     },
    # )

    text_seq_len_stats()


def text_seq_len_stats():
    """
    Target-length distribution of the Loquacious train set (spm10k), to replace the estimate in
    :data:`_TEXT_DIM_CAP` with a measurement.

    Uses the TEXT-ONLY dataset variant, so no audio is decoded (9.5M seqs otherwise).

    Caveat: ``get_loquacious_text_only_dataset_v2`` takes no ``train_vocab_opts``, so this measures
    the UNSAMPLED lengths. It reproduces Albert's numbers on our own vocab build (expected:
    conditional max 246) and thus pins the base that _TEXT_DIM_CAP's sampling margin sits on top
    of; it does not directly measure the sampled lengths.

    Output format "py" (seq_tag -> len), not "txt": the train ordering is laplace, so a plain
    length list cannot be joined with the per-seq audio durations, and the bound that matters is
    CONDITIONED on the 19.5s audio filter (only those seqs ever reach a batch).
    """
    from i6_core.returnn.dataset import ExtractSeqLensJob
    from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_text_only_dataset_v2

    prefix = get_setup_prefix_for_module(__name__)
    ds = get_loquacious_text_only_dataset_v2(vocab="spm10k", train_epoch_split=1)
    job = ExtractSeqLensJob(dataset=ds.train_dataset, key="text", output_format="py")
    job.rqmt = {"gpu": 0, "cpu": 2, "mem": 8, "time": 8}  # 9.5M seqs to tokenize
    tk.register_output(f"{prefix}/stats/loq-train-text-seq-lens.py", job.out_file)
