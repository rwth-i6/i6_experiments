"""
CTC + LM search experiments

Also implementing delayed fusion.
"""

from __future__ import annotations

import functools
from typing import Optional, Any, Dict, Tuple
from functools import cache

from sisyphus import tk
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
    ctc_recog_recomb_labelwise_prior_auto_scale,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
    get_loquacious_train_subset_dataset_v2,
    get_loquacious_text_only_dataset_for_forward,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

__setup_root_prefix__ = "exp2025_11_11_ctc_lm_search"


def py():
    prefix = get_setup_prefix_for_module(__name__)

    am, aux_ctc_layer = get_am("base", {})
    name = "base"
    vocab = "spm10k"
    task = get_loquacious_task_raw_v2(vocab=vocab)

    # ASR baseline.
    # CTC-only:
    # {"dev": 7.32, "dev_voxpopuli": 7.11, "dev_commonvoice": 11.13, "dev_librispeech": 4.8, "dev_yodas": 11.49,
    #  "test": 8.1, "test_voxpopuli": 7.28, "test_commonvoice": 13.29, "test_librispeech": 5.07, "test_yodas": 11.2}
    # AED+CTC 1st pass:
    # {"dev": 6.49, "dev_voxpopuli": 6.62, "dev_commonvoice": 9.29, "dev_librispeech": 4.22, "dev_yodas": 10.96,
    #  "test": 7.31, "test_voxpopuli": 6.79, "test_commonvoice": 11.61, "test_librispeech": 4.48, "test_yodas": 10.62}
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=am,
        aux_ctc_layer=aux_ctc_layer,
    )

    from .exp2024_04_23_baselines.recog_ext.ctc import model_recog_with_recomb

    # {"dev": 6.12, "dev_voxpopuli": 6.6, "dev_commonvoice": 8.35, "dev_librispeech": 3.76, "dev_yodas": 11.81,
    #  "test": 6.94, "test_voxpopuli": 6.63, "test_commonvoice": 10.45, "test_librispeech": 4.02, "test_yodas": 11.82}
    # dev elapsed: elapsed: 1:33:42.9703
    # dev-yodas: elapsed: 0:06:12.8701
    lm_name, lm = get_lm(prefix=prefix, vocab=vocab)
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-v2/{lm_name}",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        recog_def=model_recog_with_recomb,
    )

    from .exp2024_04_23_baselines.recog_ext.ctc_v2 import model_recog_with_recomb_v2

    # {"dev": 6.14, "dev_voxpopuli": 6.6, "dev_commonvoice": 8.38, "dev_librispeech": 3.77, "dev_yodas": 11.91,
    #  "test": 6.94, "test_voxpopuli": 6.62, "test_commonvoice": 10.45, "test_librispeech": 4.02, "test_yodas": 11.89}
    # dev elapsed: elapsed: 1:35:35.7686
    # dev-yodas: elapsed: 0:07:33.1449
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-v3/{lm_name}",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        recog_def=model_recog_with_recomb_v2,
    )

    from .exp2024_04_23_baselines.recog_ext.ctc_delayed_fusion import model_recog_with_recomb_delayed_fusion

    # {"dev": 6.14, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.41, "dev_librispeech": 3.77, "dev_yodas": 11.76,
    #  "test": 6.98, "test_voxpopuli": 6.64, "test_commonvoice": 10.5, "test_librispeech": 4.05, "test_yodas": 12.0}
    # dev: elapsed: 1:41:44.8639
    # dev-yodas: elapsed: 0:06:41.7572
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed/{lm_name}",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=11,
        recog_def=model_recog_with_recomb_delayed_fusion,
    )

    from .exp2024_04_23_baselines.recog_ext.ctc_delayed_fusion_v2 import (
        model_recog_with_recomb_delayed_fusion_v2,
        never_enable,
        enable_by_interval,
        convert_labels_func,
        convert_labels_func_no_op,
        spm_space_first_is_word_start,
        spm_label_merge_v2,
        seq_str_postprocess_lower_case,
    )

    # {"dev": 6.23, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.74, "dev_librispeech": 3.84, "dev_yodas": 11.42,
    #  "test": 7.05, "test_voxpopuli": 6.65, "test_commonvoice": 10.89, "test_librispeech": 4.13, "test_yodas": 11.46}
    enable_every20 = functools.partial(enable_by_interval, interval=20)
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2/{lm_name}",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": enable_every20,
            "should_fuse_now_func": enable_every20,
            "convert_labels_func": convert_labels_func_no_op,
        },
    )

    for interval in [1, 5, 10, 20]:
        # 1:
        # {"dev": 6.14, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.41, "dev_librispeech": 3.77, "dev_yodas": 11.76,
        #  "test": 6.98, "test_voxpopuli": 6.64, "test_commonvoice": 10.5, "test_librispeech": 4.05, "test_yodas": 12.0}
        # 5:
        # {"dev": 6.16, "dev_voxpopuli": 6.6, "dev_commonvoice": 8.52, "dev_librispeech": 3.79, "dev_yodas": 11.51,
        #  "test": 6.99, "test_voxpopuli": 6.65, "test_commonvoice": 10.63, "test_librispeech": 4.08, "test_yodas": 11.61}
        # 10:
        # {"dev": 6.19, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.59, "dev_librispeech": 3.81, "dev_yodas": 11.42,
        #  "test": 7.0, "test_voxpopuli": 6.66, "test_commonvoice": 10.67, "test_librispeech": 4.1, "test_yodas": 11.54}
        # 20:
        # {"dev": 6.23, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.74, "dev_librispeech": 3.84, "dev_yodas": 11.42,
        #  "test": 7.05, "test_voxpopuli": 6.65, "test_commonvoice": 10.89, "test_librispeech": 4.13, "test_yodas": 11.46}
        enable_every_n = functools.partial(enable_by_interval, interval=interval)
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-every{interval}/{lm_name}",
            task=task,
            ctc_model=am,
            extra_config={"aux_loss_layers": [aux_ctc_layer]},
            lm=lm,
            prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
            ctc_only_recog_version=10,
            ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
            recog_version=12,
            recog_def=model_recog_with_recomb_delayed_fusion_v2,
            first_pass_extra_config={
                "should_convert_labels_now_func": enable_every_n,
                "should_fuse_now_func": enable_every_n,
                "convert_labels_func": convert_labels_func_no_op,
            },
        )

    # never, i.e. delay until the end. this should be like rescoring:
    # {"dev": 6.33, "dev_voxpopuli": 6.66, "dev_commonvoice": 8.96, "dev_librispeech": 3.91, "dev_yodas": 11.6,
    #  "test": 7.1, "test_voxpopuli": 6.65, "test_commonvoice": 11.14, "test_librispeech": 4.19, "test_yodas": 11.14}
    # rescore:
    # {"dev": 6.34, "dev_voxpopuli": 6.66, "dev_commonvoice": 8.97, "dev_librispeech": 3.91, "dev_yodas": 11.63,
    #  "test": 7.1, "test_voxpopuli": 6.64, "test_commonvoice": 11.14, "test_librispeech": 4.19, "test_yodas": 11.15}
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-never/{lm_name}",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": never_enable,
            "should_fuse_now_func": never_enable,
            "convert_labels_func": convert_labels_func_no_op,
        },
    )

    # no delayed prior:
    # {"dev": 6.34, "dev_voxpopuli": 6.66, "dev_commonvoice": 9.02, "dev_librispeech": 3.87, "dev_yodas": 11.81,
    #  "test": 7.16, "test_voxpopuli": 6.69, "test_commonvoice": 11.26, "test_librispeech": 4.21, "test_yodas": 11.37}
    # with delayed prior (default/baseline):
    # {"dev": 6.23, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.74, "dev_librispeech": 3.84, "dev_yodas": 11.42,
    #  "test": 7.05, "test_voxpopuli": 6.65, "test_commonvoice": 10.89, "test_librispeech": 4.13, "test_yodas": 11.46}
    # So no delayed prior makes it worse.
    # ctc_recog_recomb_labelwise_prior_auto_scale(
    #     prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-noDelayedPrior/{lm_name}",
    #     task=task,
    #     ctc_model=am,
    #     extra_config={"aux_loss_layers": [aux_ctc_layer]},
    #     lm=lm,
    #     prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
    #     ctc_only_recog_version=10,
    #     ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
    #     recog_version=12,
    #     recog_def=model_recog_with_recomb_delayed_fusion_v2,
    #     first_pass_extra_config={
    #         "should_convert_labels_now_func": enable_every20,
    #         "should_fuse_now_func": enable_every20,
    #         "convert_labels_func": convert_labels_func_no_op,
    #         "delayed_prior": False,
    #     },
    # )

    # beam sizes for reference/comparison
    # Note: this is delayed fusion.
    # I.e. the LM score is always only applied after the topk pruning, at some delayed time.
    # But for beam size 1, it means, this will never have any effect.
    for beam_size in [1, 2, 4, 64]:
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-beamSize{beam_size}/{lm_name}",
            task=task,
            ctc_model=am,
            extra_config={"aux_loss_layers": [aux_ctc_layer]},
            lm=lm,
            prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
            ctc_only_recog_version=10,
            ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
            recog_version=12,
            recog_def=model_recog_with_recomb_delayed_fusion_v2,
            first_pass_extra_config={
                "should_convert_labels_now_func": enable_every20,
                "should_fuse_now_func": enable_every20,
                "convert_labels_func": convert_labels_func_no_op,
            },
            first_pass_recog_beam_size=beam_size,
        )

    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2/{lm_name}-noPrior",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=lm,
        use_prior=False,
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": enable_every20,
            "should_fuse_now_func": enable_every20,
            "convert_labels_func": convert_labels_func_no_op,
        },
    )

    from i6_experiments.users.zeyer.external_models.qwen2_finetuned import (
        get_lm as get_qwen2_lm,
        get_vocab as get_qwen2_vocab,
    )
    from i6_experiments.users.zeyer.collect_model_dataset_stats import compute_label_prior_log_probs
    from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior, PriorLabelSmoothingJob
    from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob
    from i6_experiments.users.zeyer.decoding.convert_labels import (
        spm_merge_and_lower_case_v3,
        SearchOutputConvertLabelsJob,
    )

    transcriptions_dataset = get_loquacious_text_only_dataset_for_forward(
        vocab=get_qwen2_vocab(text_preprocess_lower_case=True, bpe_dropout=0.1)
    )
    log_lm_vocab_log_prior = compute_label_prior_log_probs(transcriptions_dataset, forward_rqmt={"mem": 12, "time": 24})
    tk.register_output(f"{prefix}/lm/qwen2/log_lm_vocab_log_prior.txt", log_lm_vocab_log_prior)
    log_lm_vocab_log_prior = PriorLabelSmoothingJob(
        prior_file=log_lm_vocab_log_prior, prior_type="log_prob", uniform_weight=0.1, out_prior_type="log_prob"
    ).out_prior
    tk.register_output(f"{prefix}/lm/qwen2/log_lm_vocab_log_prior_smooth.txt", log_lm_vocab_log_prior)

    # for testing
    transcriptions_dataset_small = get_loquacious_text_only_dataset_for_forward(
        vocab=get_qwen2_vocab(text_preprocess_lower_case=True, bpe_dropout=0.1),
        take_random_sorted_subset=5000,
        take_random_sorted_subset_version=2,
    )
    log_lm_vocab_log_prior_small = compute_label_prior_log_probs(
        transcriptions_dataset_small, forward_rqmt={"mem": 12, "time": 24}
    )
    tk.register_output(f"{prefix}/lm/qwen2/log_lm_vocab_log_prior_small.txt", log_lm_vocab_log_prior_small)
    log_lm_vocab_log_prior_small = PriorLabelSmoothingJob(
        prior_file=log_lm_vocab_log_prior_small, prior_type="log_prob", uniform_weight=0.1, out_prior_type="log_prob"
    ).out_prior
    tk.register_output(f"{prefix}/lm/qwen2/log_lm_vocab_log_prior_small_smooth.txt", log_lm_vocab_log_prior_small)

    qwen2_vocab_file = ExtractVocabLabelsJob(get_qwen2_vocab().get_opts()).out_vocab
    tk.register_output(f"{prefix}/lm/qwen2/vocab.txt", qwen2_vocab_file)

    def _prior_convert_labels(out: tk.Path) -> tk.Path:
        return SearchOutputConvertLabelsJob(
            out, source_text_post_process=spm_merge_and_lower_case_v3, target_vocab=get_qwen2_vocab().get_opts()
        ).out_search_results

    qwen2_vocab_prior = Prior(file=log_lm_vocab_log_prior, type="log_prob", vocab=qwen2_vocab_file)

    qwen2_lm = get_qwen2_lm()

    # rescore:
    # {"dev": 6.26, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.84, "dev_librispeech": 3.84, "dev_yodas": 11.54,
    #  "test": 7.0, "test_voxpopuli": 6.66, "test_commonvoice": 10.84, "test_librispeech": 4.1, "test_yodas": 11.11}
    # recog-1stpass:
    # {"dev": 6.57, "dev_voxpopuli": 6.82, "dev_commonvoice": 9.3, "dev_librispeech": 4.07, "dev_yodas": 12.0,
    #  "test": 7.33, "test_voxpopuli": 6.92, "test_commonvoice": 11.23, "test_librispeech": 4.32, "test_yodas": 11.81}
    convert_labels_func_spm = functools.partial(
        convert_labels_func,
        is_am_label_word_start=spm_space_first_is_word_start,
        custom_am_label_merge=spm_label_merge_v2,
        seq_str_postprocess_func=seq_str_postprocess_lower_case,
    )
    res = ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2/qwen2",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=qwen2_lm,
        lm_rescore_config={
            "default_data_convert_labels_func": convert_labels_func_spm,
            "chunk_size_for_lm_rescoring": 16,
            "max_seqs": 32,
        },
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": enable_every20,
            "should_fuse_now_func": enable_every20,
            # specific to the AM SPM that we have here...
            "convert_labels_func": convert_labels_func_spm,
            "max_seqs": 32,
            # "___debug": 2,  # add something new random to get new hashes for debugging
        },
    )
    tk.register_output(
        f"{prefix}/aed/{name}/ctc+lm-delayed-v2/qwen2/recog-1stpass-res-dev-yodas.txt",
        res.individual_results["dev_yodas"].main_measure_value,
    )

    # Using the Qwen2-vocab prior
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-qwenPrior/qwen2",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=qwen2_lm,
        lm_rescore_config={
            "default_data_convert_labels_func": convert_labels_func_spm,
            "chunk_size_for_lm_rescoring": 16,
            "max_seqs": 32,
        },
        labelwise_prior=qwen2_vocab_prior,
        prior_custom_vocab_convert_labels=_prior_convert_labels,
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": enable_every20,
            "should_fuse_now_func": enable_every20,
            # specific to the AM SPM that we have here...
            "convert_labels_func": convert_labels_func_spm,
            # TODO somehow fix the prior...?
            "max_seqs": 32,
        },
    )

    # batch size 1
    # {"dev": 6.41, "dev_voxpopuli": 6.73, "dev_commonvoice": 9.03, "dev_librispeech": 3.98, "dev_yodas": 11.91,
    #  "test": 7.15, "test_voxpopuli": 6.82, "test_commonvoice": 10.91, "test_librispeech": 4.19, "test_yodas": 11.62}
    # baseline with batch size 32 is:
    # {"dev": 6.57, "dev_voxpopuli": 6.82, "dev_commonvoice": 9.3, "dev_librispeech": 4.07, "dev_yodas": 12.0,
    #  "test": 7.33, "test_voxpopuli": 6.92, "test_commonvoice": 11.23, "test_librispeech": 4.32, "test_yodas": 11.81}
    # TODO figure out where the inconsistency comes from...
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-batchSize1/qwen2",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=qwen2_lm,
        lm_rescore_config={
            "default_data_convert_labels_func": convert_labels_func_spm,
            "chunk_size_for_lm_rescoring": 16,
            "max_seqs": 32,
        },
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": enable_every20,
            "should_fuse_now_func": enable_every20,
            # specific to the AM SPM that we have here...
            "convert_labels_func": convert_labels_func_spm,
            "max_seqs": 1,
            "__batch_size_dependent": True,
        },
    )

    # no delayed prior:
    # {"dev": 6.61, "dev_voxpopuli": 6.78, "dev_commonvoice": 9.55, "dev_librispeech": 4.08, "dev_yodas": 12.23,
    #  "test": 7.41, "test_voxpopuli": 6.95, "test_commonvoice": 11.62, "test_librispeech": 4.36, "test_yodas": 11.66}
    # with delayed prior (default/baseline):
    # {"dev": 6.57, "dev_voxpopuli": 6.82, "dev_commonvoice": 9.3, "dev_librispeech": 4.07, "dev_yodas": 12.0,
    #  "test": 7.33, "test_voxpopuli": 6.92, "test_commonvoice": 11.23, "test_librispeech": 4.32, "test_yodas": 11.81}
    # So for qwen same as for our LM, no delayed prior makes it worse.
    # ctc_recog_recomb_labelwise_prior_auto_scale(
    #     prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-noDelayedPrior/qwen2",
    #     task=task,
    #     ctc_model=am,
    #     extra_config={"aux_loss_layers": [aux_ctc_layer]},
    #     lm=qwen2_lm,
    #     lm_rescore_config={
    #         "default_data_convert_labels_func": convert_labels_func_spm,
    #         "chunk_size_for_lm_rescoring": 16,
    #         "max_seqs": 32,
    #     },
    #     prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
    #     ctc_only_recog_version=10,
    #     ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
    #     recog_version=12,
    #     recog_def=model_recog_with_recomb_delayed_fusion_v2,
    #     first_pass_extra_config={
    #         "should_convert_labels_now_func": enable_every20,
    #         "should_fuse_now_func": enable_every20,
    #         # specific to the AM SPM that we have here...
    #         "convert_labels_func": convert_labels_func_spm,
    #         "max_seqs": 32,
    #         "delayed_prior": False,
    #     },
    # )

    # No prior:
    # rescore:
    # {"dev": 6.5, "dev_voxpopuli": 6.84, "dev_commonvoice": 9.19, "dev_librispeech": 4.01, "dev_yodas": 12.01,
    #  "test": 7.26, "test_voxpopuli": 6.79, "test_commonvoice": 11.28, "test_librispeech": 4.28, "test_yodas": 11.73}
    # recog-1stpass:
    # {"dev": 6.79, "dev_voxpopuli": 6.98, "dev_commonvoice": 9.62, "dev_librispeech": 4.16, "dev_yodas": 13.18,
    #  "test": 7.65, "test_voxpopuli": 7.07, "test_commonvoice": 11.63, "test_librispeech": 4.48, "test_yodas": 13.21}
    # With prior:
    # rescore:
    # {"dev": 6.26, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.84, "dev_librispeech": 3.84, "dev_yodas": 11.54,
    #  "test": 7.0, "test_voxpopuli": 6.66, "test_commonvoice": 10.84, "test_librispeech": 4.1, "test_yodas": 11.11}
    # recog-1stpass:
    # {"dev": 6.57, "dev_voxpopuli": 6.82, "dev_commonvoice": 9.3, "dev_librispeech": 4.07, "dev_yodas": 12.0,
    #  "test": 7.33, "test_voxpopuli": 6.92, "test_commonvoice": 11.23, "test_librispeech": 4.32, "test_yodas": 11.81}
    # So without prior makes it clearly worse.
    # ctc_recog_recomb_labelwise_prior_auto_scale(
    #     prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2/qwen2-noPrior",
    #     task=task,
    #     ctc_model=am,
    #     extra_config={"aux_loss_layers": [aux_ctc_layer]},
    #     lm=qwen2_lm,
    #     lm_rescore_config={
    #         "default_data_convert_labels_func": convert_labels_func_spm,
    #         "chunk_size_for_lm_rescoring": 16,
    #         "max_seqs": 32,
    #     },
    #     use_prior=False,
    #     ctc_only_recog_version=10,
    #     ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
    #     recog_version=12,
    #     recog_def=model_recog_with_recomb_delayed_fusion_v2,
    #     first_pass_extra_config={
    #         "should_convert_labels_now_func": enable_every20,
    #         "should_fuse_now_func": enable_every20,
    #         # specific to the AM SPM that we have here...
    #         "convert_labels_func": convert_labels_func_spm,
    #         "max_seqs": 32,
    #     },
    # )

    # Always (every1) (no delayed fusion, or only until word end):
    # {"dev": 9.99, "dev_voxpopuli": 10.69, "dev_commonvoice": 12.52, "dev_librispeech": 6.58, "dev_yodas": 21.71,
    #  "test": 10.98, "test_voxpopuli": 10.52, "test_commonvoice": 15.07, "test_librispeech": 6.58, "test_yodas": 21.39}
    # TODO why is this even worse?
    enable_always = functools.partial(enable_by_interval, interval=1)
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-always/qwen2",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=qwen2_lm,
        lm_rescore_config={
            "default_data_convert_labels_func": convert_labels_func_spm,
            "chunk_size_for_lm_rescoring": 16,
            "max_seqs": 32,
        },
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": enable_always,
            "should_fuse_now_func": enable_always,
            # specific to the AM SPM that we have here...
            "convert_labels_func": convert_labels_func_spm,
            "max_seqs": 32,
        },
    )

    # Never (delay until end), i.e. should be like rescoring:
    # {"dev": 6.26, "dev_voxpopuli": 6.61, "dev_commonvoice": 8.83, "dev_librispeech": 3.85, "dev_yodas": 11.47,
    #  "test": 7.0, "test_voxpopuli": 6.67, "test_commonvoice": 10.84, "test_librispeech": 4.1, "test_yodas": 11.1}
    # Rescoring:
    # {"dev": 6.26, "dev_voxpopuli": 6.62, "dev_commonvoice": 8.84, "dev_librispeech": 3.84, "dev_yodas": 11.54,
    #  "test": 7.0, "test_voxpopuli": 6.66, "test_commonvoice": 10.84, "test_librispeech": 4.1, "test_yodas": 11.11}
    ctc_recog_recomb_labelwise_prior_auto_scale(
        prefix=f"{prefix}/aed/{name}/ctc+lm-delayed-v2-never/qwen2",
        task=task,
        ctc_model=am,
        extra_config={"aux_loss_layers": [aux_ctc_layer]},
        lm=qwen2_lm,
        lm_rescore_config={
            "default_data_convert_labels_func": convert_labels_func_spm,
            "chunk_size_for_lm_rescoring": 16,
            "max_seqs": 32,
        },
        prior_dataset=get_loquacious_train_subset_dataset_v2(vocab=vocab),
        ctc_only_recog_version=10,
        ctc_only_recog_def=model_recog_with_recomb,  # keep hash for first ctc-only pass
        recog_version=12,
        recog_def=model_recog_with_recomb_delayed_fusion_v2,
        first_pass_extra_config={
            "should_convert_labels_now_func": never_enable,
            "should_fuse_now_func": never_enable,
            # specific to the AM SPM that we have here...
            "convert_labels_func": convert_labels_func_spm,
            "max_seqs": 32,
        },
    )

    # TODO new prior on LM vocab but ASR transcriptions...
    # TODO not fixed interval but condition like in delayed fusion paper
    # TODO ASR prior but divide by num ASR labels, mult by num LM labels
    # TODO normal ASR prior also using counting on the train transcriptions


_base_config = {
    # ("large", 100),  # 100kh in total, 4 full epochs
    # ("large", 150),  # 150kh in total, 6 full epochs
    # ("large", 200),  # 200kh in total, 8 full epochs
    # ("large", 250),  # 250kh in total, 10 full epochs
    # ("large", 500),  # 500kh in total, 20 full epochs
    "subset": "large",
    "total_k_hours": 100,
    "vocab": "spm10k",
    "model": {
        "behavior_version": 24,
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
        # Default AED decoder size: 6 layers, 512 dim
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            # When only trained on LS ASR data, keep the default dropout?
            # dropout=0.0,
            # att_dropout=0.0,
        ),
        "feature_batch_norm": True,
    },
    "train_update_func_from_n_ep": lambda n_ep: {"train": configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5)},
    "train": dict_update_deep(
        configs.config_96gb_bf16_accgrad1,
        {
            "batch_size": 100_000 * configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            # "__train_audio_preprocess": speed_pert_librosa_config,
            # "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    ),
    "train_post": dict_update_deep(
        configs.post_config, {"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}}
    ),
    # TODO there is a bug, these are not used...
    "train_vocab_opts": {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    # TODO there is a bug, these are not used...
    "dataset_train_opts": {"train_epoch_split": 1, "train_epoch_wise_filter": None},
    "env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}


def get_am(
    name: str, config: Dict[str, Any], config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[ModelWithCheckpoint, int]:
    prefix = get_setup_prefix_for_module(__name__)

    config = dict_update_deep(_base_config.copy(), config.copy())
    config = dict_update_deep(config, config_overrides, dict_value_merge=False)

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    subset = config.pop("subset")
    total_k_hours = config.pop("total_k_hours")
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)

    train_update_func_from_n_ep = config.pop("train_update_func_from_n_ep")
    if train_update_func_from_n_ep:
        config = dict_update_deep(config, train_update_func_from_n_ep(n_ep))

    model_config = config.pop("model")
    train_config: Dict[str, Any] = config.pop("train")
    post_config = config.pop("train_post")

    vocab = config.pop("vocab", "spm10k")
    task = get_loquacious_task_raw_v2(vocab=vocab, subset_name=subset, train_epoch_split=train_epoch_split)

    train_vocab_opts = config.pop("train_vocab_opts")
    dataset_train_opts = config.pop("dataset_train_opts")
    env_updates = config.pop("env_updates")

    assert not config

    exp = aed_train_exp(
        name,
        train_config,
        prefix=prefix + "/aed/",
        task=task,
        model_config=model_config,
        post_config_updates=post_config,
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        dataset_train_opts=dataset_train_opts,
        env_updates=env_updates,
    )
    return exp.get_last_fixed_epoch(), max(
        [i for i in train_config["aux_loss_layers"] if i <= model_config["enc_build_dict"]["num_layers"]]
    )


@cache
def get_lm(*, prefix: str, vocab: str, num_full_ep: int = 5, split: int = 10) -> Tuple[str, ModelWithCheckpoint]:
    from sisyphus import tk
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        config_96gb_bf16_accgrad1,
        _get_cfg_lrlin_oclr_by_bs_nep_v4,
    )
    from i6_experiments.users.zeyer.decoding.perplexity import (
        get_lm_perplexities_for_task_evals,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
    from i6_experiments.users.zeyer.train_v4 import train as _train, ModelDefWithCfg

    from i6_experiments.users.zeyer.datasets.loquacious import (
        get_loquacious_task_raw_v2,
        get_loquacious_text_only_dataset,
    )

    import returnn.frontend as rf
    from returnn.frontend.decoder.transformer import TransformerDecoder

    n_ep = round(num_full_ep * split)
    # orig name: trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop01-b400_20k-nEp...-spm10k
    name = f"trafo-n32-d1024-nFullEp{num_full_ep}-nEp{n_ep}-{vocab}"
    exp = _train(
        f"{prefix}/lm/{name}",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep),
                "batch_size": 20_000,
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_loquacious_text_only_dataset(vocab="spm10k", train_epoch_split=split),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=32,
                    model_dim=1024,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.1,
                    att_dropout=0.1,
                )
            },
        ),
        train_def=lm_train_def,
    )

    task = get_loquacious_task_raw_v2(vocab=vocab)
    perplexities_nlm = get_lm_perplexities_for_task_evals(task, label_level="task", lm=exp.get_last_fixed_epoch())
    for eval_set_name, ppl in perplexities_nlm.items():
        tk.register_output(f"{prefix}/lm/{name}/ppl/{eval_set_name}", ppl)

    return name, exp.get_last_fixed_epoch()
