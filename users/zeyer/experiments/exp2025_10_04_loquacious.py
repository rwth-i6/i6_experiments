"""
Some Loquacious baselines
"""

from __future__ import annotations

from sisyphus import tk, Path
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
    get_ctc_prior,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_lm import (
    aed_ctc_lm_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.decoding.perplexity import (
    get_ngram_perplexities_for_task_evals,
    get_lm_perplexities_for_task_evals,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
from i6_experiments.users.zeyer.train_v4 import train, ModelDefWithCfg
from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_torchaudio_ngram import (
    ctc_recog_ngram_lm_framewise_prior_auto_scale,
    model_recog_torchaudio,
    get_ctc_with_ngram_lm_and_framewise_prior,
    get_lexicon_from_task,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw,
    get_loquacious_task_raw_v2,
    get_loquacious_text_only_dataset,
    get_loquacious_train_subset_dataset,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

__setup_root_prefix__ = "exp2025_10_04_loquacious"


_public_vocab_word_list = Path(
    "/hpcwork/p0023999/az668407/loquacious/LoquaciousAdditionalResources/loquacious-vocab.txt",
    hash_overwrite="loquacious-2025-public-vocab-word-list",
)

_public_4gram_lm = Path(
    "/hpcwork/p0023999/az668407/loquacious/LoquaciousAdditionalResources/4gram-pruned-test2.arpa.gz",
    hash_overwrite="loquacious-2025-public-4gram-lm",
)


def py():
    prefix = get_setup_prefix_for_module(__name__)
    task_spm10k = get_loquacious_task_raw(vocab="spm10k")

    # Librispeech baseline name:
    # EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k
    # -> here: "base"
    name = "base"
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        task=task_spm10k,
        model_config={
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
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
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
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

    name = "base-bRnd"
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        task=get_loquacious_task_raw(vocab="spm10k", train_seq_ordering="random"),
        model_config={
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
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
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
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

    task_spm10k = get_loquacious_task_raw_v2(vocab="spm10k")

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    selected_asr = None
    ams = {}
    for subset, total_k_hours in [
        ("clean", 4 * 13),  # 52kh in total, 4 full epochs
        ("clean", 100),  # 100kh in total, 7.7 full epochs
        ("small", 25),  # 25kh in total, 100 full epochs
        ("small", 50),  # 50kh in total, 200 full epochs
        ("medium", 25),  # 25kh in total, 10 full epochs
        ("medium", 50),  # 50kh in total, 20 full epochs
        ("medium", 100),  # 100kh in total, 25 full epochs
        ("large", 25),  # 25kh in total, 1 full epoch
        ("large", 50),  # 50kh in total, 2 full epochs
        ("large", 100),  # 100kh in total, 4 full epochs
        ("large", 150),  # 150kh in total, 6 full epochs
        ("large", 200),  # 200kh in total, 8 full epochs
        ("large", 250),  # 250kh in total, 10 full epochs
        ("large", 500),  # 500kh in total, 20 full epochs
        ("large", 1000),  # 1Mh in total, 40 full epochs
        ("large", 2000),  # 2Mh in total, 80 full epochs
        ("large", 2500),  # 2.5Mh in total, 100 full epochs
    ]:
        train_epoch_split = train_epoch_split_per_subset[subset]
        num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
        n_ep = round(num_full_ep * train_epoch_split)
        name = f"base-v2-{subset}-nFullEp{num_full_ep:.1f}-nEp{n_ep}-totalHours{total_k_hours}k"
        exp = aed_train_exp(
            name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            task=get_loquacious_task_raw_v2(vocab="spm10k", subset_name=subset, train_epoch_split=train_epoch_split),
            model_config={
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
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
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
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        ams[(subset, total_k_hours)] = (name, exp.get_last_fixed_epoch())
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
        )
        if subset == "large" and total_k_hours == 200:  # for now only this
            selected_asr = (name, exp.get_last_fixed_epoch())

    # Language models on train large transcriptions

    selected_lm = None
    lms = {}
    for num_full_ep, split in [(4, 25), (5, 10), (10, 10), (20, 10), (30, 10)]:
        n_ep = round(num_full_ep * split)
        # orig name: trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop01-b400_20k-nEp...-spm10k
        name = f"trafo-n32-d1024-nFullEp{num_full_ep}-nEp{n_ep}-spm10k"
        exp = train(
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
                        decoder_layer_opts=dict(
                            self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)
                        ),
                        dropout=0.1,
                        att_dropout=0.1,
                    )
                },
            ),
            train_def=lm_train_def,
        )
        lms[name] = exp.get_last_fixed_epoch()

        perplexities_nlm = get_lm_perplexities_for_task_evals(
            task_spm10k, label_level="task", lm=exp.get_last_fixed_epoch()
        )
        for eval_set_name, ppl in perplexities_nlm.items():
            tk.register_output(f"{prefix}/lm/{name}/ppl/{eval_set_name}", ppl)

        aed_ctc_lm_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + selected_asr[0] + "/ctc+lm/" + name,
            task=task_spm10k,
            aed_ctc_model=selected_asr[1],
            aed_scale=0.0,
            aux_ctc_layer=16,
            lm=exp.get_last_fixed_epoch(),
        )

        if num_full_ep == 4:
            selected_lm = (name, exp.get_last_fixed_epoch())

    # AED+CTC+LM decoding

    # Note prior does not work yet...
    #   this is because get_ctc_prior_probs(..., task.train_dataset.copy_train_as_static(), ...):
    #   copy_train_as_static not implemented.
    #   but also, do we really want to do this on the full dataset?

    # TODO do this for all LMs? (or some selection of them)
    aed_ctc_lm_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + selected_asr[0] + "/aed+ctc+lm/" + selected_lm[0],
        task=task_spm10k,
        aed_ctc_model=selected_asr[1],
        aed_scale_max=20.0,
        aux_ctc_layer=16,
        lm=selected_lm[1],
        lm_scale_max=10.0,
    )

    perplexities_4gram = get_ngram_perplexities_for_task_evals(task_spm10k, label_level="word", lm=_public_4gram_lm)
    for eval_set_name, ppl in perplexities_4gram.items():
        tk.register_output(f"{prefix}/lm/4gram/ppl/{eval_set_name}", ppl)

    lexicon = get_lexicon_from_task(task_spm10k, lm_word_list=_public_vocab_word_list)
    for subset, total_k_hours in [
        ("small", 25),
        ("large", 100),
        ("large", 150),
        ("large", 200),
        ("large", 250),
        ("large", 500),
    ]:
        name, am = ams[(subset, total_k_hours)]

        aed_ctc_lm_timesync_recog_recomb_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm/{selected_lm[0]}",
            task=task_spm10k,
            aed_ctc_model=am,
            aed_scale=0.0,
            aux_ctc_layer=16,
            lm=selected_lm[1],
        )

        ctc_recog_ngram_lm_framewise_prior_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm/4gram",
            task=task_spm10k,
            ctc_model=am,
            extra_config={"aux_loss_layers": [16]},
            framewise_prior_dataset=get_loquacious_train_subset_dataset(vocab="spm10k"),
            ngram_language_model=_public_4gram_lm,
            lm_word_list=_public_vocab_word_list,
            ctc_decoder_opts={"beam_size": 1024, "beam_size_token": 16, "beam_threshold": 14},
        )

        if subset == "small":
            framewise_prior = get_ctc_prior(
                ctc_model=am,
                extra_config={"aux_loss_layers": [16]},
                task=task_spm10k,
                framewise_prior_dataset=get_loquacious_train_subset_dataset(vocab="spm10k"),
            )
            for lm_scale, prior_scale in [
                (0.5, 0.1),
                (0.5, 0.25),
                (0.5, 0.5),
                (1.0, 0.1),
                (1.0, 0.5),
                (1.0, 1.0),
                (2.0, 0.3),
                (2.0, 0.5),
                (2.0, 1.0),
                (2.0, 1.5),
            ]:
                model = get_ctc_with_ngram_lm_and_framewise_prior(
                    ctc_model=am,
                    prior=framewise_prior.file,
                    prior_type=framewise_prior.type,
                    prior_scale=prior_scale,
                    ngram_language_model=_public_4gram_lm,
                    lm_scale=lm_scale,
                    ctc_decoder_opts={
                        "lexicon": lexicon,
                        "beam_size": 1024,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                    },
                )
                prefix_ = (
                    f"{prefix}/aed/{name}/ctc+lm/4gram-fixedScales/recog-1stpass-res-lm{lm_scale}-prior{prior_scale}"
                )
                res = recog_model(
                    task=task_spm10k,
                    model=model,
                    recog_def=model_recog_torchaudio,
                    config={
                        "behavior_version": 24,  # should make it independent from batch size
                        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
                        "aux_loss_layers": [16],
                        "batch_size": int(20_000 * am.definition.batch_size_factor),
                    },
                    search_rqmt={"time": 24, "mem": 32},
                    name=prefix_,
                )
                tk.register_output(prefix_ + ".txt", res.output)
