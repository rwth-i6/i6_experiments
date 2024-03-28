"""
Beam search related experiments.
(Note that this is for more custom beam search variations.
 Most setups here (e.g. in aed.py) also come with a very simple standard beam search implementation.)
"""


def py():
    from .configs import config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k, _batch_size_factor
    from .aed import (
        train_exp as aed_train_exp,
        recog as aed_recog,
        model_recog_pure_torch as aed_model_recog_pure_torch,
        model_recog_dyn_beam_pure_torch as aed_model_recog_dyn_beam_pure_torch,
    )
    from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

    model = aed_train_exp(  # 5.11 (!!)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )

    # All beam search experiments using model_recog_pure_torch, beam_search_v5.
    for name, recog_config in {
        "beam12-batch200-lenNorm1": {
            # {"dev-clean": 2.35, "dev-other": 5.14, "test-clean": 2.47, "test-other": 5.72}
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
        },
        "beam12-batch200-lenReward01": {
            # {"dev-clean": 2.35, "dev-other": 5.14, "test-clean": 2.47, "test-other": 5.66}
            "beam_size": 12,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
        },
        "beam12-batch200-lenReward02": {
            "beam_size": 12,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.2,
        },
        "beam12-batch200-lenNorm0": {
            "beam_size": 12,
            "length_normalization_exponent": 0.0,
        },
        "beam12-batch1-lenNorm1": {
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
            "max_seqs": 1,
        },
        "beam12-batch1-lenReward01": {
            "beam_size": 12,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
            "max_seqs": 1,
        },
        "beam12-batch1-lenNorm0": {
            "beam_size": 12,
            "length_normalization_exponent": 0.0,
            "max_seqs": 1,
        },
        "beam60-batch50-lenNorm1": {
            # {"dev-clean": 2.36, "dev-other": 5.18, "test-clean": 2.47, "test-other": 5.64}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 1.0,
            "__with_cheating": True,
        },
        "beam60-batch50-lenReward01": {
            # {"dev-clean": 2.39, "dev-other": 5.19, "test-clean": 2.5, "test-other": 5.56}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
            "__with_cheating": True,
        },
        "beam60-batch1-lenReward01": {
            "beam_size": 60,
            "max_seqs": 1,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0": {
            # {"dev-clean": 2.39, "dev-other": 5.21, "test-clean": 2.54, "test-other": 5.55}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.0,
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm02-cov05": {
            # {"dev-clean": 2.39, "dev-other": 5.17, "test-clean": 2.52, "test-other": 5.57}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.5,
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-cov02-covInd": {
            # {"dev-clean": 2.34, "dev-other": 5.17, "test-clean": 2.5, "test-other": 5.56}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.2,
                "attention_coverage_opts": {"type": "indicator"},
            },
            "__with_cheating": True,
        },
    }.items():
        aed_recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2/recog_last_" + name,
            model.get_last_fixed_epoch(),
            aed_model_recog_pure_torch,
            {
                "beam_search_version": 5,
                "__batch_size_dependent": True,
                "__recog_def_ext": True,
                "beam_search_collect_individual_seq_scores": True,
                **recog_config,
            },
        )
    # All beam search experiments using model_recog_dyn_beam_pure_torch, beam_search_dyn_beam.
    for name, recog_config in {
        "beam1-batch1-lenReward01-v2": {
            "beam_search_opts": {
                "beam_size": 1,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "beam_search_version": 2,
            "max_seqs": 1,
        },
        "beam1-batch200-lenReward01": {
            "beam_search_opts": {
                "beam_size": 1,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
        },
        "beam1-batch200-lenReward01-v2": {
            "beam_search_opts": {
                "beam_size": 1,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "beam_search_version": 2,
        },
        "beam12-batch200-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 1.0,
            },
        },
        "beam12-batch1-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 1.0,
            },
            "max_seqs": 1,
        },
        "beam12-batch1-lenNorm0": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
            },
            "max_seqs": 1,
        },
        "beam12-batch200-lenReward01": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
        },
        "beam12-batch200-lenReward01-v2": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "beam_search_version": 2,
        },
        "beam12-batch200-lenReward02-v2": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
            },
            "beam_search_version": 2,
        },
        "beam12-batch1-lenReward01-v2": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 1,
            "beam_search_version": 2,
        },
        "beam60-batch50-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 1.0,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenReward01": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenReward01-v2": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_version": 2,
        },
        "beam60-batch1-lenReward01-v2": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 1,
            "beam_search_version": 2,
        },
        "beam60-batch50-lenNorm0-lenReward0": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
    }.items():
        for k, v in {
            "beam_search_version": 1,
            "__batch_size_dependent": True,
            "__recog_def_ext": True,
            "beam_search_collect_individual_seq_scores": True,
        }.items():
            recog_config.setdefault(k, v)
        recog_config["beam_search_opts"].setdefault(
            "beam_and_ended_size", recog_config["beam_search_opts"]["beam_size"]
        )
        aed_recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2/recog_last_dyn_" + name,
            model.get_last_fixed_epoch(),
            aed_model_recog_dyn_beam_pure_torch,
            recog_config,
        )
    # All beam search experiments using model_recog_pure_torch, beam_search_sep_ended.
    for name, recog_config in {
        "beam12-batch1-lenReward01": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 1,
        },
        "beam12-batch200-lenReward01": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
        },
        "beam12-batch200-lenReward02": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
            },
        },
        "beam12-batch200-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 1.0,
            },
        },
        "beam60-batch1-lenReward01": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 1,
        },
        "beam60-batch50-lenReward01": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 1.0,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenNorm0-lenReward0": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
    }.items():
        for k, v in {
            "beam_search_version": "sep_ended",
            "__batch_size_dependent": True,
            "__recog_def_ext": True,
            "beam_search_collect_individual_seq_scores": True,
        }.items():
            recog_config.setdefault(k, v)
        recog_config["beam_search_opts"].setdefault(
            "beam_and_ended_size", recog_config["beam_search_opts"]["beam_size"]
        )
        aed_recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2/recog_last_sep_" + name,
            model.get_last_fixed_epoch(),
            aed_model_recog_pure_torch,
            recog_config,
        )
    # All beam search experiments using model_recog_pure_torch, beam_search_sep_ended_keep_v6.
    for name, recog_config in {
        "beam12-batch200-lenReward01": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
        },
        "beam12-batch200-lenReward02": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
            },
        },
        "beam12-batch200-lenReward03": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.3,
            },
        },
        "beam12-batch200-lenReward01-thresh10": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 10.0,
            },
        },
        "beam12-batch1-lenReward01-thresh10": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 10.0,
            },
            "max_seqs": 1,
        },
        "beam12-batch200-lenReward01-thresh5": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 5.0,
            },
        },
        "beam12-batch200-lenReward01-thresh2": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 2.0,
            },
        },
        "beam12-batch200-lenReward01-thresh0": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 0.0,
            },
        },
        "beam12-batch200-lenReward01-thresh5-threshW0": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 5.0,
                "pruning_threshold_worst": 0.0,
            },
        },
        "beam12-beamEnd1-batch200-lenReward01-thresh2": {
            "beam_search_opts": {
                "beam_size": 12,
                "beam_ended_size": 1,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 2.0,
            },
        },
        "beam12-batch200-lenReward02-thresh2": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
                "pruning_threshold": 2.0,
            },
        },
        "beam12-batch200-lenReward02-thresh2-zeros01": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
                "pruning_threshold": 2.0,
            },
            "data_concat_zeros": 0.1,
        },
        "beam12-batch200-lenReward02-thresh2-cov05": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.5,
            },
        },
        "beam12-batch200-lenReward01-thresh2-cov05": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.5,
            },
        },
        "beam12-batch200-lenReward0-thresh2-cov05": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.0,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.5,
            },
        },
        "beam12-batch200-lenReward0-thresh2-cov05-covInd": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.0,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
            },
        },
        "beam12-batch200-lenReward_01-thresh2-cov05": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": -0.1,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.5,
            },
        },
        "beam12-batch200-lenReward0-thresh2-cov03": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.0,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.3,
            },
        },
        "beam12-batch200-lenReward0-thresh2-cov07": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.0,
                "pruning_threshold": 2.0,
                "attention_coverage_scale": 0.7,
            },
        },
        "beam12-batch200-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 12,
                "length_normalization_exponent": 1.0,
            },
        },
        "beam60-batch50-lenReward01": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.1,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenReward02": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": 0.2,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenNorm1": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 1.0,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenNorm0-lenReward0": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "beam60-batch50-lenReward_01-thresh5-cov05": {
            "beam_search_opts": {
                "beam_size": 60,
                "length_normalization_exponent": 0.0,
                "length_reward": -0.1,
                "pruning_threshold": 5.0,
                "attention_coverage_scale": 0.5,
            },
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
    }.items():
        for k, v in {
            "beam_search_version": "sep_ended_keep_v6",
            "__batch_size_dependent": True,
            "__recog_def_ext": True,
            "beam_search_collect_individual_seq_scores": True,
        }.items():
            recog_config.setdefault(k, v)
        recog_config["beam_search_opts"].setdefault("beam_ended_size", recog_config["beam_search_opts"]["beam_size"])
        aed_recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2/recog_last_keep_" + name,
            model.get_last_fixed_epoch(),
            aed_model_recog_pure_torch,
            recog_config,
        )

    model = aed_train_exp(  # 5.44 ("test-other": 6.34), worse than speedpertV2 (5.11)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV3",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )
    for name, recog_config in {
        "beam12-batch200-lenNorm1": {
            # {"dev-clean": 2.64, "dev-other": 5.44, "test-clean": 2.65, "test-other": 6.33}
            # WTF, why is dev-other here much better? test-clean as well.
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
            "__with_cheating": True,
        },
        "beam12-batch1-lenNorm1": {
            # {"dev-clean": 3.38, "dev-other": 6.23, "test-clean": 2.9, "test-other": 6.26}
            # WTF, why is this so much worse than batch50?
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
            "max_seqs": 1,
        },
        "beam12-lenNorm0-cov05-covInd-batch200": {
            # {"dev-clean": 2.6, "dev-other": 6.28, "test-clean": 2.68, "test-other": 6.69}
            # same beam60, batch50: {"dev-clean": 2.57, "dev-other": 5.48, "test-clean": 2.65, "test-other": 5.94}
            "beam_size": 12,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
            },
        },
        "beam60-batch50-lenNorm1": {
            # {"dev-clean": 2.92, "dev-other": 6.2, "test-clean": 2.84, "test-other": 6.52}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 1.0,
        },
        "beam60-batch50-lenNorm0": {
            # {"dev-clean": 2.64, "dev-other": 5.4, "test-clean": 2.82, "test-other": 6.02}
            # test-other: work/i6_core/recognition/scoring/ScliteJob.hHxGodUNMmaC/output
            # Percent Substitution      =    4.3%   (2254)
            # Percent Deletions         =    0.9%   ( 477)
            # Percent Insertions        =    0.8%   ( 422)
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.0,
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-zeros01": {
            # {"dev-clean": 2.59, "dev-other": 5.38, "test-clean": 2.6, "test-other": 6.64}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.0,
            "data_concat_zeros": 0.1,
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-mono005-modAttAvg": {
            # {"dev-clean": 2.64, "dev-other": 5.4, "test-clean": 2.82, "test-other": 6.02}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_opts": {"model_att_reduce_type": "avg"},
                "attention_monotonicity_scale": 0.05,
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm01-mono005-modAttAvg": {
            # {"dev-clean": 2.64, "dev-other": 5.41, "test-clean": 2.83, "test-other": 6.49}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.1,
                "attention_coverage_opts": {"model_att_reduce_type": "avg"},
                "attention_monotonicity_scale": 0.05,
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm02": {
            # {"dev-clean": 2.64, "dev-other": 5.4, "test-clean": 2.83, "test-other": 6.5}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
            },
        },
        "beam60-batch50-lenNorm02-cov02": {
            # {"dev-clean": 2.6, "dev-other": 5.4, "test-clean": 2.81, "test-other": 6.5}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.2,
            },
        },
        "beam60-batch50-lenNorm02-cov05": {
            # {"dev-clean": 2.6, "dev-other": 5.4, "test-clean": 2.61, "test-other": 6.47}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.5,
            },
        },
        "beam60-batch50-lenNorm02-cov1": {
            # {"dev-clean": 2.62, "dev-other": 5.44, "test-clean": 2.57, "test-other": 6.48}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 1.0,
            },
        },
        "beam60-batch50-lenNorm02-cov2": {
            # {"dev-clean": 2.55, "dev-other": 5.74, "test-clean": 2.82, "test-other": 6.04}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 2.0,
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm02-cov3": {
            # {"dev-clean": 3.53, "dev-other": 6.62, "test-clean": 3.76, "test-other": 6.76}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 3.0,
            },
        },
        "beam60-batch50-lenNorm02-cov005-covLog": {
            # {"dev-clean": 2.6, "dev-other": 5.37, "test-clean": 2.81, "test-other": 6.48}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.05,
                "attention_coverage_opts": {"type": "log"},
            },
        },
        "beam60-lenNorm02-cov02-covLog-batch50": {
            # {"dev-clean": 2.59, "dev-other": 5.4, "test-clean": 2.68, "test-other": 6.47}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.2,
                "attention_coverage_opts": {"type": "log"},
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm02-cov02-covLogEps01": {
            # {"dev-clean": 2.6, "dev-other": 5.41, "test-clean": 2.6, "test-other": 6.46}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.2,
                "attention_coverage_opts": {"type": "log", "eps": 0.1, "clip_min": 0.0},
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm02-cov2-covLog": {
            # {"dev-clean": 38.31, "dev-other": 42.15, "test-clean": 40.45, "test-other": 44.72}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 2.0,
                "attention_coverage_opts": {"type": "log"},
            },
        },
        "beam60-batch50-lenNorm02-cov2-covLogEps01": {
            # {"dev-clean": 17.98, "dev-other": 20.82, "test-clean": 21.22, "test-other": 24.63}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 2.0,
                "attention_coverage_opts": {"type": "log", "eps": 0.1, "clip_min": 0.0},
            },
        },
        "beam60-batch50-lenNorm02-cov2-covLogEps01-covRescale": {
            # {"dev-clean": 40.54, "dev-other": 48.15, "test-clean": 40.77, "test-other": 48.35}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 2.0,
                "attention_coverage_opts": {"type": "log", "eps": 0.1, "clip_min": 0.0, "rescale": True},
            },
        },
        "beam60-batch50-lenNorm02-lenReward03-mono0025-modAttAvg": {
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "length_reward": 0.3,
                "attention_coverage_opts": {"model_att_reduce_type": "avg"},
                "attention_monotonicity_scale": 0.025,
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-cov02-covInd": {
            # {"dev-clean": 2.59, "dev-other": 5.4, "test-clean": 2.59, "test-other": 6.47}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.2,
                "attention_coverage_opts": {"type": "indicator"},
            },
        },
        "beam60-batch50-lenNorm02-cov02-covInd": {
            # {"dev-clean": 2.6, "dev-other": 5.41, "test-clean": 2.59, "test-other": 6.49}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.2,
                "attention_coverage_opts": {"type": "indicator"},
            },
        },
        "beam60-batch50-lenNorm0-cov05-covInd": {
            # {"dev-clean": 2.57, "dev-other": 5.48, "test-clean": 2.65, "test-other": 5.94}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-cov05-covInd-zeros01": {
            # {"dev-clean": 2.59, "dev-other": 5.46, "test-clean": 2.61, "test-other": 6.02}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
            },
            "data_concat_zeros": 0.1,
        },
        "beam60-batch50-lenNorm0-cov05-covInd-mono01-modAttAvg": {
            # {"dev-clean": 12.37, "dev-other": 12.49, "test-clean": 13.64, "test-other": 12.56}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator", "model_att_reduce_type": "avg"},
                "attention_monotonicity_scale": 0.1,
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-cov05-covInd-mono05-modAttAvg": {
            # {"dev-clean": 27.81, "dev-other": 25.69, "test-clean": 28.47, "test-other": 24.76}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator", "model_att_reduce_type": "avg"},
                "attention_monotonicity_scale": 0.5,
            },
        },
        "beam60-batch50-lenNorm02-cov05-covInd": {
            # {"dev-clean": 2.59, "dev-other": 5.48, "test-clean": 2.59, "test-other": 5.96}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
            },
        },
        "beam60-batch50-lenNorm02-cov1-covInd": {
            # {"dev-clean": 2.85, "dev-other": 5.97, "test-clean": 3.23, "test-other": 6.29}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 1.0,
                "attention_coverage_opts": {"type": "indicator"},
            },
        },
        "beam60-batch50-lenNorm02-cov05-covInd-negCov05_15": {
            # {"dev-clean": 3.78, "dev-other": 6.66, "test-clean": 3.95, "test-other": 7.24}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
                "neg_attention_coverage_scale": 0.5,
                "neg_attention_coverage_opts": {"type": "indicator", "threshold": 1.5},
            },
        },
        "beam60-batch50-lenNorm02-cov05-covInd-negCov04_2": {
            # {"dev-clean": 2.92, "dev-other": 5.93, "test-clean": 2.97, "test-other": 6.46}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.2,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator"},
                "neg_attention_coverage_scale": 0.4,
                "neg_attention_coverage_opts": {"type": "indicator", "threshold": 2},
            },
        },
        "beam60-batch50-lenNorm0-cov05-covInd-modAttAvg-negCovRelu05_15": {
            # {"dev-clean": 2.9, "dev-other": 5.48, "test-clean": 2.89, "test-other": 6.67}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator", "model_att_reduce_type": "avg"},
                "neg_attention_coverage_scale": 0.5,
                "neg_attention_coverage_opts": {"type": "relu_upper", "threshold": 1.5},
            },
            "__with_cheating": True,
        },
        "beam60-batch50-lenNorm0-cov05-covInd-modAttAvg": {
            # {"dev-clean": 3.93, "dev-other": 5.49, "test-clean": 2.93, "test-other": 6.51}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator", "model_att_reduce_type": "avg"},
            },
        },
        "beam60-batch50-lenNorm0-cov05_01-covInd-modAttAvg": {
            # {"dev-clean": 3.08, "dev-other": 6.64, "test-clean": 3.13, "test-other": 7.29}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "beam_search_opts": {
                "length_normalization_exponent": 0.0,
                "attention_coverage_scale": 0.5,
                "attention_coverage_opts": {"type": "indicator", "model_att_reduce_type": "avg", "threshold": 0.1},
            },
            "__with_cheating": True,
        },
        "beam60-batch1-lenNorm1": {
            # {"dev-clean": 2.89, "dev-other": 6.21, "test-clean": 2.84, "test-other": 6.58}
            "beam_size": 60,
            "max_seqs": 1,
            "length_normalization_exponent": 1.0,
        },
        "beam60-batch50-lenReward01": {
            # {"dev-clean": 2.61, "dev-other": 5.4, "test-clean": 2.82, "test-other": 6.5}
            # test-other: work/i6_core/recognition/scoring/ScliteJob.Acv12UewxtG0/output
            # Percent Substitution      =    4.4%   (2280)
            # Percent Deletions         =    0.6%   ( 292)
            # Percent Insertions        =    1.6%   ( 830)
            # work/i6_core/returnn/forward/ReturnnForwardJobV2.ALH3cRPRSWkr/output/output.py.gz
            # v5: test-other: i6_core/returnn/forward/ReturnnForwardJobV2.RL1Gfmz1ENo2
            # lots of insertions, repeating loop at end: 4294-14317-0014
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
        },
        "beam60-batch50-lenReward02": {
            # {"dev-clean": 2.84, "dev-other": 5.39, "test-clean": 2.82, "test-other": 6.49}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.2,
            "__with_cheating": True,
        },
        "beam60-batch50-lenReward005": {
            # {"dev-clean": 2.61, "dev-other": 5.41, "test-clean": 2.83, "test-other": 6.48}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.05,
        },
        "beam60-batch50-lenReward_005": {
            # {"dev-clean": 3.76, "dev-other": 5.86, "test-clean": 3.6, "test-other": 6.26}
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": -0.05,
        },
    }.items():
        aed_recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV3/recog_last_" + name,
            model.get_last_fixed_epoch(),
            aed_model_recog_pure_torch,
            {
                "beam_search_version": 5,
                "__batch_size_dependent": True,
                "__recog_def_ext": True,
                "beam_search_collect_individual_seq_scores": True,
                **recog_config,
            },
        )
