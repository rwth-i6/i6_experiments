"""
Script to run our torch beam search implementations inside returnn with torch as engine backend
"""

from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.espnet import (
    _recog,
    model_recog_our,
    _get_orig_e_branchformer_model,
    _sis_setup_global_prefix,
    _get_orig_e_branchformer_lm_model_config,
    _get_orig_e_branchformer_lm_model_preload_opts,
)

_sis_setup_global_prefix()
model = _get_orig_e_branchformer_model()


def get_name(args):
    ignored_keys = ["__recog_def_ext", "__batch_size_dependent", "beam_search_collect_individual_seq_scores"]
    res = ""
    for k, v in sorted(args.items()):
        if k in ignored_keys:
            continue
        if isinstance(v, dict):
            res += get_name(v)
            return res
        elif isinstance(v, bool):
            if v:
                res += k + "-"
        else:
            assert isinstance(v, (int, float, str))
            res += f"{k}_{v}-"
    assert res[-1] == "-"
    res = res[:-1]
    return res


def py():
    # TODO: joint CTC
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in [5]:
            for beam in [20]:
                recog_config = dict()
                recog_config["beam_search_version"] = beam_search_variant
                recog_config["__batch_size_dependent"] = True
                recog_config["__recog_def_ext"] = True
                recog_config["beam_search_collect_individual_seq_scores"] = True

                recog_config["batch_size"] = 5000 * 160  # CTC
                recog_config["beam_search_opts"] = {
                    "beam_size": beam,
                    "ctc_weight": 0.3,
                    "max_seq_len_factor": max_seq_len_factor,
                }
                name = get_name(recog_config["beam_search_opts"])
                _recog(
                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                    model,
                    model_recog_our,
                    recog_config=recog_config,
                    vocab="spm_espnet_5k",
                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                    dev_sets=["dev-other", "test-other"],
                    audio_format="old_flac_tar_zip",
                )

                recog_config["beam_search_opts"]["lm_scale"] = 0.6
                recog_config.update(_get_orig_e_branchformer_lm_model_config())
                recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                name = get_name(recog_config["beam_search_opts"])
                _recog(
                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                    model,
                    model_recog_our,
                    recog_config=recog_config,
                    vocab="spm_espnet_5k",
                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                    dev_sets=["dev-other", "test-other"],
                    audio_format="old_flac_tar_zip",
                )

    # TODO: LM
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [20]:
                for ctc_weight in [0.3]:
                    for pruning_threshold in [5, 10, 20, 50]:
                        for adaptive in [True, False]:
                            for len_reward in [0.1]:
                                recog_config = dict()
                                recog_config["beam_search_version"] = beam_search_variant
                                recog_config["__batch_size_dependent"] = True
                                recog_config["__recog_def_ext"] = True

                                recog_config["batch_size"] = 5000 * 160  # CTC
                                recog_config["beam_search_collect_individual_seq_scores"] = True
                                recog_config["beam_search_opts"] = {
                                    "beam_size": beam,
                                    "length_reward": len_reward,
                                    "ctc_weight": ctc_weight,
                                    "max_seq_len_factor": max_seq_len_factor,
                                    "pruning_threshold": pruning_threshold,
                                    "adaptive_pruning": adaptive,
                                    "beam_ended_size": beam,
                                }
                                name = get_name(recog_config["beam_search_opts"])
                                _recog(
                                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                    model,
                                    model_recog_our,
                                    recog_config=recog_config,
                                    vocab="spm_espnet_5k",
                                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                    audio_format="old_flac_tar_zip",
                                    dev_sets=["dev-other"],
                                )
    #
    # # TODO: CTC+LM adaptive/non-adaptive
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [20]:
                for ctc_weight in [0.3]:
                    for lm_weight in [0.6]:
                        for pruning_threshold in [5, 10, 20, 50]:
                            for adaptive in [True, False]:
                                for len_reward in [0.6, 1.0]:
                                    assert lm_weight
                                    recog_config = dict()
                                    recog_config["beam_search_version"] = beam_search_variant
                                    recog_config["__batch_size_dependent"] = True
                                    recog_config["__recog_def_ext"] = True
                                    recog_config["beam_search_collect_individual_seq_scores"] = True

                                    if beam < 50:
                                        recog_config["batch_size"] = 2000 * 160  # LM + CTC
                                    else:
                                        recog_config["batch_size"] = 1000 * 160
                                    recog_config["max_seqs"] = 20

                                    recog_config["beam_search_opts"] = {
                                        "beam_size": beam,
                                        "length_reward": len_reward,
                                        "max_seq_len_factor": max_seq_len_factor,
                                        "ctc_weight": ctc_weight,
                                        "lm_scale": lm_weight,
                                        "pruning_threshold": pruning_threshold,
                                        "adaptive_pruning": adaptive,
                                        "beam_ended_size": beam,
                                    }
                                    recog_config.update(_get_orig_e_branchformer_lm_model_config())
                                    recog_config[
                                        "preload_from_files"
                                    ] = _get_orig_e_branchformer_lm_model_preload_opts()

                                    name = get_name(recog_config["beam_search_opts"])
                                    _recog(
                                        f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                        model,
                                        model_recog_our,
                                        recog_config=recog_config,
                                        vocab="spm_espnet_5k",
                                        audio_opts={
                                            "peak_normalization": False
                                        },  # speech_volume_normalize=False in ESPnet
                                        dev_sets=["dev-other"],
                                        audio_format="old_flac_tar_zip",
                                    )
