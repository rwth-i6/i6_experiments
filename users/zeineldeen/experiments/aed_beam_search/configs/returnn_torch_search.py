"""
Script to run our torch beam search implementations inside returnn with torch as engine backend
"""

from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.espnet import (
    _recog,
    model_recog_our,
    _get_orig_e_branchformer_model,
    _sis_setup_global_prefix,
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
            for beam in [20, 60]:
                for ctc_weight in [0.3]:
                    for len_reward in [0.0, 0.1, 0.2]:
                        recog_config = dict()
                        recog_config["beam_search_version"] = beam_search_variant
                        recog_config["__batch_size_dependent"] = True
                        recog_config["batch_size"] = 5_000 * 160
                        recog_config["beam_search_collect_individual_seq_scores"] = True
                        recog_config["beam_search_opts"] = {
                            "beam_size": beam,
                            "length_reward": len_reward,
                            "length_normalization_exponent": 0.0,
                            "ctc_weight": ctc_weight,
                            "max_seq_len_factor": max_seq_len_factor,
                        }
                        name = get_name(recog_config["beam_search_opts"])
                        _recog(
                            f"e_branchformer_raw_en_bpe5000_sp/{beam_search_variant}/{name}",
                            model,
                            model_recog_our,
                            recog_config=recog_config,
                            vocab="spm_espnet_5k",
                            audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                        )

                        recog_config["length_normalization_exponent"] = 1.0
                        recog_config.pop("length_reward", None)
                        name = get_name(recog_config["beam_search_opts"])
                        _recog(
                            f"e_branchformer_raw_en_bpe5000_sp/{beam_search_variant}/{name}",
                            model,
                            model_recog_our,
                            recog_config=recog_config,
                            vocab="spm_espnet_5k",
                            audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                            dev_sets=["dev-other"],
                        )

    for max_seq_len_factor in [0.5]:
        for beam_search_variant in [5]:
            for beam in [20, 60]:
                for ctc_weight in [0.0, 0.3]:
                    for lm_weight in [0.1, 0.14]:
                        for len_reward in [0.1, 0.6]:
                            recog_config = dict()
                            recog_config["beam_search_version"] = beam_search_variant
                            recog_config["__batch_size_dependent"] = True
                            recog_config["batch_size"] = 5_000 * 160
                            recog_config["beam_search_collect_individual_seq_scores"] = True
                            recog_config["beam_search_opts"] = {
                                "beam_size": beam,
                                "length_reward": len_reward,
                                "length_normalization_exponent": 0.0,
                                "ctc_weight": ctc_weight,
                                "max_seq_len_factor": max_seq_len_factor,
                                "lm_weight": lm_weight,
                            }
                            name = get_name(recog_config["beam_search_opts"])
                            _recog(
                                f"e_branchformer_raw_en_bpe5000_sp/{beam_search_variant}/{name}",
                                model,
                                model_recog_our,
                                recog_config=recog_config,
                                vocab="spm_espnet_5k",
                                audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                dev_sets=["dev-other"],
                            )
