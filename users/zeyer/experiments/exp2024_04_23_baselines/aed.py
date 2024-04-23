from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep


def py():
    train_exp(  # TODO (should give 5.11)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
        config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )
