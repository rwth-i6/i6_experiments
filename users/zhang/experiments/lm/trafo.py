"""
LM CLAIX 2023 experiments
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_train_def, lm_model_def

from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zhang.train_v4 import train, ModelDefWithCfg

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder


def py():
    from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str

    bpe128 = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=128)
    bpe128 = Bpe(dim=184, codes=bpe128.bpe_codes, vocab=bpe128.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")

    #----test-----
#     config_96gb_bf16_accgrad1.update(
#     {
#         "__gpu_mem": 12,
#         "__cpu_rqmt": 12,  # the whole c23g node has 96 CPUs, and 4 GPUs
#         "__mem_rqmt": 30,  # the whole node should have more than 500GB
#         "accum_grad_multiple_step": 1,  # per single GPU
#     }
# )
    #-----------

    train(  # 12.79
        "lm/trafo-n24-d1024-gelu-drop0-b400_20k-bpe128",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, 100, batch_size_factor=1),
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        post_config={"log_grad_norm": True},
        train_dataset=get_librispeech_lm_dataset(vocab=bpe128, train_epoch_split=20),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=1024,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
        #For test
        #time_rqmt=2,
    )
    # # Get the bpe1k vocab exactly as some others from our group (Mohammad, Robin, ...).
    # bpe1k = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=1000)
    # bpe1k = Bpe(dim=1056, codes=bpe1k.bpe_codes, vocab=bpe1k.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")
    # assert bpe1k.codes.creator.job_id() == "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV"

    # train(  # 12.79
    #     "lm/trafo-n32-d1024-gelu-drop0-b400_20k-bpe1k",
    #     config=dict_update_deep(
    #         config_96gb_bf16_accgrad1,
    #         {
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, 100, batch_size_factor=1),
    #             "max_seqs": 400,
    #             "optimizer.weight_decay": 1e-2,
    #             "calculate_exp_loss": True,
    #         },
    #     ),
    #     post_config={"log_grad_norm": True},
    #     train_dataset=get_librispeech_lm_dataset(vocab=bpe1k, train_epoch_split=20),
    #     model_def=ModelDefWithCfg(
    #         lm_model_def,
    #         {
    #             "_model_def_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 encoder_dim=None,
    #                 num_layers=32,
    #                 model_dim=1024,
    #                 ff_activation=rf.build_dict(rf.gelu),
    #                 dropout=0.0,
    #                 att_dropout=0.0,
    #             )
    #         },
    #     ),
    #     train_def=lm_train_def,
    # )