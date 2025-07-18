"""
LM CLAIX 2023 experiments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Callable, Dict, Union, Any

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

from i6_experiments.users.zhang.experiments.lm.lm_ppl import compute_ppl

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

if TYPE_CHECKING:
    from sisyphus import *
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

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

    train(
        "lm/trafo-n24-d1024-gelu-drop0-b400_20k-bpe128",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, 100, batch_size_factor=1),
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                #"max_seq_length_default_target": None,
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

    #-----------

    train(
        "lm/trafo-n12-d512-gelu-drop0-b100_10k-bpe128",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1, #Actually run on 48gb
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(10_000, 50, batch_size_factor=1),
                "max_seqs": 200,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "max_seq_length_default_target": None,
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
                    num_layers=12,
                    model_dim=512,
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

def get_trafo_lm(vocab: Bpe, num_layers: int = 24, model_dim: int = 1024,
                 max_seqs: int = 400, bs_feat: int =20_000, n_ep: int = 100, max_seq_length_default_target: bool = False, #default 75
                 dropout: float = 0.0, att_dropout: float = 0.0, epochs: list[int] = None, word_ppl: bool = False, bpe_ratio: Optional[float | tk.Variable]=None)-> Tuple[ModelWithCheckpoint, tk.path, int]:

    # from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    # from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    # from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
    #
    # bpe128 = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=128)
    # bpe128 = Bpe(dim=184, codes=bpe128.bpe_codes, vocab=bpe128.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")

    # ----test-----
    #     config_96gb_bf16_accgrad1.update(
    #     {
    #         "__gpu_mem": 12,
    #         "__cpu_rqmt": 12,  # the whole c23g node has 96 CPUs, and 4 GPUs
    #         "__mem_rqmt": 30,  # the whole node should have more than 500GB
    #         "accum_grad_multiple_step": 1,  # per single GPU
    #     }
    # )
    # -----------
    train_prefix_name = f"trafo-n{num_layers}-embd128-d{model_dim}-bpe{vocab.dim}-drop{dropout}-gelu"
    lm_dataset = get_librispeech_lm_dataset(vocab=vocab, train_epoch_split=20)

    deep_update = {
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(bs_feat=bs_feat, n_ep=n_ep, batch_size_factor=1),
                "max_seqs": max_seqs,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            }
    if max_seq_length_default_target:
        deep_update.update({"max_seq_length_default_target": None})

    model_with_checkpoints = train(  # 12.79
        f"lm/trafo-n{num_layers}-d{model_dim}-gelu-drop0-b{max_seqs}_{bs_feat//1000}k-bpe{vocab.dim}",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            deep_update
        ),
        post_config={"log_grad_norm": True},
        train_dataset=lm_dataset,
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=num_layers,
                    model_dim=model_dim,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=dropout,
                    att_dropout=att_dropout,
                )
            },
        ),
        train_def=lm_train_def,
        # For test
        # time_rqmt=2,
    )

    exponents = {184: 2.3, 10_025: 1.1} if word_ppl else {184: 1, 10_025: 1}#185-bpe128 10_025-bpe10k
    ppls = compute_ppl(
        prefix_name=train_prefix_name,
        model_with_checkpoints=model_with_checkpoints,
        dataset=lm_dataset,
        dataset_keys=["transcriptions-train", "transcriptions-test-other", "transcriptions-dev-other"],
        exponent=bpe_ratio if word_ppl else 1.0,
        epochs=epochs,
        same_seq=True,
        batch_size=10_000,
    )
    print(f"------fixed epochs of trafo_lm---------\n {model_with_checkpoints.fixed_epochs}\n--------------")
    # if ppls.
    # print(f"------PPL of ffnnlms--------")
    # for epoch, ppl in ppls.items():
    #     with open(ppl,"r") as f:
    #         ppl = f.readline()
    #     print(epoch, ppl)
    if epochs:
        for epoch in epochs:
            assert epoch in model_with_checkpoints.fixed_epochs
            yield model_with_checkpoints.get_epoch(epoch), ppls[f"epoch{epoch}"], epoch
    else:
        return model_with_checkpoints.get_last_fixed_epoch(), ppls[f"epoch{model_with_checkpoints.last_fixed_epoch_idx}"], model_with_checkpoints.last_fixed_epoch_idx
