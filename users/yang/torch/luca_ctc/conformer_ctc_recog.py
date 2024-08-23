from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
from itertools import product
import copy

from sisyphus import tk

from returnn.tensor import Tensor

from i6_core.returnn.training import PtCheckpoint
from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog

from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy
from i6_experiments.users.yang.torch.decoding.ctc_label_sync import model_recog_label_sync
from i6_experiments.users.yang.torch.decoding.recog import recog_model
from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc import from_scratch_model_def
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.aed_new import aed_model_def





_sis_prefix: Optional[str] = None

default_search_args = {
    "beam_size": 12,
}

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def get_model_ckpt(model_name, model_path, model_def=from_scratch_model_def):
    new_ckpt_path = tk.Path(model_path,
                            hash_overwrite=model_name + '_torch_ckpt')
    new_ckpt = PtCheckpoint(new_ckpt_path)
    model_ckpt = ModelWithCheckpoint(definition=model_def, checkpoint=new_ckpt)
    return model_ckpt





def sis_run_with_prefix(prefix_name: Optional[str] = None):

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
    task = get_librispeech_task_raw_v2(vocab="bpe10k")
    if _sis_prefix is None:
        _sis_setup_global_prefix()
    model_names = []
    models_with_pt_ckpt = {}

    model_name = "ctc_baseline"
    # load checkpoints:
    # model trained by Luca, eos not used in training,
    # {"dev-clean": 3.02, "dev-other": 6.8, "test-clean": 3.16, "test-other": 7.07},
    # verify the decoding result: bsf=160: the same result
    model_path = "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"
    new_ckpt_path = tk.Path(
        model_path,
        hash_overwrite=model_name + "_torch_ckpt",
    )
    new_ckpt = PtCheckpoint(new_ckpt_path)
    ctc_model_ckpt = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt)
    # bsf = 11
    #
    # search_args = {
    #     "bsf": bsf,
    #     "hash_overwrite": "debug",
    #     "beam_size": 1,
    #     "mask_eos_output": False,
    # }
    # name = _sis_prefix + '/' + f"luca_noeos_ctc_greedy_baseline_debug_no_eos_mask_bsf{bsf}"
    # res, _ = recog_model(
    #     task,
    #     ctc_model_ckpt,
    #     model_recog_greedy,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],  # set to None for all
    #     model_args={},
    #     search_args=search_args,
    #     prefix_name=name,
    # )
    # tk.register_output(
    #     name + f"/recog_results",
    #     res.output,
    # )
    #
    # # Luca ctc layer 8 decoding
    # model_args = {
    #     "ctc_output_args":{
    #         "ctc_enc_layer_id": 8,
    #     }
    # }
    # search_args = {
    #     "bsf": bsf,
    #     "hash_overwrite": "debug_ctc_layer8",
    #     "beam_size": 1,
    #     "mask_eos_output": False,
    # }
    # name = _sis_prefix + '/' + f"luca_noeos_ctc_layer8_greedy_baseline_debug_no_eos_mask_bsf{bsf}"
    # res, _ = recog_model(
    #     task,
    #     ctc_model_ckpt,
    #     model_recog_greedy,
    #     dev_sets=["dev-other"],  # set to None for all
    #     model_args=model_args,
    #     search_args=search_args,
    #     prefix_name=name,
    # )
    # tk.register_output(
    #     name + f"/recog_results",
    #     res.output,
    # )
    #label sync search, test without lm first


    # bsf = 11
    # name = _sis_prefix + '/' + f"luca_noeos_ctc_label_sync_baseline_debug_no_eos_mask_bsf{bsf}"
    # search_args = {
    #     "bsf": bsf,
    #     "hash_overwrite": "label_sync_debug",
    #     "beam_size": 24,
    # }
    # res, _ = recog_model(
    #     task,
    #     ctc_model_ckpt,
    #     model_recog_label_sync,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],  # set to None for all
    #     model_args={},
    #     search_args=search_args,
    # )
    # tk.register_output(
    #     name + f"/recog_results",
    #     res.output,
    # )

    # label sync search with lstm lm
    # bsf = 11
    # #lm_scales = [0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.8]
    # # best scale 0.65, dev other 5.07
    # lm_scales = [0.65]
    # for lm_scale in lm_scales:
    #     name = _sis_prefix + '/' + f"luca_noeos_ctc_label_sync_baseline_debug_no_eos_mask_bsf{bsf}_lm_scale{lm_scale}_lennorm1.0"
    #     search_args = {
    #         "bsf": bsf,
    #         "hash_overwrite": "label_sync_debug",
    #         "beam_size": 48,
    #         "lm_scale": lm_scale,
    #         "length_norm_scale": 1.0,
    #     }
    #     res, _ = recog_model(
    #         task,
    #         ctc_model_ckpt,
    #         model_recog_label_sync,
    #         dev_sets=[ "dev-other"],  # set to None for all
    #         model_args={},
    #         search_args=search_args,
    #     )
    #     tk.register_output(
    #         name + f"/recog_results",
    #         res.output,
    #     )

    ############## LM linear combination in prob space but not log space
    bsf = 11
    #lm_scales = [0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.8]
    # best scale 0.65, dev other 5.07
    lm_scales = [0.6,1.0]
    lm_linear_combine_scales = [0.4,0.5,0.6,0.7]
    # combine scale 0.4 -> dev other 6.7, at least not worse
    # for lm_linear_combine_scale in lm_linear_combine_scales:
    #     for lm_scale in lm_scales:
    #         name = _sis_prefix + '/' + f"luca_noeos_ctc_label_sync_baseline_debug_no_eos_mask_bsf{bsf}_lmlinear_scale{lm_scale}_combine_scale{lm_linear_combine_scale}_lennorm1.0"
    #         search_args = {
    #             "bsf": bsf,
    #             "hash_overwrite": "label_sync_debug",
    #             "beam_size": 48,
    #             "lm_scale": lm_scale,
    #             "length_norm_scale": 1.0,
    #             "lm_liner_combine_scale": lm_linear_combine_scale,
    #             "lm_linear_combine": True,
    #         }
    #         res, _ = recog_model(
    #             task,
    #             ctc_model_ckpt,
    #             model_recog_label_sync,
    #             dev_sets=[ "dev-other"],  # set to None for all
    #             model_args={},
    #             search_args=search_args,
    #             prefix_name=name,
    #         )
    #         tk.register_output(
    #             name + f"/recog_results",
    #             res.output,
    #         )


    # load checkpoints:
    # model_base_path = "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.PbIzhk86lA8B/output/models/"
    # bsf = 10
    # epochs = ["050", "005", "008"]
    # for epoch in epochs:
    #
    #     model_name = f"kd_scale-0.4_layer4_lm_scale-0.6-bsf{bsf}epoch{epoch}"
    #     model_path = model_base_path + f"epoch.{epoch}.pt"
    #     new_ckpt_path = tk.Path(
    #         model_path,
    #         hash_overwrite=model_name + "_torch_ckpt",
    #     )
    #     new_ckpt = PtCheckpoint(new_ckpt_path)
    #     ctc_model_ckpt = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_ckpt)
    #
    #
    #     search_args = {
    #         "bsf": bsf,
    #         "hash_overwrite": "debug",
    #         "beam_size": 1,
    #         "mask_eos_output": False,
    #     }
    #     name = _sis_prefix + '/' + model_name
    #     res, _ = recog_model(
    #         task,
    #         ctc_model_ckpt,
    #         model_recog_greedy,
    #         dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],  # set to None for all
    #         model_args={},
    #         search_args=search_args,
    #         prefix_name=name,
    #     )
    #     tk.register_output(
    #         name + f"/recog_results",
    #         res.output,
    #     )
    #
    # ### lm kd, model: eosmask_kd_scale-0.2-trainlm-1.0-top-20_lr1-1e-06_lr2-1e-05_lr3-1e-06_ep1-28_ep2-2_ep3-30
    # model_base_path = "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.Zh1t4GvlSqnD/output/models/"
    # model_name_base = "eosmask_kd_scale-0.2-trainlm-1.0-top-20_lr1-1e-06_lr2-1e-05_lr3-1e-06_ep1-28_ep2-2_ep3-30"
    #
    # epochs = ["005", "008", "010", "020", "040", "050"]
    #
    # for epoch in epochs:
    #     model_name = model_name_base + f"epoch{epoch}"
    #     model_path = model_base_path + f"epoch.{epoch}.pt"
    #     model_ckpt = get_model_ckpt(model_name, model_path)
    #     name = _sis_prefix + '/' + model_name
    #     search_args = {
    #         "bsf": bsf,
    #         "hash_overwrite": "kd_trained_label_sync",
    #         "beam_size": 48,
    #         #"length_norm_scale": 1.0,
    #     }
    #     res, _ = recog_model(
    #         task,
    #         model_ckpt,
    #         model_recog_label_sync,
    #         dev_sets=["dev-other"],  # set to None for all
    #         model_args={},
    #         search_args=search_args,
    #         prefix_name=name,
    #     )
    #     tk.register_output(
    #         _sis_prefix + '/' + model_name_base + 'label_sync' + f'/{epoch}',
    #         res.output,
    #     )


    ### Albert's aed model， finetuned, ctc layer 12 greedy search
    #
    # model_path = "/work/asr4/zyang/torch_checkpoints/aed/albert_bpe10k/epoch.489.pt"
    # model_name = "albert-aed-bpe10k-ctc12"
    # model_ckpt = get_model_ckpt(model_name, model_path, model_def=aed_model_def)
    # search_args = {
    #     "bsf": bsf,
    #     "hash_overwrite": "albert-aed-ctc-8",
    #     "beam_size": 1,
    #     "mask_eos_output": False,
    # }
    # name = model_name + 'ctc-8-greedy'
    # res, _ = recog_model(
    #     task,
    #     model_ckpt,
    #     model_recog_greedy,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],  # set to None for all
    #     model_args={},
    #     search_args=search_args,
    #     prefix_name=name,
    # )
    # tk.register_output(
    #     _sis_prefix + '/' + name,
    #     res.output,
    # )

    ### Albert's aed model, ctc layer 8 greedy search

    # model_path = "/work/asr4/zyang/torch_checkpoints/aed/albert_bpe10k/epoch.489.pt"
    # model_name = "albert-aed-bpe10k"
    # model_ckpt = get_model_ckpt(model_name, model_path, model_def=aed_model_def)
    # search_args = {
    #     "bsf": bsf,
    #     "hash_overwrite": "albert-aed-ctc-8",
    #     "beam_size": 1,
    #     "mask_eos_output": False,
    # }
    # name = model_name + 'ctc-8-greedy'
    # res, _ = recog_model(
    #     task,
    #     model_ckpt,
    #     model_recog_greedy,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],  # set to None for all
    #     model_args={},
    #     search_args=search_args,
    #     prefix_name=name,
    # )
    # tk.register_output(
    #     _sis_prefix + '/' + name,
    #     res.output,
    # )


    # Albert's aed model, fine-tuned for ctc-layer 12 greedy search

    # 12 layer bad: {"dev-clean": 4.35, "dev-other": 9.78, "test-clean": 4.54, "test-other": 9.75}
    # model_path = "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.vAf8dtyKa2gG/output/models/epoch.100.pt"
    # model_name = "albert-aed-bpe10k-ctc12"
    # model_ckpt = get_model_ckpt(model_name, model_path, model_def=aed_model_def)
    # search_args = {
    #     "bsf": bsf,
    #     "hash_overwrite": "albert-aed-finetune-ctc-8",
    #     "beam_size": 1,
    #     "mask_eos_output": False,
    # }
    # model_args = {
    #     "ctc_output_args":{
    #         "ctc_enc_layer_id": 8,
    #     }
    # }
    # name = model_name + '-finetune100-sample0001-ctc-8-greedy'
    # res, _ = recog_model(
    #     task,
    #     model_ckpt,
    #     model_recog_greedy,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],  # set to None for all
    #     model_args=model_args,
    #     search_args=search_args,
    #     prefix_name=name,
    # )
    # tk.register_output(
    #     _sis_prefix + '/' + name,
    #     res.output,
    # )
    #
    #
    # ### aed fine-tune model with ctc 12 scale 1.0, others 0.1
    # model_path = "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.egjEiJJCFAti/output/models/epoch.100.pt"
    # model_name = "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100--speedpertV2-bpe10k-bpeSample001-finetune-ctc12-scale1.0-others0.1"
    # model_ckpt = get_model_ckpt(model_name, model_path, model_def=aed_model_def)
    # search_args = {
    #     "bsf": bsf,
    #     #"hash_overwrite": "albert-aed-finetune-ctc1.0-12",
    #     "beam_size": 1,
    #     "mask_eos_output": False,
    # }
    # model_args = {
    #     "ctc_output_args": {
    #         "ctc_enc_layer_id": 12,
    #     }
    # }
    # name = model_name + '-ctc-12-greedy'
    # res, _ = recog_model(
    #     task,
    #     model_ckpt,
    #     model_recog_greedy,
    #     dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
    #     model_args=model_args,
    #     search_args=search_args,
    #     prefix_name=name,
    # )
    # tk.register_output(
    #     _sis_prefix + '/' + name,
    #     res.output,
    # )


    ####

    ### aed from scratch, ctc loss 4,8,12
    # epoch 482, layer 12
    # aed result: {"best_scores": {"dev-clean": 2.21, "dev-other": 5.15, "test-clean": 2.49, "test-other": 5.48}, "best_epoch": 482}
    # bsf 10: ctc-12 greedy: {"dev-clean": 2.87, "dev-other": 7.42, "test-clean": 3.16, "test-other": 7.48}
    # bsf 120 ctc-12 greedy: {"dev-clean": 2.82, "dev-other": 7.43, "test-clean": 3.13, "test-other": 7.42}, almost the same
    # bsf 10 ctc-8 greedy: {"dev-clean": 3.59, "dev-other": 9.01, "test-clean": 3.86, "test-other": 9.14}

    bsfs = [10]
    ctc_layers = [8,12]

    model_path = "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.V18BvuJ52QAA/output/models/epoch.482.pt"
    model_name = "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100--speedpertV2-bpe10k-bpeSample001-from_scratch-ctc12"
    model_ckpt = get_model_ckpt(model_name, model_path, model_def=aed_model_def)
    for bsf in bsfs:
        for ctc_layer in ctc_layers:
            search_args = {
                "bsf": bsf,
                #"hash_overwrite": "albert-aed-finetune-ctc1.0-12",
                "beam_size": 1,
                "mask_eos_output": False,
            }
            model_args = {
                "ctc_output_args": {
                    "ctc_enc_layer_id": ctc_layer,
                }
            }
            name = model_name + f'-ctc-{ctc_layer}-greedy-bsf{bsf}'
            res, _ = recog_model(
                task,
                model_ckpt,
                model_recog_greedy,
                dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
                model_args=model_args,
                search_args=search_args,
            )
            tk.register_output(
                _sis_prefix + '/' + name,
                res.output,
            )
    for bsf in bsfs:
        for ctc_layer in ctc_layers:
            search_args = {
                "bsf": bsf,
                #"hash_overwrite": "albert-aed-finetune-ctc1.0-12",
                "beam_size": 1,
                "mask_eos_output": False,
            }
            model_args = {
                "ctc_output_args": {
                    "ctc_enc_layer_id": ctc_layer,
                },
                "feature_batch_norm": True,
            }
            name = model_name + f'-ctc-{ctc_layer}-greedy-bsf{bsf}-feature-batch-norm'
            res, _ = recog_model(
                task,
                model_ckpt,
                model_recog_greedy,
                dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
                model_args=model_args,
                search_args=search_args,
            )
            tk.register_output(
                _sis_prefix + '/' + name,
                res.output,
            )

    # albert's sampling 001 aed model, check if sampling influences the ctc result

    # v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenNone-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-bpe10k-bpeSample001-ctc-8-greedy-bsf10
    # {"dev-clean": 3.42, "dev-other": 9.07, "test-clean": 3.62, "test-other": 8.71}
    # sampling hurts ctc performance? or it's just not trained enough?
    ctc_layers = [8]

    bsfs = [10]

    model_path = "/work/asr4/zyang/torch_checkpoints/aed/albert_bpe10k_sampling001/epoch.487.pt"
    model_name = "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenNone-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-bpe10k-bpeSample001"
    model_ckpt = get_model_ckpt(model_name, model_path, model_def=aed_model_def)
    for bsf in bsfs:
        for ctc_layer in ctc_layers:
            search_args = {
                "bsf": bsf,
                #"hash_overwrite": "albert-aed-finetune-ctc1.0-12",
                "beam_size": 1,
                "mask_eos_output": False,
            }
            model_args = {
                "ctc_output_args": {
                    "ctc_enc_layer_id": ctc_layer,
                }
            }
            name = model_name + f'-ctc-{ctc_layer}-greedy-bsf{bsf}'
            res, _ = recog_model(
                task,
                model_ckpt,
                model_recog_greedy,
                dev_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
                model_args=model_args,
                search_args=search_args,
            )
            tk.register_output(
                _sis_prefix + '/' + name,
                res.output,
            )








py = sis_run_with_prefix


