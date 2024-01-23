"""
Attention-based encoder-decoder (AED) experiments, using ESPnet models
"""

from __future__ import annotations

import os
import copy
import sys
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, List

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef, ModelDefWithCfg
from i6_experiments.users.zeyer.accum_grad_schedules.piecewise_linear import dyn_accum_grad_piecewise_linear

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.zeyer.datasets.task import Task
    from espnet2.asr.espnet_model import ESPnetASRModel

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix(prefix_name)

    train_exp(
        "v6-24gb-bs30k-wd1e_6-lrlin1e_5_587k-EBranchformer",
        config_24gb_v6,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(30_000, 2000),
        },
    )

    # uncomment this to get the CUDA OOM error in dist.all_reduce: https://github.com/rwth-i6/returnn/issues/1482
    # train_exp(
    #     "v6-11gb-f32-bs8k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-ncclError",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
    #         "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
    #     },
    # )

    train_exp(
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": _dyn_accum_grad_multiple_step_v2,
        },
    )

    # TODO also try model average

    train_exp(
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV1a",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": _dyn_accum_grad_multiple_step_v1a,
        },
    )

    train_exp(
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_2-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )

    train_exp(
        "v6-11gb-f32-bs8k-mgpu2-nep500-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            "__num_processes": 2,
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )

    train_exp(
        "v6-11gb-f32-bs8k-nep500-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            "__num_processes": None,
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
        config_deletes=["__num_processes", "torch_distributed"],
    )

    train_exp(
        "v6-11gb-f32-bs8k-mgpu4-nep250-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 250),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000 // 2, 100_000 // 2, 1_100_000 // 2, 1_242_000 // 2],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )

    train_exp(
        "v6-11gb-f32-bs8k-mgpu4-nep125-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 125),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000 // 4, 100_000 // 4, 1_100_000 // 4, 1_242_000 // 4],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from .sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    model_config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    with_eos_postfix: bool = False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ls_task(with_eos_postfix=with_eos_postfix)
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=ModelDefWithCfg(from_scratch_model_def, model_config),
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(
        prefix,
        task,
        model_with_checkpoint,
        recog_def=model_recog,
        search_config={"search_version": 4, "num_epochs": num_epochs},
    )

    return model_with_checkpoint


_ls_task: Dict[bool, Task] = {}  # with_eos_postfix -> Task


def _get_ls_task(*, with_eos_postfix: bool = False):
    if with_eos_postfix in _ls_task:
        return _ls_task[with_eos_postfix]

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    _ls_task[with_eos_postfix] = get_librispeech_task_bpe10k_raw(with_eos_postfix=with_eos_postfix)
    return _ls_task[with_eos_postfix]


py = sis_run_with_prefix  # if run directly via `sis m ...`

_batch_size_factor = 160


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def _dyn_accum_grad_multiple_step(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    if global_train_step <= 10_000:
        return 4
    if global_train_step <= 50_000:
        return 2
    return 1


def _dyn_accum_grad_multiple_step_v1a(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    if global_train_step <= 20_000:
        return 4
    if global_train_step <= 100_000:
        return 2
    return 1


def _dyn_accum_grad_multiple_step_v2(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    # Schedule:
    # start low (to get from random init somewhere more sensible fast),
    # increase to almost 100 (to get it to convergence),
    # decay again (to get faster convergence),
    # and then maybe at the very end increase again (for finetuning).
    # Assume ~1.242k steps in total.

    steps = [50_000, 100_000, 1_100_000, 1_242_000]
    values = [1, 100, 1, 1, 10]
    assert len(steps) + 1 == len(values)

    last_step = 0
    for i, step in enumerate(steps):
        assert step > last_step
        assert global_train_step >= last_step
        if global_train_step < step:
            factor = (global_train_step + 1 - last_step) / (step - last_step)
            return int(values[i + 1] * factor + values[i] * (1 - factor))
        last_step = step

    return values[-1]


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> ESPnetASRModel:
    """Function is run within RETURNN."""
    import returnn
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config(return_empty_if_none=True)  # noqa

    # Load some train yaml file for model def.
    # References:
    # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/run.sh
    # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh
    # https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_train.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/asr.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/abs_task.py

    tools_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(returnn.__file__))))
    print("tools dir:", tools_dir)
    sys.path.append(tools_dir + "/espnet")

    import espnet2

    espnet_repo_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(espnet2.__file__)))

    from espnet2.tasks.asr import ASRTask
    from espnet2.asr.espnet_model import ESPnetASRModel

    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    espnet_config_file = config.value("espnet_config", None)
    assert espnet_config_file
    parser = ASRTask.get_parser()
    args = parser.parse_args(["--config", espnet_repo_root_dir + "/" + espnet_config_file])
    args.token_list = target_dim.vocab.labels
    assert config.bool("espnet_fixed_sos_eos", False)
    args.model_conf["sym_sos"] = target_dim.vocab.labels[_get_bos_idx(target_dim)]
    args.model_conf["sym_eos"] = target_dim.vocab.labels[_get_eos_idx(target_dim)]

    # TODO any of these relevant?
    #             --use_preprocessor true \
    #             --bpemodel "${bpemodel}" \
    #             --token_type "${token_type}" \
    #             --token_list "${token_list}" \
    #             --non_linguistic_symbols "${nlsyms_txt}" \
    #             --cleaner "${cleaner}" \
    #             --g2p "${g2p}" \
    #             --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
    #             --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
    #             --resume true \
    #             ${pretrained_model:+--init_param $pretrained_model} \
    #             --ignore_init_mismatch ${ignore_init_mismatch} \
    #             --fold_length "${_fold_length}" \

    model = ASRTask.build_model(args)
    assert isinstance(model, ESPnetASRModel)
    print("Target dim:", target_dim)
    print("Vocab size:", model.vocab_size)
    print("Vocab:", target_dim.vocab.labels[:5], "...", target_dim.vocab.labels[-5:])
    print("Ignore:", model.ignore_id)
    print("Blank:", model.blank_id)
    print("SOS/EOS:", model.sos, model.eos)
    model.returnn_epoch = epoch
    return model


from_scratch_model_def: ModelDef[ESPnetASRModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: ESPnetASRModel, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    import torch
    import returnn.frontend as rf

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    loss, stats, weight = model(
        speech=data.raw_tensor,
        speech_lengths=data_spatial_dim.dyn_size,
        text=targets.raw_tensor.to(torch.int64),
        text_lengths=targets_spatial_dim.dyn_size,
    )
    # TODO the following is correct for CE and CTC, but not correct for CER and probably others, need to check...
    # ESPnet usually does divide the loss by num seqs (batch dim) but not by seq length.
    custom_inv_norm_factor = targets_spatial_dim.get_size_tensor()
    custom_inv_norm_factor = rf.cast(custom_inv_norm_factor, "float32")
    batch_dim_value = custom_inv_norm_factor.dims[0].get_dim_value_tensor()
    custom_inv_norm_factor /= rf.cast(batch_dim_value, "float32")
    rf.get_run_ctx().mark_as_loss(loss, "total", custom_inv_norm_factor=custom_inv_norm_factor)
    for k, v in stats.items():
        if v is not None:
            rf.get_run_ctx().mark_as_loss(v, k, as_error=True)


from_scratch_training: TrainDef[ESPnetASRModel]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(
    *,
    model: ESPnetASRModel,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from returnn.config import get_global_config

    config = get_global_config()
    search_version = config.int("search_version", 0)
    assert search_version >= 3, f"search version {search_version} unsupported, likely there was a bug earlier..."
    # version 3 was setting RETURNN_FIX_BLANK to have ESPnet blank fixed.
    #   But now this has been merged in ESPnet. https://github.com/espnet/espnet/pull/5620
    #   We maybe should check for the right ESPnet version... Look out for unusual long recognized seqs.

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    # References:
    # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/run.sh
    # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh
    # https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_inference.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/asr.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/abs_task.py

    # decode_asr.yaml:
    # beam_size: 60
    # ctc_weight: 0.3
    # lm_weight: 0.6

    beam_size = config.int("beam_size", 12)  # like RETURNN, not 60 for now...
    ctc_weight = 0.3
    lm_weight = 0.6  # not used currently...
    ngram_weight = 0.9  # not used currently...
    penalty = 0.0
    normalize_length = False
    maxlenratio = config.float("maxlenratio", 0.0)
    minlenratio = 0.0

    # Partly taking code from espnet2.bin.asr_inference.Speech2Text.

    import torch
    from espnet.nets.scorers.ctc import CTCPrefixScorer
    from espnet.nets.scorers.length_bonus import LengthBonus
    from espnet.nets.scorer_interface import BatchScorerInterface
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.beam_search import Hypothesis

    scorers = {}
    asr_model = model
    decoder = asr_model.decoder

    ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
    token_list = asr_model.token_list
    scorers.update(
        decoder=decoder,
        ctc=ctc,
        length_bonus=LengthBonus(len(token_list)),
    )

    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        ngram=ngram_weight,
        length_bonus=penalty,
    )

    assert all(isinstance(v, BatchScorerInterface) for k, v in scorers.items()), f"non-batch scorers: {scorers}"

    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=asr_model.sos,
        eos=asr_model.eos,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        normalize_length=normalize_length,
    )

    speech = data.raw_tensor  # [B, Nsamples]
    print("Speech shape:", speech.shape, "device:", speech.device)
    lengths = data_spatial_dim.dyn_size  # [B]
    batch = {"speech": speech, "speech_lengths": lengths}
    logging.info("speech length: " + str(speech.size(1)))

    # Encoder forward (batched)
    enc, enc_olens = asr_model.encode(**batch)
    print("Encoded shape:", enc.shape, "device:", enc.device)

    batch_dim = data.dims[0]
    batch_size = speech.size(0)
    beam_dim = Dim(beam_size, name="beam")
    olens = torch.zeros([batch_size, beam_size], dtype=torch.int32)
    out_spatial_dim = Dim(Tensor("out_spatial", [batch_dim, beam_dim], "int32", raw_tensor=olens))
    outputs = [[] for _ in range(batch_size)]
    oscores = torch.zeros([batch_size, beam_size], dtype=torch.float32)
    seq_log_prob = Tensor("scores", [batch_dim, beam_dim], "float32", raw_tensor=oscores)

    # BatchBeamSearch is misleading: It still only operates on a single sequence,
    # but just handles all hypotheses in a batched way.
    # So we must iterate over all the sequences here from the input.
    for i in range(batch_size):
        nbest_hyps: List[Hypothesis] = beam_search(
            x=enc[i, : enc_olens[i]], maxlenratio=maxlenratio, minlenratio=minlenratio
        )
        print("best:", " ".join(token_list[v] for v in nbest_hyps[0].yseq))
        # I'm not exactly sure why, but sometimes we get even more hyps?
        # And then also sometimes, we get less hyps?
        very_bad_score = min(-1e32, nbest_hyps[-1].score - 1)  # not -inf because of serialization issues
        while len(nbest_hyps) < beam_size:
            nbest_hyps.append(Hypothesis(score=very_bad_score, yseq=torch.zeros(0, dtype=torch.int32)))
        for j in range(beam_size):
            hyp: Hypothesis = nbest_hyps[j]
            olens[i, j] = hyp.yseq.size(0)
            outputs[i].append(hyp.yseq)
            oscores[i, j] = hyp.score

    outputs_t = torch.zeros([batch_size, beam_size, torch.max(olens)], dtype=torch.int32)
    for i in range(batch_size):
        for j in range(beam_size):
            outputs_t[i, j, : olens[i, j]] = outputs[i][j]
    seq_targets = Tensor("outputs", [batch_dim, beam_dim, out_spatial_dim], "int32", raw_tensor=outputs_t)

    from returnn.datasets.util.vocabulary import Vocabulary

    target_dim = Dim(name="target", dimension=len(token_list), kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(token_list, eos_label=model.eos)
    seq_targets.sparse_dim = target_dim

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[ESPnetASRModel]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<s>"
model_recog.batch_size_dependent = False
