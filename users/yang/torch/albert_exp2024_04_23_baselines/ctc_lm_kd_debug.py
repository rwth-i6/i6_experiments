"""
CTC experiments.
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence
import torch
from torch import nn

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import log_ctc_pref_beam_scores

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
    from i6_experiments.users.zeyer.datasets.task import Task


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def py():
    for vocab in [
    #    "spm20k",
        "bpe10k",  # 8.23, best epoch 482, checkpoint /work/asr4/zeyer/setups-data/combined/2021-05-31/work/i6_core/returnn/training/ReturnnTrainingJob.6k24VqNUdOqz/output/models/epoch.482.pt
    #    "spm10k",  # 8.12
    #    "spm_bpe10k",  # 7.97
    #    "spm4k",  # 9.86
    #    "spm1k",
    #    "spm_bpe1k",  # 11.76
    ]:
        train_exp(  # 8.23
            #f"v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-{vocab}",
            'debug_run_lm',
            #config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            debug_config,
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [1.1],#[0.7, 0.8, 0.9, 1.0, 1.1],
            },
            vocab=vocab,
        )

    # for alpha in [
    #     0.3,  # 7.88
    # #    0.5,
    # #    0.7,
    # ]:
    #     train_exp(
    #         "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-spm10k"
    #         f"-spmSample{str(alpha).replace('.', '')}",
    #         config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
    #             "optimizer.weight_decay": 1e-2,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         },
    #         vocab="spm10k",
    #         train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": alpha}},
    #     )
    #
    # for alpha in [
    #     0.3,
    # ]:
    #     train_exp(
    #         "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-spm_bpe10k"
    #         f"-spmSample{str(alpha).replace('.', '')}",
    #         config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
    #             "optimizer.weight_decay": 1e-2,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         },
    #         vocab="spm_bpe10k",
    #         train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": alpha}},
    #     )


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.zeyer.train_v3 import train
    from i6_experiments.users.zeyer.recog import recog_training_exp
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = get_librispeech_task_raw_v2(vocab=vocab, train_vocab_opts=train_vocab_opts) # original and correct version
    #task = get_librispeech_task_raw_v2(vocab=vocab, train_vocab_opts=train_vocab_opts,with_eos_postfix=True)
    # from i6_experiments.users.zeyer.datasets.librispeech import (
    #     get_librispeech_task_bpe10k_raw,
    # )
    # task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)
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

    if not model_def:
        model_def = ctc_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = ctc_training
    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=model_def,
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else None,
        time_rqmt=time_rqmt,
    )
    recog_training_exp(prefix, task, model_with_checkpoint, recog_def=model_recog) # recognition

    return model_with_checkpoint


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
        enc_att_num_heads=8,
        enc_conformer_layer_opts=dict(
            conv_norm_opts=dict(use_mask=True),
            self_att_opts=dict(
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
                pos_emb_dropout=pos_emb_dropout,
            ),
            ff_activation=lambda x: rf.relu(x) ** 2.0,
        ),
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
    )


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


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


def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    logits, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

    # LM model and outputs:
    ex_lm = model.lstm_model

    target_tensor = targets.raw_tensor.long()
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )
    lm_target = targets_w_eos.raw_tensor.long()

    # move the LM to device
    cur_device = target_tensor.device
    if next(ex_lm.parameters()).device != cur_device:
        print("move the LM to gpu")
        ex_lm.to(cur_device)
    target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    shifted_target = torch.nn.functional.pad(target_tensor, (1,0,0,0))
    shifted_target_length = target_lengths + 1

    lm_output = ex_lm(shifted_target)
    lm_loss = torch.nn.functional.cross_entropy(lm_output.transpose(1, 2), lm_target, reduction='none')
    seq_mask = get_seq_mask(shifted_target_length, lm_output.shape[1], lm_target.device)
    lm_loss = (lm_loss * seq_mask).sum()
    ppl = torch.exp(lm_loss/shifted_target_length.sum())
    rf.get_run_ctx().mark_as_loss(name='ppl', loss=ppl, as_error=True)

    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_loss = rf.ctc_loss(
                logits=aux_logits,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    loss = rf.ctc_loss(
        logits=logits,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )
    extra_loss = pytorch_loss(logits, targets, model.blank_idx, input_lengths=enc_spatial_dim, target_lengths=targets_spatial_dim, lm_outputs=lm_output)
    prefix_posterior = ctc_prefix_posterior(logits,targets, input_lengths=enc_spatial_dim, target_lengths=targets_spatial_dim ,blank_index=model.blank_idx)
    # prefix_posterior shape (B,S+1,V+1) , LM output shape (B, S+1, V)
    prefix_posterior_no_blank = prefix_posterior[:,:,:-1]
    lm_log_prob = lm_output.detach() # just to make sure no grad computed for the LM
    kl_div_loss = torch.nn.functional.kl_div(prefix_posterior_no_blank,lm_log_prob,reduction='sum', log_target=True)
    rf.get_run_ctx().mark_as_loss(name='lm_kl_loss',
                                  loss=kl_div_loss,
                                  custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                                  use_normalized_loss=use_normalized_loss, )

    print('get run#############', prefix_posterior[0])

    rf.get_run_ctx().mark_as_loss(name='debug_loss',
                                  loss=extra_loss,
                                  custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                                  use_normalized_loss=use_normalized_loss,)


def pytorch_loss(logits, targets, blank_index, input_lengths: Dim, target_lengths: Dim, lm_outputs=None):
    print(lm_outputs.shape)
    torch_logits = logits.raw_tensor # shape (B, T, V)
    torch_logits = torch_logits.transpose(0,1) # (T, B, V) for nn.functional.ctc_loss
    log_probs = nn.functional.log_softmax(torch_logits, dim=-1)
    torch_input_lengths = input_lengths.dyn_size_ext.raw_tensor
    torch_target_lengths = target_lengths.dyn_size_ext.raw_tensor
    torch_targets = targets.raw_tensor.long()
    test_loss = nn.functional.ctc_loss(log_probs, torch_targets, input_lengths=torch_input_lengths,
                                       target_lengths=torch_target_lengths,
                                       blank=blank_index, reduction='sum')
    print('test_loss##############:', test_loss.detach().cpu().numpy())
    return test_loss


def ctc_prefix_posterior(logits, targets, input_lengths, target_lengths, blank_index):
    '''
    logits: ctc outputs with shape (B,T,V)
    targets: target seq without eos
    blank_index:
    input_lengths: length of the input
    '''
    log_probs = nn.functional.log_softmax(logits.raw_tensor.transpose(0,1), dim=-1) # to shape (T,B,V)
    # confirmed that the log prob of eos (index 0) is very low, around -50 or so
    torch_input_lengths = input_lengths.dyn_size_ext.raw_tensor
    torch_targets = targets.raw_tensor.long()
    batch_size, max_seq_len = torch_targets.shape
    prefix_score, _ = log_ctc_pref_beam_scores(log_probs, torch_targets, torch_input_lengths, blank_idx=blank_index)
    prefix_score_norm_1 = prefix_score[:, :-1, :]
    indices = torch_targets.unsqueeze(-1)
    prefix_score_norm_2 = prefix_score_norm_1.gather(-1, indices)
    prefix_score_norm = prefix_score_norm_2.squeeze(-1)
    prefix_score_norm = torch.cat([torch.zeros((batch_size, 1),device=prefix_score_norm.device), prefix_score_norm], dim=-1)
    prefix_posterior_v2 = torch.logsumexp(prefix_score, dim=-1)
    # print('prefix score norm', prefix_score_norm[0].detach().cpu().numpy())
    # print('prefix posterior v2 norm', prefix_posterior_v2[0].detach().cpu().numpy())
    prefix_posterior = prefix_score - prefix_score_norm.unsqueeze(-1)
    target_posterior = prefix_posterior[:, :-1, :].gather(-1, indices).squeeze(-1)[0]
    #print("each position posterior", torch.exp(target_posterior).detach().cpu().numpy())
    torch_target_lengths = target_lengths.dyn_size_ext.raw_tensor.long()
    # print('questionalbe rf tensor', target_lengths.dyn_size_ext)
    # print('questionable get size tensor', target_lengths.get_size_tensor().raw_tensor)
    #print(torch_target_lengths.device)
    torch_target_lengths = torch_target_lengths.to(prefix_posterior.device)
    # print('questionable tensor', torch_target_lengths)
    # print('questionable tensor shape', torch_target_lengths.shape)
    # print('log probs shape', log_probs.shape)
    # print('blank idx', blank_index)
    print('prefix_posterior shape', prefix_posterior.shape)
    print('prefix_score shape', prefix_score.shape)

    final_ctc_prob = prefix_score[:,:,blank_index].gather(-1,torch_target_lengths.unsqueeze(-1)).squeeze(-1)
    target_last_position = torch_target_lengths-1
    last_target_label = torch_targets.gather(-1, target_last_position.unsqueeze(-1)).squeeze(-1)
    print('targets shape', torch_targets.shape)
    print('target_lengths', torch_target_lengths.detach().cpu().numpy())
    print('last label index*******', last_target_label.detach().cpu().numpy())
    print('eos log prob', log_probs[:,0,0].detach().cpu().numpy())
    final_prob_sum = torch.sum(final_ctc_prob)

    print('prefix ctc score#########', final_prob_sum.detach().cpu().numpy())


    return prefix_posterior



ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"


def model_recog(
    *,
    model: Model,
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
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.log_softmax(logits, axis=model.wb_target_dim)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.wb_target_dim]
    )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...


def _gather_backrefs(s, *, backrefs: Tensor):
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        assert s.dimension or backrefs not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")

from i6_experiments.users.yang.torch.lm.network.lstm_lm import LSTMLM, LSTMLMConfig
from i6_experiments.users.yang.torch.utils import get_seq_mask
# for debugging, use a default LSTM config


# class Model(rf.Module):
#     def __init__(self, in_dim, **kwargs):
#         super().__init__()
#         self.ctc_model = CTCModel(in_dim, **kwargs)
#         self.lstm_model = LSTMLM(step=0, cfg=get_lstm_default_config()) # dirty way to add the LM, will be called in the loss function
#
#
#
#     def __call__(
#         self,
#         source: Tensor,
#         *,
#         in_spatial_dim: Dim,
#         collected_outputs: Optional[Dict[str, Tensor]] = None,):
#         am_output, enc_spatial_dim= self.ctc_model(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
#
#         return am_output, enc_spatial_dim





def get_lstm_default_config(**kwargs):
    num_outputs = kwargs.get('num_outputs', 10025)
    embed_dim = kwargs.get('embed_dim', 512)
    hidden_dim = kwargs.get('hidden_dim', 2048)
    num_lstm_layers = kwargs.get('num_lstm_layers',2)
    bottle_neck = kwargs.get('bottle_neck', False)
    bottle_neck_dim = kwargs.get('bottle_neck_dim', 512)
    dropout = kwargs.get('dropout', 0.2)
    default_init_args = {
        'init_args_w':{'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}},
        'init_args_b': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}
    }
    init_args = kwargs.get('init_args', default_init_args)
    model_config = LSTMLMConfig(
        vocab_dim=num_outputs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_lstm_layers=num_lstm_layers,
        init_args=init_args,
        dropout=dropout,
        trainable= False,
    )
    return model_config


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
    ):
        super(Model, self).__init__()
        ############ external LM, pure pytorch code
        lstm_cfg = get_lstm_default_config()
        self.lstm_model = LSTMLM(step=0, cfg=lstm_cfg)
        # load the hyperparameters from checkpoints
        # modelpath, hardcoded:
        lstm_path = "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.la2CPTQHhFyg/output/models/epoch.030.pt"
        self.lstm_model.load_state_dict(torch.load(lstm_path)["model"])
        if not lstm_cfg.trainable:
            self.lstm_model._param_freeze()



        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        else:
            enc_sequential = rf.Sequential

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
            sequential=enc_sequential,
        )


        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(enc_att_num_heads)

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        #self.enc_logits = rf.Linear(self.encoder.out_dim, wb_target_dim)
        self.enc_aux_logits_12 = rf.Linear(self.encoder.out_dim, wb_target_dim)
        self.wb_target_dim = wb_target_dim

        if target_dim.vocab and not wb_target_dim.vocab:
            from returnn.datasets.util.vocabulary import Vocabulary

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [model_recog.output_blank_label]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={model_recog.output_blank_label: blank_idx}
            )

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        """encode, get logits"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
        )
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )
        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        #logits = self.enc_logits(enc)
        logits = self.enc_aux_logits_12(enc)
        return logits, enc_spatial_dim