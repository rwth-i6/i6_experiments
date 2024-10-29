from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection, Dict
import tree
import math
import numpy as np
import torch
import torch.nn as nn
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import (
    trafo_lm_kazuki_import,
)
from i6_experiments.users.yang.torch.utils.tensor_ops import mask_eos_label
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask, get_seq_mask_v2
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import ctc_prefix_posterior
from i6_experiments.users.yang.torch.lm.network.lstm_lm import LSTMLM, LSTMLMConfig
from i6_experiments.users.yang.torch.loss.ctc_forward_backward import ctc_forward

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

_log_mel_feature_dim = 80


###################################################################
def print_gpu_memory_usage(pos='0'):
    print("*********************************************************************************************************")
    unused = torch.cuda.memory_reserved(0) / 1e9 - torch.cuda.memory_allocated(0) / 1e9
    print("Pos: {} Total GPU Memory: {:.2f} GB".format(pos, torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("Pos: {} Allocated GPU Memory: {:.2f} GB".format(pos, torch.cuda.memory_allocated(0) / 1e9))
    print("Pos: {} Cached GPU Memory: {:.2f} GB".format(pos, torch.cuda.memory_reserved(0) / 1e9))
    print("Pos: {} Reserved but Unused GPU Memory: {:.2f} GB".format(pos, unused))
################## extern LM in training ##########################
 # default_config
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
        trainable=False,
        log_prob_output=kwargs.get('log_prob_output', False),
    )
    return model_config
class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers: int = 12,
        train_extern_lm: str =  None,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers
        self.train_extern_lm = train_extern_lm

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers, train_extern_lm=self.train_extern_lm)

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        *,
        num_enc_layers: int = 12,
        pos_emb_dropout: float = 0.0,
        language_model: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> Model:
        """make"""
        lm = None
        if language_model:
            assert isinstance(language_model, dict)
            language_model = language_model.copy()
            cls_name = language_model.pop("class")
            assert cls_name == "TransformerDecoder"
            language_model.pop("vocab_dim", None)  # will just overwrite

            from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm.trafo_lm import (
                trafo_lm,
            )

            lm = trafo_lm.MakeModel(vocab_dim=target_dim, **language_model)()
            lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

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
            language_model=lm,
            **extra,
        )


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
        # enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        # att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        # att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
        language_model: Optional[RFModelWithMakeLabelScorer] = None,
        train_extern_lm: str =None,
        joiner_dim: int = 640,
        model_args: dict = {},
        **kwargs,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)
        if 'use_tedlium_mel_norm' in model_args:
            use_tedlium_mel_norm = model_args['use_tedlium_mel_norm']
        else:
            use_tedlium_mel_norm = False
        self.mel_normalization = use_tedlium_mel_norm
        if 'use_librispeech_mel' in model_args:
            # scale the features to librispeech
            self.transfer_to_librispeech = model_args['use_librispeech_mel']
        else:
            self.transfer_to_librispeech = False

        #self.mel_normalization = config.typed_value("mel_normalization_ted2", False)

        self.use_specaugment = config.typed_value("use_specaugment", True)

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.target_dim = target_dim
        self.target_dim_w_blank = wb_target_dim if wb_target_dim is not None else target_dim + 1
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        # self.enc_key_total_dim = enc_key_total_dim
        # self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        # self.att_num_heads = att_num_heads
        # self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.search_args= {}

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        for p in self.parameters():
            p.weight_decay = l2

        for i in enc_aux_logits:
            setattr(
                self,
                f"enc_aux_logits_{i}",
                rf.Linear(self.encoder.out_dim, self.target_dim_w_blank),
            )

        self.enc_aux_logits_12 = rf.Linear(self.encoder.out_dim, self.target_dim_w_blank)

        if model_args is not None:
            ctc_output_args = model_args.get("ctc_output_args", None)
        else:
            ctc_output_args = None
        if ctc_output_args is not None:
            ctc_enc_layer_id = ctc_output_args.get("ctc_enc_layer_id", 12)
            ctc_output_layer_name = f"enc_aux_logits_{ctc_enc_layer_id}"
            if int(ctc_enc_layer_id) not in enc_aux_logits:
                setattr(self, ctc_output_layer_name, rf.Linear(self.encoder.out_dim, self.target_dim_w_blank))
            self.ctc_output_layer = getattr(self, ctc_output_layer_name)

            self.ctc_enc_layer_id = str(int(ctc_enc_layer_id)-1)
        else:
            # by default take the output of the last layer

            self.ctc_output_layer = self.enc_aux_logits_12 # alias, for decoding
            self.ctc_enc_layer_id = str(11)

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value(
                "specaugment_max_consecutive_spatial_dims"
            )
            or 20,
            "max_consecutive_feature_dims": config.typed_value(
                "specaugment_max_consecutive_feature_dims"
            )
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value(
                "specaugment_num_spatial_mask_factor"
            )
            or 100,
        }

        self._pretrain_opts: Optional[Dict[str, Any]] = config.typed_value(
            "pretrain_opts"
        )

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import (
                Mixup,
                MixupOpts,
            )

            self._mixup = Mixup(
                feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup"))
            )

        # Note: Even though we have this here, it is not used in loop_step or decode_logits.
        # Instead, it is intended to make a separate label scorer for it.
        self.language_model = None
        self.language_model_make_label_scorer = None
        if language_model:
            self.language_model, self.language_model_make_label_scorer = language_model
        #print_gpu_memory_usage(pos='before LM Load')
        if train_extern_lm == 'lstm':
            lstm_cfg = get_lstm_default_config()
            self.train_extern_lm = LSTMLM(step=0, cfg=lstm_cfg)
            lstm_path = "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.la2CPTQHhFyg/output/models/epoch.030.pt"
            self.train_extern_lm.load_state_dict(torch.load(lstm_path)["model"])
            if not lstm_cfg.trainable:
                self.train_extern_lm._param_freeze()
            print('*************************train extern lm loaded***********************************')
            #print_gpu_memory_usage(pos='after LM Load')

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )

        if self.mel_normalization:
            ted2_global_mean = rf.Tensor(
                name="ted2_global_mean",
                dims=[source.feature_dim],
                dtype=source.dtype,
                raw_tensor=torch.tensor(
                    np.loadtxt(
                        "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/mean",
                        dtype="float32",
                    )
                ),
            )
            ted2_global_stddev = rf.Tensor(
                name="ted2_global_stddev",
                dims=[source.feature_dim],
                dtype=source.dtype,
                raw_tensor=torch.tensor(
                    np.loadtxt(
                        "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/std_dev",
                        dtype="float32",
                    )
                ),
            )

            source = (source - rf.copy_to_device(ted2_global_mean)) / rf.copy_to_device(
                ted2_global_stddev
            )
        if self.transfer_to_librispeech:
            assert self.mel_normalization, "feature should be normalized first"

            librispeech_gloabl_mean = rf.Tensor(
                name="libri_global_mean",
                dims=[source.feature_dim],
                dtype=source.dtype,
                raw_tensor=torch.tensor(
                    np.loadtxt(
                        "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/forward/ReturnnForwardJobV2.hZgirhXSIs6U/work/stats.mean.txt",
                        dtype="float32",
                    )
                ),
            )
            librispeech_gloabl_std = rf.Tensor(
                name="ted2_global_stddev",
                dims=[source.feature_dim],
                dtype=source.dtype,
                raw_tensor=torch.tensor(
                    np.loadtxt(
                        "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/forward/ReturnnForwardJobV2.hZgirhXSIs6U/work/stats.std_dev.txt",
                        dtype="float32",
                    )
                ),
            )
            source = source * rf.copy_to_device(librispeech_gloabl_std) + rf.copy_to_device(librispeech_gloabl_mean)

        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        if self.use_specaugment:
            # SpecAugment
            source = rf.audio.specaugment(
                source,
                spatial_dim=in_spatial_dim,
                feature_dim=self.in_dim,
                **self._specaugment_opts,
            )
        # Encoder including convolutional frontend
        with _opt_apply_pretrain_to_encoder(
            self.encoder, collected_outputs, self._pretrain_opts
        ):
            enc, enc_spatial_dim = self.encoder(
                source,
                in_spatial_dim=in_spatial_dim,
                collected_outputs=collected_outputs,
            )

        return (
            dict(enc=enc),
            enc_spatial_dim,
        )

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


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim, **kwargs) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    lm_opts = config.typed_value("external_language_model")
    train_extern_lm = config.typed_value("train_load_extern_lm", "lstm")
    return MakeModel.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        language_model=lm_opts,
        train_extern_lm=train_extern_lm,
        **kwargs,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 21 #16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = 160




def from_scratch_training(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    # import for training only, will fail on CPU servers
    # from i6_native_ops import warp_rnnt

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value(
        "aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None
    )
    # aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(
        data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs
    )
    mask_eos = config.bool("mask_eos_output", True)
    add_eos_to_blank = config.bool("add_eos_to_blank", False)
    kd_layer = config.typed_value("kd_layer", 12) # for now only one layer to do kd
    kd_layer = int(kd_layer)
    ctc_kd_logits = None


    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            if mask_eos:
                mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)
            if layer_idx == kd_layer:
                ctc_kd_logits = aux_logits
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
            # Does not work yet. Was commented out before.
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    aux_logits_12 = model.enc_aux_logits_12(collected_outputs[str(11)])
    if mask_eos:
        mask_eos_label(aux_logits_12, add_to_blank=add_eos_to_blank)
    if kd_layer == 12:
        aux_logits = aux_logits_12
    else:
        assert ctc_kd_logits is not None
        aux_logits = ctc_kd_logits

    ctc_scale = config.typed_value("ctc_scale", 1.0)
    # pure torch LM KD loss
    compute_lm_kd_loss = config.bool('lm_kd_loss', True)
    # print_gpu_memory_usage(pos='before kd loss')
    if compute_lm_kd_loss:
        assert hasattr(model, "train_extern_lm")
        # only makes sense when eos is set to 0
        assert mask_eos
        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
        )
        targets_w_eos, _ = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx,
            out_dims=[targets_w_eos_spatial_dim]
        )
        torch_lm_input_labels = input_labels.raw_tensor # in shape (B, S+1)
        torch_lm_target_labels = targets_w_eos.raw_tensor
        extern_lm = model.train_extern_lm
        cur_device = torch_lm_input_labels.device
        if next(extern_lm.parameters()).device != cur_device:
            print("move the LM to gpu")
            extern_lm.to(cur_device)
        # print_gpu_memory_usage('before compute LM')
        # use log prob for debugging, later should be changed to logits
        lm_output = extern_lm(torch_lm_input_labels)
        ##### top-k applied to reduce memory usage
        K = config.typed_value("kd_top_k", 200)
        freeze_gamma = config.bool("freeze_gamma", False)
        freeze_ctc_p = config.bool("freeze_ctc_p", False)
        lm_output_top_k, top_k_list = torch.topk(lm_output, K, dim=-1)
        torch_targets = targets.raw_tensor



        # manually repeat the targets to check the label mask
        torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor
        torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
        torch_target_w_eos_lengths = targets_w_eos_spatial_dim.dyn_size_ext.raw_tensor
        max_target_w_eos_length = lm_output.shape[1]
        lm_scale = config.typed_value("train_lm_scale", 1.0)
        lm_output_top_k = lm_scale * lm_output_top_k
        top_k_list = top_k_list[:,:-1,:] # (B,S,K) no eos position for fw-bw kd
        lm_output_top_k = lm_output_top_k[:, :-1,:] # (B,S,K), no eos position
        if config.bool("eos_mask", False):
            fw_bw_eos_mask = (top_k_list != model.eos_idx).float()
            fw_bw_eos_log_mask = -1e25 * (1. - fw_bw_eos_mask) # unmasked pos 0, masked pos log_zero
            lm_output_top_k = lm_output_top_k + fw_bw_eos_log_mask

        log_lm_output_top_k_renorm = torch.nn.functional.log_softmax(lm_output_top_k, dim=-1) # used for kl loss computation


        if add_eos_to_blank:
            ctc_log_prob = aux_logits.raw_tensor # the output is already normalized by mask_eos_label func # shape (B,T,V)
        else:
            ctc_log_prob = nn.functional.log_softmax(aux_logits.raw_tensor, dim=-1)

        #torch.cuda.empty_cache()
        # the computation of gamma should be correct
        # print_gpu_memory_usage(pos='after compute ctc prefix')

        backward = True
        batch_size = ctc_log_prob.shape[0]
        # debug_top_k = True
        # if debug_top_k:
        #     top_k_list = torch.cat([top_k_list, torch_targets.unsqueeze(-1)], dim=-1) # always add target

        gamma_fw, (gamma_bw, fw_bw) = ctc_forward(
            log_probs=ctc_log_prob.transpose(0,1),
            targets=torch_targets,  # (B, S)
            targets_w_bos=torch_lm_input_labels,  # (B S+1)
            targets_w_eos=torch_lm_target_labels,  # (B, S+1)
            input_lengths=torch_input_lengths,  # (B,)
            target_length=torch_target_lengths,  # (B,)
            blank_idx=model.blank_idx,
            eos_idx=model.eos_idx,
            bos_idx=model.bos_idx,
            log_zero=-1e25,  # maybe better than float min for preventing overflowing
            backward=backward,
            top_k_list=top_k_list,)

        # gamma_fw [T,B,2,S+1]
        # based on the input and output length, get the corresponding value of gamma
        #
        final_score = gamma_fw[torch_input_lengths-1, torch.arange(batch_size), :, torch_target_lengths] # shape (B,2)?
        final_score = final_score.logsumexp(dim=-1)

        # aux_loss = rf.ctc_loss(
        #     logits=aux_logits_12,
        #     targets=targets,
        #     input_spatial_dim=enc_spatial_dim,
        #     targets_spatial_dim=targets_spatial_dim,
        #     blank_index=model.blank_idx,
        # )
        #print("!!!aux_loss", aux_loss.raw_tensor.detach().cpu().numpy())

        #torch_ctc_loss = torch.nn.functional.ctc_loss(ctc_log_prob.transpose(0,1), torch_targets, torch_input_lengths, torch_target_lengths, blank=model.blank_idx, reduction='none')

        #print("!!!torch ctc loss", torch_ctc_loss.detach().cpu().numpy())




        if top_k_list is not None:
            if config.bool("eos_mask", False):
                fw_bw = fw_bw + fw_bw_eos_log_mask
            fw_bw_renorm = fw_bw.log_softmax(dim=-1)
            # fake_fw_bw_renorm = fake_fw_bw.log_softmax(dim=-1)
            # print("!!! top_k_list", top_k_list[0,:7,:].detach().cpu().numpy())
            # print("!!!fwbw  value", fake_fw_bw[0, :7, :].squeeze(-1).detach().cpu().numpy())
            # print("!!!fwbw  last values", fake_fw_bw[0, -7:, :].squeeze(-1).detach().cpu().numpy())
            fwbw_kl_loss = torch.nn.functional.kl_div(input=fw_bw_renorm,  target=log_lm_output_top_k_renorm, reduction='none', log_target=True)
            # print('!!!#### fwbw kl shape', fwbw_kl_loss.shape)
            target_mask = get_seq_mask(seq_lens=torch_target_lengths, max_seq_len=torch_targets.shape[1], device=fwbw_kl_loss.device)
            target_mask = target_mask.unsqueeze(-1) * fw_bw_eos_mask
            fwbw_kl_loss = fwbw_kl_loss * target_mask


        # if config.bool("target_in_top_mask", False):
        #     target_mask = torch.any(torch_lm_target_labels.unsqueeze(-1) == top_k_list, dim=-1)  # (B, S+1) check if the target label is in the top-k list
        #     kl_div_loss = torch.where(target_mask.unsqueeze(-1), kl_div_loss, torch.zeros_like(kl_div_loss))

        # print_gpu_memory_usage(pos='after compute kl loss')

        kd_scale = config.typed_value("kd_scale", 0.2)
        rf.get_run_ctx().mark_as_loss(
            name="lm_kd_loss",
            loss=fwbw_kl_loss.sum(),
            scale=kd_scale,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )


        # final layer ctc loss
        if freeze_gamma or freeze_ctc_p:
            assert False
            aux_loss = rf.ctc_loss(
                logits=aux_logits,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_12",
                scale=ctc_scale,
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
        else:
            if kd_layer == 12:
                rf.get_run_ctx().mark_as_loss(
                    name="debug_ctc",
                    loss=-final_score.sum(),
                    scale=ctc_scale,
                    custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                    use_normalized_loss=use_normalized_loss,
                )
            else:
                # the kd forward path does not compute the final score, compute the final ctc loss explicitly
                aux_loss = rf.ctc_loss(
                    logits=aux_logits_12,
                    targets=targets,
                    input_spatial_dim=enc_spatial_dim,
                    targets_spatial_dim=targets_spatial_dim,
                    blank_index=model.blank_idx,
                )
                aux_loss.mark_as_loss(
                    f"ctc_12",
                    scale=ctc_scale,
                    custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                    use_normalized_loss=use_normalized_loss,
                )
    else:
        # only standard ctc loss
        aux_loss = rf.ctc_loss(
            logits=aux_logits_12,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_12",
            scale=ctc_scale,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )








from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


@contextlib.contextmanager
def _opt_apply_pretrain_to_encoder(
    encoder: ConformerEncoder,
    collected_outputs: Optional[Dict[str, Tensor]],
    pretrain_opts: Optional[Dict[str, Any]],
):
    """Function is run within RETURNN."""
    if not pretrain_opts:
        yield
        return
    step = rf.get_run_ctx().step
    steps: Union[
        Sequence[Tuple[int, Dict[str, Any]]], Dict[int, Dict[str, Any]]
    ] = pretrain_opts["steps"]
    if isinstance(steps, (list, tuple)):
        steps_ = {}
        step_bound = 0
        for step_bound_rel, opts in steps:
            step_bound += step_bound_rel
            steps_[step_bound] = opts
        steps = steps_
    assert isinstance(steps, dict)
    for step_bound, opts in sorted(steps.items()):
        if step < step_bound:
            assert isinstance(opts, dict)
            opts_ = opts.copy()
            # somewhat hacky but that is still the easiest way I can think of, without touching a lot of other code
            pretrain_num_layers = opts_.pop("num_layers")
            assert (
                not opts_
            ), f"unhandled opts: {opts_} in opts {opts} for step bound {step_bound}"
            orig_layers = encoder.layers[:]
            del encoder.layers[pretrain_num_layers:]
            yield
            encoder.layers[:] = orig_layers
            if collected_outputs is not None:
                assert len(collected_outputs) == pretrain_num_layers
                for i in range(pretrain_num_layers, len(orig_layers)):
                    collected_outputs[str(i)] = collected_outputs[
                        str(pretrain_num_layers - 1)
                    ]
            return
    yield
    return
