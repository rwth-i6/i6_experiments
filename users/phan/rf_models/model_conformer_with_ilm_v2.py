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

from returnn.tensor import Tensor, Dim, single_step_dim, batch_dim
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
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask_v2
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import ctc_prefix_posterior
#from i6_experiments.users.yang.torch.lm.network.lstm_lm import LSTMLM, LSTMLMConfig

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

_log_mel_feature_dim = 80

from i6_experiments.users.phan.rf_models.lstm_lm import LSTMLMRF
from i6_experiments.users.phan.rf_models.bigram import BigramLMRF
from i6_experiments.users.phan.rf_models.trafo_lm_luca import Trafo_LM_Model
from i6_experiments.users.yang.torch.loss.ctc_forward_backward import ctc_forward
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import kldiv_ctc_lm_loss, kldiv_ctc_lm_sample_batch_loss, ctc_double_softmax_loss
from i6_experiments.users.phan.ctc_lf_mmi import ctc_lf_mmi_context_1_topk
from i6_experiments.users.phan.utils.masking import get_seq_mask, mask_audio_features_with_alignments, \
    mask_audio_features_exact_label_pos_single_seq
from i6_experiments.users.phan.alignment.convert import map_sublabels_to_pseudo_label_indices

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
        internal_language_model: Optional[dict] = None,
        external_language_model: Optional[dict] = None, # for recog only
        freeze_encoder: bool = True,
        load_vocab: Optional[str] = None,
        batch_norm_eval_mode = True,
        **extra,
    ) -> Model:
        """make"""
        lm = None
        # if language_model:
        #     assert isinstance(language_model, dict)
        #     language_model = language_model.copy()
        #     cls_name = language_model.pop("class")
        #     assert cls_name == "TransformerDecoder"
        #     language_model.pop("vocab_dim", None)  # will just overwrite

        #     from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm.trafo_lm import (
        #         trafo_lm,
        #     )

        #     lm = trafo_lm.MakeModel(vocab_dim=target_dim, **language_model)()
        #     lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True, eval_mode=batch_norm_eval_mode), # eval_mode=True to prevent updating running stats
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
            internal_language_model=internal_language_model,
            external_language_model=external_language_model,
            freeze_encoder=freeze_encoder,
            load_vocab=load_vocab,
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
        freeze_encoder: bool = True,
        internal_language_model: Optional[dict] = None,
        external_language_model: Optional[dict] = None, # for recog
        load_vocab: Optional[str] = None,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.mel_normalization = config.typed_value("mel_normalization_ted2", False)
        self.use_specaugment = config.typed_value("use_specaugment", True) # Should be False

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
        self.target_dim_w_blank = target_dim + 1
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        # self.enc_key_total_dim = enc_key_total_dim
        # self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        # self.att_num_heads = att_num_heads
        # self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        for p in self.parameters():
            p.weight_decay = l2

        if enc_aux_logits:
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(
                self,
                f"enc_aux_logits_{i}",
                rf.Linear(self.encoder.out_dim, wb_target_dim),
            )

        self.enc_aux_logits_12 = rf.Linear(self.encoder.out_dim, self.target_dim_w_blank)

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
        
        # Define the accompanying LM, can be ILM or extern LM used in training
        label_target_dim = target_dim
        if internal_language_model is not None:
            ilm_cls = internal_language_model.pop("class")
            if ilm_cls == "LSTMLMRF":
                self.ilm = LSTMLMRF(
                    label_target_dim,
                    label_target_dim,
                    **internal_language_model,
                )
            elif ilm_cls == "BigramLMRF":
                self.ilm = BigramLMRF(
                    label_target_dim,
                    label_target_dim,
                    **internal_language_model,
                )
            else:
                raise NotImplementedError(f"The ILM class {ilm_cls} is not supported !!!!!!")

        if external_language_model is not None:
            lm_cls = external_language_model.pop("class")
            if lm_cls == "Trafo_LM_Model":
                self.language_model = Trafo_LM_Model(
                    label_target_dim,
                    label_target_dim,
                    **external_language_model,
                )
            elif lm_cls == "LSTM_LM_Model":
                from i6_experiments.users.phan.rf_models.lstm_lm_luca import LSTM_LM_Model
                self.language_model = LSTM_LM_Model(
                    label_target_dim,
                    label_target_dim,
                    **external_language_model,
                )
            elif lm_cls == "LSTM_LM_Model_Hardcoded_Layers":
                from i6_experiments.users.phan.rf_models.lstm_lm_luca_hardcoded_layers import LSTM_LM_Model_Hardcoded_Layers
                self.language_model = LSTM_LM_Model_Hardcoded_Layers(
                    label_target_dim,
                    label_target_dim,
                    **external_language_model,
                )
            else:
                raise NotImplementedError(f"The LM class {lm_cls} is not supported !!!!!!")

        # Freeze the encoder
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.trainable = False

        # Freeze ilm if needed
        from returnn.config import get_global_config
        config = get_global_config()
        freeze_ilm = config.typed_value("freeze_ilm", False) # was the default behavior
        if freeze_ilm and hasattr(self, "ilm"):
            for name, param in self.ilm.named_parameters():
                param.trainable = False

        # load the vocab. This is needed in masking training, because the model
        # needs to know which BPE is EOW for word to mask the whole word
        if load_vocab is not None:
            with open(load_vocab, "r") as vocab_file:
                vocab = eval(vocab_file.read()) #{ '<s>': 0, '</s>': 0, '<unk>': 1, 'THE': 2, 'AND': 3, 'OF': 4, 'TO': 5, ..}
            self.bpe_idx_to_label = {idx: label for label, idx in vocab.items()} # {0: '</s>', 1: '<unk>', 2: 'THE', 10002: 'TAGN@@', ...}
            self.bpe_idx_is_eow = {idx: idx != self.eos_idx and idx != self.bos_idx and not label.endswith("@@") for idx, label in self.bpe_idx_to_label.items()}
    
    def ilm_forward(
        self,
        targets: Tensor,
        out_spatial_dim: Dim,
    ):
        """
        Feed the target label sequences to the LM and forward
        Shape in: (B, S), shape out: (S, B, V)
        """
        ilm_out = self.ilm(targets, out_spatial_dim)
        return ilm_out

    def feature_extraction(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
    ):
        """
        Simply do features extraction and return the feature-extracted input
        with the spatial dim
        """
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )
        return source, in_spatial_dim

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        audio_features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder
        
        :param audio_features_mask: Now only implemented for torch Tensor. RF later.
        Mask the audio according to this mask (1 = no mask, 0 = mask). Interpolated
        to match the number of frames after log mel. (B, T_align)
        """
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

        if audio_features_mask is not None:
            if isinstance(audio_features_mask, torch.Tensor):
                _, time_size, feature_size = source.raw_tensor.shape
                mask_with_channel = audio_features_mask.unsqueeze(1) # (B, 1, T_align)
                mask_resampled = torch.nn.functional.interpolate(mask_with_channel, (time_size,)) # (B, 1, T)
                mask_resampled_shaped = mask_resampled.squeeze(1).unsqueeze(-1).expand(-1, -1, feature_size) # (B, T, F)
                source.raw_tensor = source.raw_tensor * mask_resampled_shaped
            else:
                raise NotImplementedError("Now the audio_features_mask can only be torch.Tensor")
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

# where the model is defined
def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    internal_language_model = config.typed_value("internal_language_model")
    external_language_model = config.typed_value("external_language_model")
    freeze_encoder = config.typed_value("freeze_encoder", True) # was the default behavior
    load_vocab = config.typed_value("load_vocab", None)
    return MakeModel.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        train_extern_lm="lstm",
        internal_language_model=internal_language_model,
        external_language_model=external_language_model,
        freeze_encoder=freeze_encoder,
        load_vocab=load_vocab,
        batch_norm_eval_mode=freeze_encoder
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
            # Does not work yet. Was commented out before.
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])
    aux_loss = rf.ctc_loss(
        logits=aux_logits,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    aux_loss.mark_as_loss(
        f"ctc_12",
        scale=1.0,
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )



from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"



# train step
# kldiv train step, more comes later
def from_scratch_training_kldiv(
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

    if False: # disable ctc losses
        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
                if layer_idx > len(model.encoder.layers):
                    continue
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                if mask_eos:
                    mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)
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

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)
    if False: # disable ctc losses
        aux_loss = rf.ctc_loss(
            logits=aux_logits,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_12",
            scale=1.0,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )

    ######### Needed stuffs for ILM estimation ##########
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=model.eos_idx
    )
    ilm_out = model.ilm_forward(targets_w_bos, targets_spatial_dim_pad[0])
    ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor.detach() # (B, T, V + blank), good
    targets_raw = targets.raw_tensor
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor


    ##### Debug
    # print(targets_raw)
    # print(lm_out_raw) # RF output is always (T, B, V + blank)
    # print(collected_outputs["11"].raw_tensor) # This is just before the final linear, that's why it's 512
    # print(aux_logits_raw) # this contains the EOS symbol, but should be okay, EOS probs are masked anw

    ###### apply KD with top K
    # just use the old implementation, but eventually should move to top K
    # Swap blank to position 0 for safety
    ####### Remeber to adjust the targets!!!!
    # aux_logits_raw = torch.concat( 
    #     [aux_logits_raw[:, :, -1:], aux_logits_raw[:, :, :-1]],
    #     dim=-1,
    # ).transpose(0, 1) # (T, B, blank + V)
    
    # expect the kldiv in the beginning to be near -ln(1/10026) ~ 9.x
    log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, S, V)
    am_scale = config.typed_value("teacher_scale", 1.0)
    kldiv = kldiv_ctc_lm_loss(
        aux_logits_raw.transpose(0, 1).log_softmax(-1).detach(),
        targets_raw.clone().long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        ilm_out_raw.transpose(0, 1).log_softmax(-1),
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
        am_scale=am_scale,
    )
    targets_len_rf = targets_spatial_dim_pad[0].dyn_size_ext
    rf.get_run_ctx().mark_as_loss(
        kldiv,
        "kldiv",
        custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    )

    # Also report PPL of the LM
    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(torch_target_lengths, max_seq_len+1, targets_raw.device)
    log_ppl = (ce*seq_mask).sum()/(targets_len_rf.raw_tensor.sum())
    ppl = torch.exp(log_ppl)
    rf.get_run_ctx().mark_as_loss(
        name="student_lm_ppl", loss=ppl, as_error=True,
    )
    # print("blank kept")
    # print(kldiv)


from_scratch_training_kldiv: TrainDef[Model]
from_scratch_training_kldiv.learning_rate_control_error_measure = "dev_score_full_sum"


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


# sample from batch method
def from_scratch_training_kldiv_sample_batch(
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

    if False: # disable ctc losses
        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
                if layer_idx > len(model.encoder.layers):
                    continue
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                if mask_eos:
                    mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)
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

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)
    if False: # disable ctc losses
        aux_loss = rf.ctc_loss(
            logits=aux_logits,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_12",
            scale=1.0,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )

    ######### Needed stuffs for ILM estimation ##########
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=model.eos_idx
    )
    ilm_out = model.ilm_forward(targets_w_bos, targets_spatial_dim_pad[0])
    ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor.detach() # (B, T, V + blank), good
    targets_raw = targets.raw_tensor
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor


    ##### Debug
    # print(targets_raw)
    # print(lm_out_raw) # RF output is always (T, B, V + blank)
    # print(collected_outputs["11"].raw_tensor) # This is just before the final linear, that's why it's 512
    # print(aux_logits_raw) # this contains the EOS symbol, but should be okay, EOS probs are masked anw

    ###### apply KD with top K
    # just use the old implementation, but eventually should move to top K
    # Swap blank to position 0 for safety
    ####### Remeber to adjust the targets!!!!
    # aux_logits_raw = torch.concat( 
    #     [aux_logits_raw[:, :, -1:], aux_logits_raw[:, :, :-1]],
    #     dim=-1,
    # ).transpose(0, 1) # (T, B, blank + V)
    
    # expect the kldiv in the beginning to be near -ln(1/10026) ~ 9.x
    log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, S, V)
    weight = config.typed_value("kldiv_sampling_weight", None)
    assert weight is not None, "Must provide kldiv_sampling_weight"
    kldiv = kldiv_ctc_lm_sample_batch_loss( # it's computing some bullshit here
        aux_logits_raw.transpose(0, 1).log_softmax(-1),
        targets_raw.clone().long(), # +1 due to blank moved from last to 0
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        ilm_out_raw.transpose(0, 1).log_softmax(-1),
        blank_idx=10025,
        eos_idx=model.eos_idx,
        ground_truth_weight=weight,
    ) # no need to normalize this loss when passing to returnn!!!
    targets_len_rf = targets_spatial_dim_pad[0].dyn_size_ext
    rf.get_run_ctx().mark_as_loss(
        kldiv,
        "kldiv_sample_batch",
    )

    # Also report PPL of the LM
    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(torch_target_lengths, max_seq_len+1, targets_raw.device)
    log_ppl = (ce*seq_mask).sum()/(targets_len_rf.raw_tensor.sum())
    ppl = torch.exp(log_ppl)
    rf.get_run_ctx().mark_as_loss(
        name="student_lm_ppl", loss=ppl, as_error=True,
    )


from_scratch_training_kldiv_sample_batch: TrainDef[Model]
from_scratch_training_kldiv_sample_batch.learning_rate_control_error_measure = "dev_score_full_sum"


# double softmax
def from_scratch_training_double_softmax(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN.
    
    The training lm (transcription LM) is embedded in model.ilm (make sense)
    """
    from returnn.config import get_global_config

    # import for training only, will fail on CPU servers
    # from i6_native_ops import warp_rnnt

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value(
        "aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None
    )

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

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)


    ######### Needed stuffs for ILM estimation ##########
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=model.eos_idx
    )
    ilm_out = model.ilm_forward(targets_w_bos, targets_spatial_dim_pad[0])
    ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor # (B, T, V + blank), good
    targets_raw = targets.raw_tensor
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor


    ##### Debug
    # print(targets_raw)
    # print(lm_out_raw) # RF output is always (T, B, V + blank)
    # print(collected_outputs["11"].raw_tensor) # This is just before the final linear, that's why it's 512
    # print(aux_logits_raw) # this contains the EOS symbol, but should be okay, EOS probs are masked anw

    
    # expect the kldiv in the beginning to be near -ln(1/10026) ~ 9.x
    log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1).detach() # (B, S, V)
    ctc_log_posteriors = aux_logits_raw.transpose(0, 1).log_softmax(-1)
    am_scale = config.typed_value("am_scale", None)
    lm_scale = config.typed_value("lm_scale", None)
    assert am_scale is not None, "Must provide am_scale in config"
    assert lm_scale is not None, "Must provide lm_scale in config"
    double_softmax_loss = ctc_double_softmax_loss(
        ctc_log_posteriors,
        targets_raw.clone().long(), # +1 due to blank moved from last to 0
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        log_lm_score,
        am_scale=am_scale,
        lm_scale=lm_scale,
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
    ) # need to normalize this loss when passing to returnn!!!
    targets_len_rf = targets_spatial_dim_pad[0].dyn_size_ext
    rf.get_run_ctx().mark_as_loss(
        double_softmax_loss,
        "double_softmax",
        custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    )

    # Also report the CTC loss (should go up gradually)
    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ctc = torch.nn.functional.ctc_loss(
        ctc_log_posteriors.detach(),
        targets_raw.long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        blank=model.blank_idx,
        reduction="sum",
        zero_infinity=True,
    )
    rf.get_run_ctx().mark_as_loss(
        name="ctc", loss=ctc, as_error=True,
        custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    )


from_scratch_training_double_softmax: TrainDef[Model]
from_scratch_training_double_softmax.learning_rate_control_error_measure = "dev_score_full_sum"


# LF MMI standard version with bigram
def from_scratch_training_lfmmi_context_1(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN.
    
    The training lm (transcription LM) is embedded in model.ilm (make sense)
    """
    from returnn.config import get_global_config

    # import for training only, will fail on CPU servers
    # from i6_native_ops import warp_rnnt

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value(
        "aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None
    )

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

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)


    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor # (B, T, V + blank), good
    targets_raw = targets.raw_tensor
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor

    ######### In this case need the bigram (V, V) ##########
    raw_range = torch.arange(aux_logits_raw.shape[-1], device=aux_logits_raw.device)
    raw_range = raw_range[raw_range != model.blank_idx].unsqueeze(0) # (1, V)
    rf_range = rf.Tensor(
        name="vocab",
        dims=[batch_dim, targets_spatial_dim],
        raw_tensor=raw_range,
        dtype="int64",
        sparse_dim=targets.sparse_dim,
    )
    # The "ILM" here is the bigram
    ilm_out = model.ilm_forward(rf_range, out_spatial_dim=targets_spatial_dim)
    log_bigram_probs = ilm_out["output"].raw_tensor.squeeze(0).detach() # (V, V), what we need
    ctc_log_posteriors = aux_logits_raw.transpose(0, 1).log_softmax(-1)
    # Finally the target scores
    targets_w_bos, targets_spatial_dim_pad = rf.pad( # (B, S+1)
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=model.eos_idx
    )
    log_target_scores = model.ilm_forward(targets_w_bos, out_spatial_dim=targets_spatial_dim_pad[0])["output"]
    log_target_scores_raw = log_target_scores.raw_tensor.log_softmax(-1).detach() # (B, S+1, V)
    batch_size = targets_raw.shape[0]
    targets_raw_w_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=aux_logits_raw.device)],
        dim=-1
    )
    log_target_probs_raw = torch.nn.functional.cross_entropy( # (B, S+1)
        input=log_target_scores_raw.transpose(1, 2),
        target=targets_raw_w_eos,
        reduction="none",
    )
    am_scale = config.typed_value("am_scale", None)
    lm_scale = config.typed_value("lm_scale", None)
    top_k = config.typed_value("top_k", None)
    assert am_scale is not None, "Must provide am_scale in config"
    assert lm_scale is not None, "Must provide lm_scale in config"
    assert top_k is not None, "Must provide top_k in config"
    lfmmi_loss = ctc_lf_mmi_context_1_topk(
        ctc_log_posteriors,
        targets_raw,
        torch_input_lengths,
        torch_target_lengths,
        log_target_probs_raw,
        log_bigram_probs,
        am_scale,
        lm_scale,
        top_k=top_k,
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx
    ) # need to normalize this loss when passing to returnn!!!
    targets_len_rf = targets_spatial_dim.dyn_size_ext
    rf.get_run_ctx().mark_as_loss(
        lfmmi_loss,
        "lf_mmi",
        custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    )

    # Also report the CTC loss (should go up gradually)
    ctc = torch.nn.functional.ctc_loss(
        ctc_log_posteriors,
        targets_raw.long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        blank=model.blank_idx,
        reduction="sum",
        zero_infinity=True,
    )
    rf.get_run_ctx().mark_as_loss(
        name="ctc", loss=ctc, as_error=True,
        custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
    )


from_scratch_training_lfmmi_context_1: TrainDef[Model]
from_scratch_training_lfmmi_context_1.learning_rate_control_error_measure = "dev_score_full_sum"


# masking acoustic input method
def from_scratch_training_kldiv_masking(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    align: Optional[rf.Tensor] = None,
):
    """Function is run within RETURNN.
    This one has the alignment, but the alignment can be None
    Some Returnn code was changed to allow this
    """

    if align is None: # eval on dev or dev-other, whatever
        from_scratch_training_kldiv(
            model=model,
            data=data,
            data_spatial_dim=data_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
        )
        return

    assert hasattr(model, "bpe_idx_to_label"), "Must load vocab for model"
    assert hasattr(model, "bpe_idx_is_eow"), "Must load vocab for model"
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

    
    ######### Needed stuffs for ILM estimation ##########
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=model.eos_idx
    )


    # Now mask audio and the "words"
    masking_rate = config.typed_value("input_masking_rate", None)
    feature_mask, word_mask = mask_audio_features_with_alignments(
        align.raw_tensor,
        mask_ratio=masking_rate,
        sil_index=model.eos_idx, #  in this case it is 0 anyway, same as in the alignment
        )

    # Now "mask" the target labels to know which labels to penalize the loss
    # This is the same as having a predetermined label mask and then mask out
    # the corresponding labels
    targets_w_bos_raw = targets_w_bos.raw_tensor
    label_loss_mask = torch.zeros_like(targets_w_bos_raw, device=data.raw_tensor.device)
    for b in range(targets_w_bos_raw.shape[0]):
        pseudo_word_seq_np = map_sublabels_to_pseudo_label_indices(
            targets_w_bos_raw[b].cpu(),
            is_eow_func=lambda x: model.bpe_idx_is_eow[x.item()], # x is torch tensor, need x.item()
            sil_idx=model.eos_idx
            )
        pseudo_word_seq = torch.tensor(pseudo_word_seq_np, device=data.raw_tensor.device)
        # here it is guaranteed that pseudo_word_seq has the same number or pseudo-words as the "alignments"
        # For word_mask: 1 = mask, 0 = no mask
        # For label_loss_mask: 1 = no mask, 0 = mask
        label_loss_mask[b] = mask_audio_features_exact_label_pos_single_seq(
            pseudo_word_seq,
            word_mask[b],
            sil_index=model.eos_idx,
            )

    label_loss_mask = 1. - label_loss_mask # we want 1 = mask, 0 = no mask for the loss masking

    # # --------------------------- debug -----------------------------------
    # # verifying that targets and alignments have the same number of words, seems OK
    # # verifying the masking are working: all seems OK
    # # - acoustic feature masking are the same as penalized labels
    # # - group of labels belonging to the same word are always masked together
    # # - the masking for the kldiv compuation is correct
    # # and some extra debug print
    # # In doubt, just out comment these
    # torch.set_printoptions(threshold=1000, linewidth=100, precision=2)
    # print()
    # print("-------------- EXTRA DEBUG PRINT --------------")
    # print("Shape of the label loss masking: ", label_loss_mask.shape)
    # import numpy as np
    # targets_rawraw = targets_w_bos.raw_tensor.clone().cpu().detach()
    # targets_words = np.apply_along_axis(lambda x: [model.bpe_idx_to_label[v] for v in x], -1, targets_rawraw.numpy())
    # for b in range(targets_words.shape[0]):
    #     print("Sequence: ", b)
    #     print("Sequence in words:")
    #     print(targets_words[b])
    #     print("Words appearing in the alignment:")
    #     print(align.raw_tensor[b].unique_consecutive())
    #     pseudo_word_seq = map_sublabels_to_pseudo_label_indices(targets_rawraw[b], is_eow_func=lambda x: model.bpe_idx_is_eow[x.item()], sil_idx=model.eos_idx)
    #     print("Label groups (each group is a true word) in the target sequence")
    #     print(pseudo_word_seq)
    #     # print(feature_mask[b].unique_consecutive())
    #     print("The masked alignment, collapsed:")
    #     print((align.raw_tensor[b]*feature_mask[b]).unique_consecutive())
    #     # print(label_loss_mask[b].cpu()*torch.tensor(pseudo_word_seq))
    #     print("The label sequence with masked words:")
    #     print(np.where(label_loss_mask[b].cpu().numpy() == 1., "<masked>", pseudo_word_seq))
    #     targets_words_with_mask = np.where(label_loss_mask[b].cpu().numpy() == 1., "<masked>", targets_words[b])
    #     print("The label sequence next to the masked label sequence and the label loss masking:")
    #     print(np.stack([targets_words[b], targets_words_with_mask, label_loss_mask[b].cpu().numpy()], axis=-1))
    # print()
    # # -----------------------------------------------------------------------

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(
        data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs,
        audio_features_mask=feature_mask,
    )
    mask_eos = config.bool("mask_eos_output", True)
    add_eos_to_blank = config.bool("add_eos_to_blank", False)

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)

    ilm_out = model.ilm_forward(targets_w_bos, targets_spatial_dim_pad[0])
    ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor.detach() # (B, T, V + blank), good
    targets_raw = targets.raw_tensor
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor
    # expect the kldiv in the beginning to be near -ln(1/10026) ~ 9.x
    log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, S, V)
    
    assert masking_rate is not None, "Must provide input_masking_rate in config"
    kldiv = kldiv_ctc_lm_loss(
        aux_logits_raw.transpose(0, 1).log_softmax(-1),
        targets_raw.clone().long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        ilm_out_raw.transpose(0, 1).log_softmax(-1),
        blank_idx=10025,
        eos_idx=model.eos_idx,
        target_mask=label_loss_mask,
    ) # no need to normalize this loss when passing to returnn!!!
    loss = kldiv / label_loss_mask.sum()
    targets_len_rf = targets_spatial_dim_pad[0].dyn_size_ext
    rf.get_run_ctx().mark_as_loss(
        loss,
        "kldiv_masking",
    )

    # Also report PPL of the LM
    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(torch_target_lengths, max_seq_len+1, targets_raw.device)
    log_ppl = (ce*seq_mask).sum()/(targets_len_rf.raw_tensor.sum())
    ppl = torch.exp(log_ppl)
    rf.get_run_ctx().mark_as_loss(
        name="student_lm_ppl", loss=ppl, as_error=True,
    )


from_scratch_training_kldiv_masking: TrainDef[Model]
from_scratch_training_kldiv_masking.learning_rate_control_error_measure = "dev_score_full_sum"


# masking acoustic input method
def train_masked_bi_ilm(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    align: Optional[rf.Tensor] = None,
):
    """Function is run within RETURNN.
    This one has the alignment, but the alignment can be None
    Some Returnn code was changed to allow this
    """

    if align is None: # eval on dev or dev-other, whatever
        from_scratch_training_kldiv(
            model=model,
            data=data,
            data_spatial_dim=data_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
        )
        return

    assert hasattr(model, "bpe_idx_to_label"), "Must load vocab for model"
    assert hasattr(model, "bpe_idx_is_eow"), "Must load vocab for model"
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

    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    
    targets_raw = targets.raw_tensor
    batch_size, max_seq_len = targets_raw.shape
    align_raw = align.raw_tensor

    # Now mask audio and the "words"
    masking_rate = config.typed_value("input_masking_rate", None)

    # Generate a label mask for all of the label sequences
    label_mask = (torch.rand((max_seq_len,)) < masking_rate).long() # 1 = mask, 0 = no mask

    # Calculate the feature mask
    # If one of the BPE label inside a word is masked, that whole word's acoustic input is masked
    feature_mask = torch.ones_like(align_raw, dtype=torch.float32)
    for b in range(batch_size):
        # Transform the target label sequence to "pseudo word"
        pseudo_word_seq_np = map_sublabels_to_pseudo_label_indices(
            targets_raw[b].cpu(),
            is_eow_func=lambda x: model.bpe_idx_is_eow[x.item()], # x is torch tensor, need x.item()
            sil_idx=model.eos_idx
            )
        pseudo_word_seq = torch.tensor(pseudo_word_seq_np, device=data.raw_tensor.device)
        # Determine the words to mask acoustic input
        masked_acoustic_words = pseudo_word_seq[label_mask.bool()].unique()
        feature_mask[b] = torch.where( # 0 = mask, 1 = no mask
            torch.isin(align_raw[b], masked_acoustic_words),
            0.,
            1.,
        )

    label_loss_mask = label_mask.unsqueeze(0).expand(batch_size, -1).float()

    # # --------------------------- debug -----------------------------------
    # # verifying that targets and alignments have the same number of words, seems OK
    # # verifying the masking are working: all seems OK
    # # - acoustic feature masking are the same as penalized labels
    # # - the masking for the kldiv compuation is correct
    # # and some extra debug print
    # # In doubt, just out comment these
    # torch.set_printoptions(threshold=1000, linewidth=100, precision=2)
    # print()
    # print("-------------- EXTRA DEBUG PRINT --------------")
    # print("Shape of the label loss masking: ", label_loss_mask.shape)
    # import numpy as np
    # targets_rawraw = targets.raw_tensor.clone().cpu().detach()
    # targets_words = np.apply_along_axis(lambda x: [model.bpe_idx_to_label[v] for v in x], -1, targets_rawraw.numpy())
    # for b in range(targets_words.shape[0]):
    #     print("Sequence: ", b)
    #     print("Sequence in words:")
    #     print(targets_words[b])
    #     print("Words appearing in the alignment:")
    #     print(align.raw_tensor[b].unique_consecutive())
    #     pseudo_word_seq = map_sublabels_to_pseudo_label_indices(targets_rawraw[b], is_eow_func=lambda x: model.bpe_idx_is_eow[x.item()], sil_idx=model.eos_idx)
    #     print("Label groups (each group is a true word) in the target sequence")
    #     print(pseudo_word_seq)
    #     # print(feature_mask[b].unique_consecutive())
    #     print("The masked alignment, collapsed:")
    #     print((align.raw_tensor[b]*feature_mask[b]).unique_consecutive())
    #     # print(label_loss_mask[b].cpu()*torch.tensor(pseudo_word_seq))
    #     print("The label sequence with masked words:")
    #     print(np.where(label_loss_mask[b].cpu().numpy() == 1., "<masked>", pseudo_word_seq))
    #     targets_words_with_mask = np.where(label_loss_mask[b].cpu().numpy() == 1., "<masked>", targets_words[b])
    #     print("The label sequence next to the masked label sequence and the label loss masking:")
    #     print(np.stack([targets_words[b], targets_words_with_mask, pseudo_word_seq, label_loss_mask[b].cpu().numpy()], axis=-1))
    # print()
    # # -----------------------------------------------------------------------

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(
        data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs,
        audio_features_mask=feature_mask,
    )
    mask_eos = config.bool("mask_eos_output", True)
    add_eos_to_blank = config.bool("add_eos_to_blank", False)

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)

    ilm_out = model.ilm_forward(targets, targets_spatial_dim)
    ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor.detach() # (B, T, V + blank), good
    
    
    # expect the kldiv in the beginning to be near -ln(1/10026) ~ 9.x
    log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, S, V)

    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor
    
    assert masking_rate is not None, "Must provide input_masking_rate in config"
    kldiv = kldiv_ctc_lm_loss(
        aux_logits_raw.transpose(0, 1).log_softmax(-1),
        targets_raw.clone().long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        ilm_out_raw.transpose(0, 1).log_softmax(-1),
        blank_idx=10025,
        eos_idx=model.eos_idx,
        target_mask=label_loss_mask,
    ) # no need to normalize this loss when passing to returnn!!!
    loss = kldiv / label_loss_mask.sum()
    targets_len_rf = targets_spatial_dim_pad[0].dyn_size_ext
    rf.get_run_ctx().mark_as_loss(
        loss,
        "kldiv_masking",
    )

    # Also report PPL of the LM
    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(torch_target_lengths, max_seq_len+1, targets_raw.device)
    log_ppl = (ce*seq_mask).sum()/(targets_len_rf.raw_tensor.sum())
    ppl = torch.exp(log_ppl)
    rf.get_run_ctx().mark_as_loss(
        name="student_lm_ppl", loss=ppl, as_error=True,
    )


from_scratch_training_kldiv_masking: TrainDef[Model]
from_scratch_training_kldiv_masking.learning_rate_control_error_measure = "dev_score_full_sum"


from i6_experiments.users.phan.ctc_ilm_sequence_level_loss import kldiv_ctc_lm_sequence_level, \
    kldiv_ctc_lm_sequence_level_ground_truth_weight_1
def ilm_kldiv_sequence_level(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Like KLDiv but on sequence level"""
    from returnn.config import get_global_config

    # import for training only, will fail on CPU servers
    # from i6_native_ops import warp_rnnt

    config = get_global_config()  # noqa

    # sampling weight given to the ground truth
    weight = config.typed_value("sequence_sampling_weight", None)
    assert weight is not None, "Must provide sequence_sampling_weight in config"

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

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)



    ######### Needed stuffs for ILM estimation ##########
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        targets,
        padding=[(1, 0)],
        axes=[targets_spatial_dim],
        value=model.eos_idx
    )
    ilm_out = model.ilm_forward(targets_w_bos, targets_spatial_dim_pad[0])
    ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor.detach() # (B, T, V + blank), good
    targets_raw = targets.raw_tensor
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor


    log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, S, V)
    targets_len_rf = targets_spatial_dim_pad[0].dyn_size_ext


    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(torch_target_lengths, max_seq_len+1, targets_raw.device)
    ce = ce*seq_mask
    

    # report the PPL of the ILM
    log_ppl = ce.sum()/(targets_len_rf.raw_tensor.sum())
    ppl = torch.exp(log_ppl)
    rf.get_run_ctx().mark_as_loss(
        name="student_lm_ppl", loss=ppl, as_error=True,
    )

    # calculate the sequence KL divergence
    log_lm_seq_probs = -ce.sum(1) # (B,), ce[b] = log p_ILM(w_1^N), sequence level
    if weight != 1.0:
        kldiv = kldiv_ctc_lm_sequence_level(
            aux_logits_raw.transpose(0, 1).log_softmax(-1),
            targets_raw.clone().long(),
            torch_input_lengths.long(),
            torch_target_lengths.long(),
            log_lm_seq_probs,
            blank_idx=model.blank_idx,
            ground_truth_weight=weight,
        ) # do we normalize this ? no
    else:
        kldiv = kldiv_ctc_lm_sequence_level_ground_truth_weight_1(
            aux_logits_raw.transpose(0, 1).log_softmax(-1),
            targets_raw.clone().long(),
            torch_input_lengths.long(),
            torch_target_lengths.long(),
            log_lm_seq_probs,
            blank_idx=model.blank_idx,
        ) # do we normalize this ? no
    rf.get_run_ctx().mark_as_loss(
        kldiv,
        "kldiv_sequence_level",
    )

ilm_kldiv_sequence_level: TrainDef[Model]
ilm_kldiv_sequence_level.learning_rate_control_error_measure = "dev_score_full_sum"
