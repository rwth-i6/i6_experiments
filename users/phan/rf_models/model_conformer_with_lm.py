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
from i6_experiments.users.phan.rf_models.trafo_lm_luca import Trafo_LM_Model
from i6_experiments.users.yang.torch.loss.ctc_forward_backward import ctc_forward
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import kldiv_ctc_lm_loss, kldiv_ctc_lm_sample_batch_loss
from i6_experiments.users.phan.utils.masking import get_seq_mask

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
        log_prob_output=False,
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
        external_language_model: Optional[dict] = None, # for recog only
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
            external_language_model=external_language_model,
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
        joiner_dim: int = 640,
        freeze_encoder: bool = True,
        external_language_model: Optional[Dict] = None, # for recog
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.mel_normalization = config.typed_value("mel_normalization_ted2", False)
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
        self.ilm = LSTMLMRF( # better to hardcode to make sure all ILM share the same hyperparams
            label_target_dim,
            label_target_dim,
            symbol_embedding_dim = 512,
            emebdding_dropout = 0.0,
            num_lstm_layers = 2,
            lstm_hidden_dim = 2048,
            lstm_dropout = 0.0,
            use_bottleneck = False,
            bottleneck_dim = 512,
        )

        if external_language_model is not None:
            lm_cls = external_language_model.pop("class")
            if lm_cls == "Trafo_LM_Model":
                self.language_model = Trafo_LM_Model(
                    label_target_dim,
                    label_target_dim,
                    **external_language_model,
                )
            else:
                raise NotImplementedError("Only the Kazuki Trafo LM is supported!!!!!!!!")

        # Freeze the encoder
        for name, param in self.encoder.named_parameters():
            param.trainable = False
    
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
    external_language_model = config.typed_value("external_language_model")
    return MakeModel.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        external_language_model=external_language_model
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 21 #16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = 160



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
    kldiv = kldiv_ctc_lm_loss(
        aux_logits_raw.transpose(0, 1).log_softmax(-1).detach(),
        targets_raw.clone().long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        ilm_out_raw.transpose(0, 1).log_softmax(-1),
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
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
        aux_logits_raw.transpose(0, 1).log_softmax(-1).detach(),
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

