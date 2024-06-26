from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
    Tuple,
    Sequence,
    List,
    Collection,
    Dict,
)
import tree
import math
import numpy as np
import torch
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsampleV2

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import (
    trafo_lm_kazuki_import,
)

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

_log_mel_feature_dim = 80


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers: int = 12,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers)

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


class MakeModelV2:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers: int = 12,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers)

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
                conv_norm = rf.LayerNorm,
                # conv_norm_opts=dict(
                #     in_dim=
                # ),  # Changed below
                self_att=rf.SelfAttention,
                self_att_opts=dict(
                    with_bias=True,  # Changed: with_bias=True
                    # with_linear_pos=False,
                    # with_pos_bias=False,
                    # learnable_pos_emb=False,  # Changed: learnable_pos_emb=False
                    # separate_pos_emb_per_head=False,
                    # pos_emb_dropout=pos_emb_dropout,
                ),
                ff_activation=rf.silu,  # Changed: rf.silu
                conv_kernel_size=31,  # Changed: conv_kernel_size=31
            ),
            enc_input_layer=ConformerConvSubsampleV2(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                    Dim(32, name="conv4"),  # Changed: Dim(64, name="conv4")
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],  # Changed
                activation_times=[False, True, False, True],  # Changed
                pool_sizes=[(1, 1), (3, 1), (1, 1), (2, 1)],  # Changed
                strides=[(1, 1), (1, 1), (1, 1), (1, 1)],  # Changed
                padding="same",  # Changed: padding="valid"
                pool_padding="valid",  # Changed
                swap_merge_dim_order=True,  # Changed
                # Note: uses relu activation by default
            ),
            enc_use_input_proj_bias=True,  # Changed: enc_use_input_proj_bias=True
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            language_model=lm,
            use_i6_models_feat_ext = True, # Changed
            lstm_biases = True, # Changed
            # feat_ext_opts=dict(
            #     f_min=60,
            #     f_max=7600,
            #     n_fft=400,
            # ),
            **extra,
        )


class Predictor(rf.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    From torchaudio, modified to be used with rf
    """

    def __init__(
        self,
        # cfg: PredictorConfig,
        label_target_size: Dim,
        output_dim: Dim,
        symbol_embedding_dim: int = 256,
        emebdding_dropout: float = 0.2,
        num_lstm_layers: int = 1,
        lstm_hidden_dim: int = 512,
        lstm_dropout: float = 0.1,
        lstm_biases: bool = False,
    ) -> None:
        """

        :param cfg: model configuration for the predictor
        :param label_target_size: shared value from model
        :param output_dim: shared value from model
        """
        super().__init__()

        self.label_target_size = label_target_size
        self.output_dim = output_dim
        self.embedding_dropout = emebdding_dropout
        self.lstm_dropout = lstm_dropout
        self.num_lstm_layers = num_lstm_layers

        self.symbol_embedding_dim = Dim(
            name="symbol_embedding", dimension=symbol_embedding_dim
        )
        self.lstm_hidden_dim = Dim(name="lstm_hidden", dimension=lstm_hidden_dim)

        self.embedding = rf.Embedding(label_target_size, self.symbol_embedding_dim)
        self.input_layer_norm = rf.LayerNorm(self.symbol_embedding_dim)

        self.layers = rf.Sequential(
            rf.LSTM(
                self.symbol_embedding_dim if idx == 0 else self.lstm_hidden_dim,
                self.lstm_hidden_dim,
                with_bias_rec=lstm_biases,
            )
            for idx in range(self.num_lstm_layers)
        )

        # self.lstm_layers = torch.nn.ModuleList(
        #     [
        #         nn.LSTM(
        #             input_size=symbol_embedding_dim
        #             if idx == 0
        #             else lstm_hidden_dim,
        #             hidden_size=lstm_hidden_dim,
        #         )
        #         for idx in range(num_lstm_layers)
        #     ]
        # )
        self.linear = rf.Linear(self.lstm_hidden_dim, output_dim)
        self.output_layer_norm = rf.LayerNorm(output_dim)

    def default_initial_state(
        self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
    ) -> rf.State:
        """default initial state"""
        state = rf.State(
            {
                k: v.default_initial_state(batch_dims=batch_dims)
                for k, v in self.layers.items()
            }
        )

        return state

    def __call__(
        self,
        input: rf.Tensor,
        # lengths: torch.Tensor,
        state: Optional[rf.State] = None,
        spatial_dim: Dim = single_step_dim,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        embedding_out = self.embedding(input)
        embedding_out = rf.dropout(
            embedding_out,
            drop_prob=self.embedding_dropout,
            axis=embedding_out.feature_dim,
        )
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out

        if state is None:
            state = self.default_initial_state(batch_dims=input.dims[:-1])

        new_state = rf.State()

        for layer_name, layer in self.layers.items():
            layer: rf.LSTM  # or similar
            # if layer_name in ["0"]: # "0"
            #     breakpoint()
            lstm_out, new_state[layer_name] = layer(
                lstm_out, spatial_dim=spatial_dim, state=state[layer_name]
            )
            # if collected_outputs is not None:
            #     collected_outputs[layer_name] = decoded

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out, new_state


class Joiner(rf.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(
        self,
        input_dim: Dim,
        output_dim: Dim,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = rf.Linear(self.input_dim, self.output_dim)
        self.dropout = dropout

        if activation == "relu":
            self.activation = rf.relu
        elif activation == "tanh":
            self.activation = rf.tanh
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def __call__(
        self,
        source_encodings: rf.Tensor,
        # source_lengths: rf.Tensor,
        target_encodings: rf.Tensor,
        # target_lengths: rf.Tensor,
        batch_dims: Sequence[Dim],
    ) -> Tuple[rf.Tensor, rf.Tensor, rf.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """

        time_axis = len(batch_dims)

        joint_encodings_raw = (
            source_encodings.raw_tensor.unsqueeze(time_axis + 1).contiguous()
            + target_encodings.raw_tensor.unsqueeze(time_axis).contiguous()
        )

        joint_encodings = rf.Tensor(
            name="joint_encodings",
            raw_tensor=joint_encodings_raw,
            dims=batch_dims
            + [
                source_encodings.dims[time_axis],  # T
                target_encodings.dims[time_axis],  # U
                source_encodings.dims[-1],  # F
            ],
            dtype=source_encodings.dtype,
        )

        # joint_encodings = rf.copy_to_device(joint_encodings, "cuda")

        joint_encodings = rf.dropout(
            joint_encodings, drop_prob=self.dropout, axis=joint_encodings.feature_dim
        )
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output  # source_lengths, target_lengths


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
        enc_input_layer: Optional[ConformerConvSubsampleV2] = None,
        enc_use_input_proj_bias: bool = False,
        # enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        # att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        # att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
        language_model: Optional[RFModelWithMakeLabelScorer] = None,
        joiner_dim: int = 640,
        use_i6_models_feat_ext: bool = False,
        feat_ext_opts: Optional[Dict[str, Any]] = None,
        lstm_biases: bool = False,
        loss_type: str = "rnnt",
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.mel_normalization = config.typed_value("mel_normalization_ted2", True)
        self.use_i6_models_feat_ext = use_i6_models_feat_ext
        if self.use_i6_models_feat_ext:
            from i6_models.primitives.feature_extraction import (
                LogMelFeatureExtractionV1,
                LogMelFeatureExtractionV1Config,
            )

            mel_config = LogMelFeatureExtractionV1Config(
                sample_rate=16000,
                win_size=0.025,
                hop_size=0.01,
                f_min=60,
                f_max=7600,
                min_amp=1e-10,
                num_filters=80,
                center=False,
                **(feat_ext_opts or {}),
            )
            self.feature_extraction = LogMelFeatureExtractionV1(cfg=mel_config)

        self.feat_ext_opts = feat_ext_opts

        if enc_input_layer is None:
            self.enc_input_layer = ConformerConvSubsampleV2(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            )
        else:
            self.enc_input_layer = enc_input_layer

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=self.enc_input_layer,
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.enc_use_input_proj_bias = enc_use_input_proj_bias

        if self.enc_use_input_proj_bias:
            self.encoder.input_projection = rf.Linear(
                self.encoder.input_layer.out_dim
                if self.encoder.input_layer
                else self.encoder.in_dim,
                self.encoder.out_dim,
                with_bias=True,
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

        self.joiner_dim = Dim(name="joiner", dimension=joiner_dim)

        self.joiner = Joiner(self.joiner_dim, self.target_dim_w_blank)
        self.encoder_out_linear = rf.Linear(self.encoder.out_dim, self.joiner_dim)

        self.predictor = Predictor(
            label_target_size=self.target_dim_w_blank,
            output_dim=self.joiner_dim,
            lstm_biases=lstm_biases,
        )

        self.loss_type = loss_type

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

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""

        breakpoint()
        if self.use_i6_models_feat_ext:
            orig_device = source.device
            squeezed_features = torch.squeeze(source.raw_tensor)
            squeezed_features = squeezed_features.to("cpu")
            raw_audio_len = in_spatial_dim.dyn_size_ext.raw_tensor
            audio_features, audio_features_len_raw = self.feature_extraction(
                squeezed_features, raw_audio_len
            )
            audio_features_len = rf.Tensor(
                name="audio-features-len",
                dims=[source.dims[0]],
                raw_tensor=audio_features_len_raw,
                dtype="int32",
            )
            in_spatial_dim = Dim(None, name="in-spatial-dim", dyn_size_ext=audio_features_len)
            source = rf.Tensor(
                name="audio-features",
                dims=[source.dims[0], in_spatial_dim, self.in_dim],
                raw_tensor=audio_features,
                dtype=source.dtype,
            )
            source = rf.copy_to_device(source, orig_device)
        else:
            # log mel filterbank features
            source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw_v2(
                source,
                in_spatial_dim=in_spatial_dim,
                out_dim=self.in_dim,
                sampling_rate=16_000,
                log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
                **(self.feat_ext_opts or {}),
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

    def decoder_default_initial_state(
        self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim
    ) -> rf.State:
        """Default initial state"""
        state = rf.State(
            predictor=self.predictor.default_initial_state(batch_dims=batch_dims),
            # s=self.s.default_initial_state(batch_dims=batch_dims),
            # att=rf.zeros(
            #     list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]
            # ),
            # accum_att_weights=rf.zeros(
            #     list(batch_dims) + [enc_spatial_dim, self.att_num_heads],
            #     feature_dim=self.att_num_heads,
            # ),
        )
        # state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    # def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    #     """loop step out"""
    #     return {
    #         "pred_lstm": Tensor(
    #             "pred_lstm",
    #             dims=batch_dims + [self.pred_lstm.out_dim],
    #             dtype=rf.get_default_float_dtype(),
    #             feature_dim_axis=-1,
    #         ),
    #         # "s": Tensor(
    #         #     "s",
    #         #     dims=batch_dims + [self.s.out_dim],
    #         #     dtype=rf.get_default_float_dtype(),
    #         #     feature_dim_axis=-1,
    #         # ),
    #         # "att": Tensor(
    #         #     "att",
    #         #     dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
    #         #     dtype=rf.get_default_float_dtype(),
    #         #     feature_dim_axis=-1,
    #         # ),
    #     } # TODO

    def loop_step(
        self,
        *,
        enc: rf.Tensor,
        # enc_ctx: rf.Tensor,
        # inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        target: rf.Tensor,
        target_spatial_dim: Dim,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        batch_dims = enc.remaining_dims(
            remove=(enc.feature_dim, enc_spatial_dim)
            if enc_spatial_dim != single_step_dim
            else (enc.feature_dim,)
        )

        if state is None:
            state = self.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            )
        state_ = rf.State()
        enc_lin = self.encoder_out_linear(enc)

        pred_lstm, state_.predictor = self.predictor(
            target, state.predictor, spatial_dim=target_spatial_dim
        )

        pred_out = pred_lstm.copy_swap_axes(0, 1)

        joiner = self.joiner(enc_lin, pred_out, batch_dims=batch_dims)

        return {"output": joiner}, state_

    # def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    #     """logits for the decoder"""
    #     readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
    #     readout = rf.reduce_out(
    #         readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim
    #     )
    #     readout = rf.dropout(
    #         readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim
    #     )
    #     logits = self.output_prob(readout)
    #     return logits


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


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    lm_opts = config.typed_value("external_language_model")
    return MakeModel.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        language_model=lm_opts,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = 160


def from_scratch_model_def_v2(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    lm_opts = config.typed_value("external_language_model")
    return MakeModelV2.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        language_model=lm_opts,
    )


from_scratch_model_def_v2: ModelDef[Model]
from_scratch_model_def_v2.behavior_version = 16
from_scratch_model_def_v2.backend = "torch"
from_scratch_model_def_v2.batch_size_factor = 160


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
    from i6_native_ops import warp_rnnt

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

    targets_mod = targets.copy()
    targets_mod.sparse_dim = model.target_dim_w_blank
    blanks = rf.expand_dim(
        rf.full(
            dims=targets_mod.dims[:-1],
            fill_value=model.blank_idx,
            dtype=targets_mod.dtype,
        ),
        Dim(1),
    )
    blanks.sparse_dim = model.target_dim_w_blank

    targets_mod, targets_spatial_dim = rf.concat(
        (blanks, blanks.dims[1]), (targets_mod, targets.dims[1])
    )

    step_out, _ = model.loop_step(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        target=targets_mod,
        target_spatial_dim=targets_spatial_dim,
    )

    logits = step_out["output"]

    logprobs = rf.log_softmax(logits, axis=model.target_dim_w_blank)

    # input_embeddings = model.target_embed(targets)
    # input_embeddings = rf.shift_right(
    #     input_embeddings, axis=targets_spatial_dim, pad_value=0.0
    # )

    labels_len = rf.copy_to_device(targets_spatial_dim.get_size_tensor(), "cuda")
    frames_len = rf.copy_to_device(enc_spatial_dim.get_size_tensor(), "cuda")

    if model.loss_type == "monotonic_rnnt":
        from returnn.extern_private.BergerMonotonicRNNT.monotonic_rnnt.pytorch_binding import monotonic_rnnt_loss

        loss = monotonic_rnnt_loss(
            acts=logprobs.raw_tensor,
            labels=targets.raw_tensor,
            input_lengths=frames_len.raw_tensor,
            label_lengths=labels_len.raw_tensor,
            blank_label=model.blank_idx,
        )

        loss = rf.convert_to_tensor(loss, name="full_sum_loss")
        loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)
    else:
        rnnt_loss_raw = warp_rnnt.rnnt_loss(
            log_probs=logprobs.raw_tensor,
            frames_lengths=frames_len.raw_tensor,
            labels=targets.raw_tensor,
            labels_lengths=labels_len.raw_tensor,
            blank=model.blank_idx,
            fastemit_lambda=0.0,
            reduction="sum",
            gather=True,
        )

        rnnt_loss = Tensor(
            name="rnnt_loss",
            dims=[Dim(name="rnnt_loss_dim", dimension=1)],
            raw_tensor=rnnt_loss_raw.unsqueeze(0),
            dtype=logprobs.dtype,
        )

        rnnt_loss.mark_as_loss(
            name="rnnt",
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )

    # def _body(input_embed: Tensor, state: rf.State):
    #     new_state = rf.State()
    #     loop_out_, new_state.decoder = model.loop_step(
    #         **enc_args,
    #         enc_spatial_dim=enc_spatial_dim,
    #         input_embed=input_embed,
    #         state=state.decoder,
    #     )
    #     return loop_out_, new_state
    #
    # loop_out, _, _ = rf.scan(
    #     spatial_dim=targets_spatial_dim,
    #     xs=input_embeddings,
    #     ys=model.loop_step_output_templates(batch_dims=batch_dims),
    #     initial=rf.State(
    #         decoder=model.decoder_default_initial_state(
    #             batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
    #         ),
    #     ),
    #     body=_body,
    # )

    # logits = model.decode_logits(input_embed=input_embeddings, **loop_out)
    # logits_packed, pack_dim = rf.pack_padded(
    #     logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False
    # )
    # targets_packed, _ = rf.pack_padded(
    #     targets,
    #     dims=batch_dims + [targets_spatial_dim],
    #     enforce_sorted=False,
    #     out_dim=pack_dim,
    # )
    #
    # log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    # loss = rf.cross_entropy(
    #     target=targets_packed,
    #     estimated=log_prob,
    #     estimated_type="log-probs",
    #     axis=model.target_dim,
    # )
    # loss.mark_as_loss(
    #     "ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss
    # )

    # best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    # frame_error = best != targets_packed
    # frame_error.mark_as_loss(name="fer", as_error=True)


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
