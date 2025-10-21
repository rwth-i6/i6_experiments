"""
Chunked CTC / Conformer.

For earlier code and some reference, see:
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_conformer.py
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_ctc.py
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_aed_import.py

Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition, https://arxiv.org/abs/2309.08436
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union
import copy as _copy

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
)

from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

__setup_root_prefix__ = "exp2025_10_21_chunked_ctc"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    task_spm10k = get_loquacious_task_raw_v2(vocab="spm10k")

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    for subset, total_k_hours in [
        ("large", 100),  # 100kh in total, 4 full epochs
        # ("large", 150),  # 150kh in total, 6 full epochs
        # ("large", 200),  # 200kh in total, 8 full epochs
        # ("large", 250),  # 250kh in total, 10 full epochs
        # ("large", 500),  # 500kh in total, 20 full epochs
    ]:
        train_epoch_split = train_epoch_split_per_subset[subset]
        num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
        n_ep = round(num_full_ep * train_epoch_split)
        name = f"base-{subset}-nFullEp{num_full_ep:.1f}-nEp{n_ep}-totalHours{total_k_hours}k"
        exp = aed_train_exp(
            name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            task=get_loquacious_task_raw_v2(vocab="spm10k", subset_name=subset, train_epoch_split=train_epoch_split),
            model_config={
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                # "__train_audio_preprocess": speed_pert_librosa_config,
                # "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                "dec_aux_loss_layers": [3],
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
        )

    subset = "large"
    total_k_hours = 100
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)
    name = "chunked"
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        task=get_loquacious_task_raw_v2(vocab="spm10k", subset_name=subset, train_epoch_split=train_epoch_split),
        model_config={
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ChunkedConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ChunkedConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            # "__train_audio_preprocess": speed_pert_librosa_config,
            # "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )


class ChunkedConformerConvBlock(rf.Module):
    """
    Conformer convolution block
        FF -> GLU -> depthwise conv -> BN -> Swish -> FF
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        kernel_size: int,
        norm: Union[rf.BatchNorm, Any],
        chunk_history: int,
        end_chunk_size_dim: Dim,
    ):
        """
        :param out_dim: output feature dimension
        :param kernel_size: kernel size of depthwise convolution
        :param norm: Batch norm originally
        :param chunk_history:
        :param end_chunk_size_dim:
        """
        super().__init__()
        self.out_dim = out_dim

        self.positionwise_conv1 = rf.Linear(out_dim, 2 * out_dim)
        self.depthwise_conv = rf.Conv1d(
            out_dim, out_dim, filter_size=kernel_size, groups=out_dim.dimension, padding="same"
        )
        self.positionwise_conv2 = rf.Linear(out_dim, out_dim)
        self.norm = norm

        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunked_time_dim: Dim) -> Tensor:
        """forward"""
        x_conv1 = self.positionwise_conv1(inp)
        x_act, _ = rf.gating(x_conv1)
        x_act, ext_spatial_dim = _mem_chunks(
            x_act,
            spatial_dim=spatial_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=self.chunk_history,
            end_chunk_size_dim=self.end_chunk_size_dim,
        )
        x_depthwise_conv, _ = self.depthwise_conv(x_act, in_spatial_dim=ext_spatial_dim)
        x_depthwise_conv, _ = rf.slice(
            x_depthwise_conv,
            axis=ext_spatial_dim,
            start=self.chunk_history * self.end_chunk_size_dim.get_dim_value_tensor(),
            out_dim=spatial_dim,
        )
        x_normed = self.norm(x_depthwise_conv)
        x_swish = rf.swish(x_normed)
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2


class ChunkedConformerEncoderLayer(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        chunk_history: int,
        end_chunk_size_dim: Dim,
        ff_dim: Dim = NotSpecified,
        ff_activation: Callable[[Tensor], Tensor] = rf.swish,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        conv_norm_opts: Optional[Dict[str, Any]] = None,
        num_heads: int = 4,
        self_att: Optional[Union[rf.RelPosSelfAttention, rf.Module, type, Any]] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.1,
    ):
        """
        :param out_dim: the output feature dimension
        :param chunk_history:
        :param end_chunk_size_dim:
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
        :param conv_norm: used for the conv block. Batch norm originally
        :param conv_norm_opts: for nn.BatchNorm or other conv_norm type.
          In case of nn.BatchNorm, uses use_mask=False by default.
            use_mask means whether to properly mask the spatial dim in batch norm.
            Most existing implementations don't do this. Except of RETURNN.
            It's faster when you don't do this.
        :param num_heads: the number of attention heads
        :param self_att: the self-attention layer. RelPosSelfAttention originally and default
        :param self_att_opts: options for the self-attention layer, for :class:`nn.RelPosSelfAttention`
        :param att_dropout: attention dropout value
        """
        super().__init__()

        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

        if ff_dim is None:
            ff_dim = 4 * out_dim
        self.ffn1 = ConformerPositionwiseFeedForward(
            out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation
        )
        self.ffn1_layer_norm = rf.LayerNorm(out_dim)

        self.ffn2 = ConformerPositionwiseFeedForward(
            out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation
        )
        self.ffn2_layer_norm = rf.LayerNorm(out_dim)

        if conv_norm is NotSpecified or conv_norm is rf.BatchNorm:
            conv_norm_opts = conv_norm_opts.copy() if conv_norm_opts else {}
            conv_norm_opts.setdefault("use_mask", False)
            conv_norm = rf.BatchNorm(out_dim, **conv_norm_opts)
        elif isinstance(conv_norm, type):
            conv_norm = conv_norm(out_dim, **(conv_norm_opts or {}))
        self.conv_block = ChunkedConformerConvBlock(
            out_dim=out_dim,
            kernel_size=conv_kernel_size,
            norm=conv_norm,
            chunk_history=chunk_history,
            end_chunk_size_dim=end_chunk_size_dim,
        )
        self.conv_layer_norm = rf.LayerNorm(out_dim)

        if self_att is None or isinstance(self_att, type):
            self_att_opts_ = dict(
                in_dim=out_dim,
                proj_dim=out_dim,
                key_dim_total=out_dim,
                value_dim_total=out_dim,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            if self_att_opts:
                self_att_opts_.update(self_att_opts)
            if self_att is None:
                self.self_att = ChunkedRelPosSelfAttention(
                    chunk_history=chunk_history, end_chunk_size_dim=end_chunk_size_dim, **self_att_opts_
                )
            else:
                self.self_att = self_att(**self_att_opts_)
        else:
            self.self_att = self_att
        self.self_att_layer_norm = rf.LayerNorm(out_dim)

        self.final_layer_norm = rf.LayerNorm(out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunked_time_dim: Dim) -> Tensor:
        """forward"""
        # FFN
        x_ffn1_ln = self.ffn1_layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

        # MHSA
        x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim, chunked_time_dim=chunked_time_dim)
        x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x_mhsa_out = x_mhsa + x_ffn1_out

        # Conv
        x_conv_ln = self.conv_layer_norm(x_mhsa_out)
        x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim, chunked_time_dim=chunked_time_dim)
        x_conv_out = rf.dropout(x_conv, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

        # FFN
        x_ffn2_ln = self.ffn2_layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_conv_out

        # last LN layer
        return self.final_layer_norm(x_ffn2_out)


class ChunkedConformerEncoder(rf.Module):
    """
    Represents Conformer encoder architecture
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        num_layers: int,
        input_layer: Union[ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Any],
        input_dropout: float = 0.1,
        ff_dim: Dim = NotSpecified,
        ff_activation: Callable[[Tensor], Tensor] = rf.swish,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Any] = NotSpecified,
        num_heads: int = 4,
        att_dropout: float = 0.1,
        encoder_layer: Optional[Union[ChunkedConformerEncoderLayer, rf.Module, type, Any]] = None,
        encoder_layer_opts: Optional[Dict[str, Any]] = None,
        input_chunk_size_dim: Union[int, Dim],
        chunk_stride: int,
        chunk_history: int,
        end_chunk_size_dim: Union[int, Dim],
    ):
        """
        :param out_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param input_layer: input/frontend/prenet with potential subsampling.
            (x, in_spatial_dim) -> (y, out_spatial_dim)
        :param input_dropout: applied after input_projection(input_layer(x))
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
        :param conv_norm: used for the conv block. Batch norm originally
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param encoder_layer: an instance of :class:`ConformerEncoderLayer` or similar
        :param encoder_layer_opts: options for the encoder layer
        :param input_chunk_size_dim:
        :param chunk_stride:
        :param chunk_history:
        :param end_chunk_size_dim:
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        if isinstance(input_chunk_size_dim, int):
            input_chunk_size_dim = Dim(input_chunk_size_dim, name="input_chunk_size")
        if isinstance(end_chunk_size_dim, int):
            end_chunk_size_dim = Dim(end_chunk_size_dim, name="end_chunk_size")

        self.input_chunk_size_dim = input_chunk_size_dim
        self.chunk_stride = chunk_stride
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

        if isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim)
            input_layer: ConformerConvSubsample  # maybe not true, but assume for some attribs

        self.input_layer = input_layer
        self.input_projection = rf.Linear(
            self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False
        )
        self.input_dropout = input_dropout

        if not encoder_layer or isinstance(encoder_layer, type):
            encoder_layer_opts_ = dict(
                out_dim=out_dim,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
                conv_norm=conv_norm,
                num_heads=num_heads,
                att_dropout=att_dropout,
                chunk_history=chunk_history,
                end_chunk_size_dim=end_chunk_size_dim,
            )
            if encoder_layer_opts:
                encoder_layer_opts_.update(encoder_layer_opts)
            if not encoder_layer:
                encoder_layer = ChunkedConformerEncoderLayer(**encoder_layer_opts_)
            elif isinstance(encoder_layer, type):
                encoder_layer = encoder_layer(**encoder_layer_opts_)
            else:
                raise TypeError(f"unexpected encoder_layer {encoder_layer!r}")

        self.layers = rf.Sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        """forward"""
        # Chunk
        source, chunked_time_dim = rf.window(
            source,
            spatial_dim=in_spatial_dim,
            window_dim=self.input_chunk_size_dim,
            window_left=0,
            stride=self.chunk_stride,
        )

        if self.input_layer:
            x_subsample, chunk_size_dim = self.input_layer(source, in_spatial_dim=self.input_chunk_size_dim)
        else:
            x_subsample, chunk_size_dim = source, self.input_chunk_size_dim
        x_linear = self.input_projection(x_subsample)

        x = rf.dropout(x_linear, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)
        x = self.layers(
            x,
            spatial_dim=chunk_size_dim,
            chunked_time_dim=chunked_time_dim,
            collected_outputs=collected_outputs,
        )

        # Unchunk
        x, _ = rf.slice(x, axis=chunk_size_dim, size=self.end_chunk_size_dim)
        x, out_spatial_dim_ = rf.merge_dims(x, dims=(chunked_time_dim, self.end_chunk_size_dim))

        if collected_outputs:
            for k, v in list(collected_outputs.items()):
                v, _ = rf.slice(v, axis=chunk_size_dim, size=self.end_chunk_size_dim)
                v, _ = rf.merge_dims(v, dims=(chunked_time_dim, self.end_chunk_size_dim), out_dim=out_spatial_dim_)
                collected_outputs[k] = v

        return x, out_spatial_dim_


class ChunkedRelPosSelfAttention(rf.RelPosSelfAttention):
    def __init__(self, *, chunk_history: int, end_chunk_size_dim: Dim, **kwargs):
        super().__init__(**kwargs)
        self.chunk_history = chunk_history
        self.end_chunk_size_dim = end_chunk_size_dim

    def __call__(self, source: Tensor, *, axis: Dim, chunked_time_dim: Dim, **_kwargs) -> Tensor:
        """forward"""
        q, k, v = self.forward_qkv(source)
        hist_dim = Dim(None, name=f"{axis.description}:kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=hist_dim)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=hist_dim)
        k, hist_dim_ = _mem_chunks(
            k,
            spatial_dim=hist_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=self.chunk_history,
            end_chunk_size_dim=self.end_chunk_size_dim,
        )
        v, _ = _mem_chunks(
            v,
            spatial_dim=hist_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=self.chunk_history,
            end_chunk_size_dim=self.end_chunk_size_dim,
            out_spatial_dim=hist_dim_,
        )
        q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

        # NOTE: This changed from the earlier RF/TF implementation.
        query_offset = self.chunk_history * (
            self.end_chunk_size_dim.dimension
            if self.end_chunk_size_dim.is_static()
            else self.end_chunk_size_dim.get_size_tensor(device=source.device)
        )

        if self.learned_pos_emb is not None:
            pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(
                query_spatial_dim=axis, key_value_spatial_dim=hist_dim_, query_offset=query_offset
            )
        else:
            pos_emb, pos_emb_spatial_dim = rf.relative_positional_encoding(
                query_spatial_dim=axis,
                key_value_spatial_dim=hist_dim_,
                feat_dim=self.pos_emb_feat_dim,
                query_offset=query_offset,
            )
        if self.pos_emb_dropout:
            pos_emb = rf.dropout(pos_emb, self.pos_emb_dropout)
        if self.linear_pos is not None:
            pos_emb = self.linear_pos(pos_emb)
        if self.separate_pos_emb_per_head:
            pos_emb = rf.split_dims(pos_emb, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        # pos_emb: (head, 2*time1-1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = rf.matmul(q_with_bias_u, k, reduce=self.key_dim_per_head)

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
        matrix_bd = self._rel_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim_)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = rf.softmax(scores, axis=hist_dim_)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=self.att_dropout_broadcast and hist_dim_)
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=hist_dim_, use_mask=False)
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output


def _mem_chunks(
    source: rf.Tensor,
    *,
    spatial_dim: Dim,
    chunked_time_dim: Dim,
    mem_size: int,
    end_chunk_size_dim: Dim,
    out_spatial_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    Concat the prev chunks to the current chunk, i.e. add history / memory.

    :param source: (batch..., chunked_time, spatial_dim=chunk_size, feat)
    :param spatial_dim: chunk size / window size
    :param chunked_time_dim: the chunks
    :param mem_size: how many previous chunks to concat
    :param end_chunk_size_dim: ...?
    :param out_spatial_dim: if given, use this as output spatial dim
    :return: concatenated prev chunks, concatenated spatial dim
    """
    concats = []
    source_sliced, _ = rf.slice(source, axis=spatial_dim, size=end_chunk_size_dim)
    for shift_amount in range(mem_size, 0, -1):
        shifted = rf.shift_right(source_sliced, axis=chunked_time_dim, amount=shift_amount, pad_value=0.0)
        concats.append((shifted, end_chunk_size_dim))
    concats.append((source, spatial_dim))
    return rf.concat(*concats, out_dim=out_spatial_dim)
