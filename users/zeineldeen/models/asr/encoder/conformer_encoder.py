"""
Conformer encoder
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from returnn.tensor import Dim
from returnn.tf.util.data import SpatialDim, FeatureDim
from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork


class ConformerEncoder:
    """
    Represents Conformer Encoder Architecture

    * Conformer: Convolution-augmented Transformer for Speech Recognition
    * Ref: https://arxiv.org/abs/2005.08100
    """

    def __init__(
        self,
        input="data",
        input_layer="lstm-6",
        input_layer_conv_act="relu",
        add_abs_pos_enc_to_input=False,
        frontend_conv_l2=0.0,
        num_blocks=16,
        conv_kernel_size=32,
        specaug=True,
        pos_enc="rel",
        activation="swish",
        block_final_norm=True,
        ff_dim=512,
        ff_bias=True,
        ctc_loss_scale=None,
        dropout=0.1,
        att_dropout=0.1,
        enc_key_dim=256,
        att_num_heads=4,
        target="bpe",
        l2=0.0,
        lstm_dropout=0.1,
        rec_weight_dropout=0.0,
        with_ctc=False,
        native_ctc=False,
        ctc_dropout=0.0,
        ctc_l2=0.0,
        ctc_opts=None,
        ctc_self_align_delay: Optional[int] = None,
        ctc_self_align_scale: float = 0.5,
        subsample=None,
        start_conv_init=None,
        conv_module_init=None,
        mhsa_init=None,
        mhsa_out_init=None,
        ff_init=None,
        rel_pos_clipping=16,
        dropout_in=0.1,
        batch_norm_opts=None,
        use_ln=False,
        pooling_str=None,
        self_att_l2=0.0,
        sandwich_conv=False,
        add_to_prefix_name=None,
        output_layer_name="encoder",
        create_only_blocks=False,
        no_mhsa_module=False,
        proj_input=False,
        use_sqrd_relu=False,
        use_causal_layers=False,
        use_causal_conv=None,
        conv_alternative_name: Optional[str] = None,
        fix_merge_dims=False,
        weight_noise=None,
        weight_noise_layers=None,
        convolution_first=False,
        ff_weight_dropout=None,
        mhsa_weight_dropout=None,
        conv_weight_dropout=None,
        memory_variant_opts: Optional[ConformerMemoryVariantOpts] = None,
    ):
        """
        :param str input: input layer name
        :param str|None input_layer: type of input layer which does subsampling
        :param int num_blocks: number of Conformer blocks
        :param int conv_kernel_size: kernel size for conv layers in Convolution module
        :param bool|None specaug: If true, then SpecAug is appliedi wi
        :param str|None activation: activation used to sandwich modules
        :param bool block_final_norm: if True, apply layer norm at the end of each conformer block
        :param bool final_norm: if True, apply layer norm to the output of the encoder
        :param int|None ff_dim: dimension of the first linear layer in FF module
        :param str|None ff_init: FF layers initialization
        :param bool|None ff_bias: If true, then bias is used for the FF layers
        :param float embed_dropout: dropout applied to the source embedding
        :param float dropout: general dropout
        :param float att_dropout: dropout applied to attention weights
        :param int enc_key_dim: encoder key dimension, also denoted as d_model, or d_key
        :param int att_num_heads: the number of attention heads
        :param str target: target labels key name
        :param float l2: add L2 regularization for trainable weights parameters
        :param float lstm_dropout: dropout applied to the input of the LSTMs in case they are used
        :param float rec_weight_dropout: dropout applied to the hidden-to-hidden weight matrices of the LSTM in case used
        :param bool with_ctc: if true, CTC loss is used
        :param bool native_ctc: if true, use returnn native ctc implementation instead of TF implementation
        :param float ctc_dropout: dropout applied on input to ctc
        :param float ctc_l2: L2 applied to the weight matrix of CTC softmax
        :param dict[str] ctc_opts: options for CTC
        """

        self.input = input
        self.input_layer = input_layer
        self.input_layer_conv_act = input_layer_conv_act
        self.add_abs_pos_enc_to_input = add_abs_pos_enc_to_input
        self.frontend_conv_l2 = frontend_conv_l2

        self.num_blocks = num_blocks
        self.conv_kernel_size = conv_kernel_size

        self.pos_enc = pos_enc
        self.rel_pos_clipping = rel_pos_clipping

        self.ff_bias = ff_bias

        self.specaug = specaug

        self.activation = activation

        self.block_final_norm = block_final_norm

        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lstm_dropout = lstm_dropout

        self.dropout_in = dropout_in

        # key and value dimensions are the same
        self.enc_key_dim = enc_key_dim
        self.enc_value_dim = enc_key_dim
        self.att_num_heads = att_num_heads
        self.enc_key_per_head_dim = enc_key_dim // att_num_heads
        self.enc_val_per_head_dim = enc_key_dim // att_num_heads

        self.ff_dim = ff_dim
        if self.ff_dim is None:
            self.ff_dim = 2 * self.enc_key_dim

        self.target = target

        self.l2 = l2
        self.self_att_l2 = self_att_l2
        self.rec_weight_dropout = rec_weight_dropout

        if batch_norm_opts is None:
            batch_norm_opts = {}

        bn_momentum = batch_norm_opts.pop("momentum", 0.1)
        bn_eps = batch_norm_opts.pop("epsilon", 1e-3)
        bn_update_sample_only_in_train = batch_norm_opts.pop("update_sample_only_in_training", True)
        bn_delay_sample_update = batch_norm_opts.pop("delay_sample_update", True)
        self.batch_norm_opts = {
            "momentum": bn_momentum,
            "epsilon": bn_eps,
            "update_sample_only_in_training": bn_update_sample_only_in_train,
            "delay_sample_update": bn_delay_sample_update,
        }
        self.batch_norm_opts.update(**batch_norm_opts)

        self.with_ctc = with_ctc
        self.native_ctc = native_ctc
        self.ctc_dropout = ctc_dropout
        self.ctc_loss_scale = ctc_loss_scale
        self.ctc_l2 = ctc_l2
        self.ctc_opts = ctc_opts
        if not self.ctc_opts:
            self.ctc_opts = {}
        self.ctc_self_align_delay = ctc_self_align_delay
        self.ctc_self_align_scale = ctc_self_align_scale

        self.start_conv_init = start_conv_init
        self.conv_module_init = conv_module_init
        self.mhsa_init = mhsa_init
        self.mhsa_out_init = mhsa_out_init
        self.ff_init = ff_init

        self.sandwich_conv = sandwich_conv

        # add maxpooling layers
        self.subsample = subsample
        self.subsample_list = [1] * num_blocks
        if subsample:
            for idx, s in enumerate(map(int, subsample.split("_")[:num_blocks])):
                self.subsample_list[idx] = s

        self.network = ReturnnNetwork()

        self.use_ln = use_ln

        self.pooling_str = pooling_str

        self.add_to_prefix_name = add_to_prefix_name
        self.output_layer_name = output_layer_name

        self.create_only_blocks = create_only_blocks

        self.no_mhsa_module = no_mhsa_module
        self.proj_input = proj_input

        self.use_sqrd_relu = use_sqrd_relu

        self.use_causal_layers = use_causal_layers
        self.use_causal_conv = use_causal_conv if use_causal_conv is not None else self.use_causal_layers

        self.conv_alternative_name = conv_alternative_name
        self.fix_merge_dims = fix_merge_dims

        self.weight_noise = weight_noise
        self.weight_noise_layers = weight_noise_layers
        if self.weight_noise_layers is None:
            self.weight_noise_layers = []
        for layer in self.weight_noise_layers:
            assert layer in ["mhsa", "conv", "frontend_conv"]

        self.convolution_first = convolution_first

        self.ff_weight_drop = ff_weight_dropout
        self.conv_weight_drop = conv_weight_dropout
        self.mhsa_weight_drop = mhsa_weight_dropout

        self.memory_variant_opts = memory_variant_opts
        if self.memory_variant_opts:
            self.concat_window_dim = SpatialDim("concat-window")
            self.enc_att_num_heads_dim = SpatialDim("enc-att-num-heads", att_num_heads)
            self.enc_per_head_dim = FeatureDim("enc-dim-per-head", self.enc_key_per_head_dim)

    def _create_ff_module(self, prefix_name, i, source, layer_index):
        """
        Add Feed Forward Module:
          LN -> FFN -> Swish -> Dropout -> FFN -> Dropout

        :param str prefix_name: some prefix name
        :param int i: FF module index
        :param str source: name of source layer
        :param int layer_index: index of layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = prefix_name + "_ffmod_{}".format(i)

        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)

        ff1 = self.network.add_linear_layer(
            "{}_ff1".format(prefix_name),
            ln,
            n_out=self.ff_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=self.ff_bias,
            param_dropout=self.ff_weight_drop,
        )

        if self.use_sqrd_relu:
            swish_act = self.network.add_activation_layer("{}_relu".format(prefix_name), ff1, activation="relu")
            swish_act = self.network.add_eval_layer(
                "{}_square_relu".format(prefix_name), swish_act, eval="source(0) ** 2"
            )
        else:
            swish_act = self.network.add_activation_layer(
                "{}_swish".format(prefix_name), ff1, activation=self.activation
            )

        drop1 = self.network.add_dropout_layer("{}_drop1".format(prefix_name), swish_act, dropout=self.dropout)

        ff2 = self.network.add_linear_layer(
            "{}_ff2".format(prefix_name),
            drop1,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=self.ff_bias,
            param_dropout=self.ff_weight_drop,
        )

        drop2 = self.network.add_dropout_layer("{}_drop2".format(prefix_name), ff2, dropout=self.dropout)

        half_step_ff = self.network.add_eval_layer("{}_half_step".format(prefix_name), drop2, eval="0.5 * source(0)")

        res_inputs = [half_step_ff, source]

        ff_module_res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=res_inputs, n_out=self.enc_key_dim
        )

        return ff_module_res

    def _create_mhsa_module(self, prefix_name, source, layer_index):
        """
        Add Multi-Headed Selft-Attention Module:
          LN + MHSA + Dropout

        :param str prefix: some prefix name
        :param str source: name of source layer
        :param int layer_index: index of layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = "{}_self_att".format(prefix_name)
        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)
        ln_rel_pos_enc = None

        if self.pos_enc == "rel":
            ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
                "{}_ln_rel_pos_enc".format(prefix_name),
                ln,
                n_out=self.enc_key_per_head_dim,
                forward_weights_init=self.ff_init,
                clipping=self.rel_pos_clipping,
            )

        if self.memory_variant_opts is not None:
            # ln: [B*C, W, D]
            # We want now to have [B, C, W, D]
            ln_split_chunk = self.network.add_generic_layer(
                f"{prefix_name}_ln_split_chunk",
                cls="split_batch_time",
                source=ln,
                base=self.memory_variant_opts.split_batch_time_base,
            )  # [B, C, W, D], C = chunked_time_dim
            mem_chunks = []
            mem_paddings = []
            for mem_idx in range(self.memory_variant_opts.mem_size):
                ln_chunk_shifted = self.network.add_generic_layer(
                    f"{prefix_name}_ln_chunk_shifted" + (f"_{mem_idx}" if mem_idx > 0 else ""),
                    cls="shift_axis",
                    source=ln_split_chunk,
                    axis=self.memory_variant_opts.chunked_time_dim,  # C
                    amount=mem_idx + 1,
                    pad=True,
                    adjust_size_info=False,  # no change in dim tag, C stays the same
                )  # [B, C, W, D]
                # Merge batch and chunk dim again.
                ln_chunk_shifted = self.network.add_generic_layer(
                    f"{prefix_name}_ln_chunk_shifted_" + (f"_{mem_idx}" if mem_idx > 0 else ""),
                    cls="merge_dims",
                    source=ln_chunk_shifted,
                    axes=("B", self.memory_variant_opts.chunked_time_dim),
                )  # [B*C, W, D]
                # Make sure the time_dim_axis (T) is set to the correct dim (W).
                ln_chunk_shifted = self.network.add_generic_layer(
                    f"{prefix_name}_ln_chunk_shifted__" + (f"_{mem_idx}" if mem_idx > 0 else ""),
                    cls="reinterpret_data",
                    source=ln_chunk_shifted,
                    set_axes={"T": "spatial"},
                )  # [B*C, W, D] but not time_dim_axis is set to W
                mem_chunks.append((ln_chunk_shifted, "T"))

                # mask paddings
                if layer_index == 1 and self.memory_variant_opts.mask_paddings:
                    # mask padded frames when computing att vector. this mask is shared for all layers.
                    energy_mask_ = self.network.add_generic_layer(
                        f"energy_mask_",  # shared for all layers
                        cls="constant",
                        value=True,
                        shape=(self.memory_variant_opts.chunked_time_dim, self.memory_variant_opts.chunk_size_dim),
                        shape_deps=[ln_split_chunk],  # [B,C,W,D], only used to define dim tag C
                        dtype="bool",
                        source=None,
                    )  # [C,W]
                    energy_mask = self.network.add_generic_layer(
                        f"energy_mask_shift_{mem_idx + 1}",
                        cls="shift_axis",
                        source=energy_mask_,
                        axis=self.memory_variant_opts.chunked_time_dim,
                        amount=mem_idx + 1,
                        pad=True,  # pad by False
                        adjust_size_info=False,  # no change in dim tag, C stays the same
                    )
                    mem_paddings.append((energy_mask, self.memory_variant_opts.chunk_size_dim))

            # reverse to concat left-most first
            mem_chunks.reverse()
            # Concatenate the shifted and the original tensor.
            ln_ = self.network.add_generic_layer(
                f"{prefix_name}_ln_concat",
                cls="concat",
                source=[*mem_chunks, (ln, "T")],
                out_dim=self.concat_window_dim,
            )  # [B*C, W*N, D]

            concat_mask_paddings = None
            if mem_paddings:
                mem_paddings.reverse()
                concat_mask_paddings = self.network.add_generic_layer(
                    f"concat_mask_paddings",
                    cls="concat",
                    source=[*mem_paddings, ("energy_mask_", self.memory_variant_opts.chunk_size_dim)],
                    out_dim=self.concat_window_dim,
                )  # [C, W*N]

            if self.memory_variant_opts.self_att_version == 0:
                assert self.memory_variant_opts.memory_input == "ln"
                # this implementation is not efficient.
                ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
                    f"{prefix_name}_ln_rel_pos_enc",
                    ln_,
                    n_out=self.enc_key_per_head_dim,
                    forward_weights_init=self.ff_init,
                    clipping=self.rel_pos_clipping,
                )  # [B*C, W*N, D/H]
                # same param name as before
                mhsa_ = self.network.add_self_att_layer(
                    name=prefix_name,
                    source=ln_,
                    n_out=self.enc_value_dim,
                    num_heads=self.att_num_heads,
                    total_key_dim=self.enc_key_dim,
                    att_dropout=self.att_dropout,
                    forward_weights_init=self.mhsa_init,
                    key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None,
                    l2=self.self_att_l2,
                    attention_left_only=self.use_causal_layers,
                    param_variational_noise=self.weight_noise if "mhsa" in self.weight_noise_layers else None,
                    param_dropout=self.mhsa_weight_drop,
                )  # [B*C, W*N, D]
                mhsa_splits = self.network.add_generic_layer(
                    f"{prefix_name}_splits",
                    cls="split",
                    source=mhsa_,
                    axis="T",
                    num_splits=2,
                )  # two tensors of shape [B*C, W, D]
                mhsa = self.network.add_generic_layer(
                    f"{prefix_name}_split_2",
                    cls="copy",
                    source=mhsa_splits + "/1",  # select second half
                )  # [B*C, W, D]
            else:
                # For efficient computation:
                # We cannot use SelfAttentionLayer on the concatenated tensor,
                # because it would waste compute time for the frames of the last chunk.
                # So reimplement the SelfAttentionLayer here explicitly.
                # key, value: via concatenated, [B*C, W*N, D]
                K = self.network.add_generic_layer(
                    f"{prefix_name}_ln_K",
                    cls="linear",
                    source=ln_,
                    n_out=self.enc_key_dim,
                    forward_weights_init=self.mhsa_init,
                    with_bias=False,
                    L2=self.self_att_l2,
                )  # [B*C, W*N, D]
                K_H = self.network.add_generic_layer(
                    f"{prefix_name}_ln_K_H",
                    cls="split_dims",
                    source=K,
                    axis="f",
                    dims=(self.enc_att_num_heads_dim, self.enc_per_head_dim),
                )  # [B*C, W*N, H, D/H]
                V = self.network.add_generic_layer(
                    f"{prefix_name}_ln_V",
                    cls="linear",
                    source=ln_,
                    n_out=self.enc_value_dim,
                    forward_weights_init=self.mhsa_init,
                    with_bias=False,
                    L2=self.self_att_l2,
                )  # [B*C, W*N, D]
                V_H = self.network.add_generic_layer(
                    f"{prefix_name}_ln_V_H",
                    cls="split_dims",
                    source=V,
                    axis="f",
                    dims=(self.enc_att_num_heads_dim, self.enc_per_head_dim),
                )  # [B*C, W*N, H, D/H]
                Q = self.network.add_generic_layer(
                    f"{prefix_name}_ln_Q",
                    cls="linear",
                    source=ln,
                    n_out=self.enc_key_dim,
                    forward_weights_init=self.mhsa_init,
                    with_bias=False,
                    L2=self.self_att_l2,
                )  # second half of shape [B*C, W, D]
                Q_H_ = self.network.add_generic_layer(
                    f"{prefix_name}_ln_Q_H_",
                    cls="split_dims",
                    source=Q,
                    axis="f",
                    dims=(self.enc_att_num_heads_dim, self.enc_per_head_dim),
                )  # [B*C, W, H, D/H]

                # this scaling is actually a bug in self_attention layer. so to be comparable, we do same here.
                dim_per_head_const = self.network.add_generic_layer(
                    f"{prefix_name}_dim_per_head_const",
                    cls="constant",
                    value=self.enc_key_per_head_dim,
                    source=None,
                    dtype="float32",
                )  # [1]
                Q_energy_factor = self.network.add_generic_layer(
                    f"{prefix_name}_Q_energy_factor", cls="eval", source=dim_per_head_const, eval="source(0) ** -0.5"
                )  # [1]
                Q_H = self.network.add_generic_layer(
                    f"{prefix_name}_ln_Q_H",
                    cls="combine",
                    kind="mul",
                    source=[Q_H_, Q_energy_factor],
                )  # [B*C, W, H, D/H]

                energy = self.network.add_generic_layer(
                    f"{prefix_name}_ln_energy",
                    cls="dot",
                    source=[K_H, Q_H],
                    reduce=self.enc_per_head_dim,  # D/H
                    var1="auto",
                    var2="auto",
                )  # [B*C, H, W*N, W]

                ln_rel_pos_enc = self.network.add_generic_layer(
                    f"{prefix_name}_ln_rel_pos_enc",
                    cls="relative_positional_encoding",
                    source=ln_,
                    out_dim=self.enc_per_head_dim,  # D/H
                    forward_weights_init=self.ff_init,
                    clipping=self.rel_pos_clipping,
                    query_spatial_dim=self.memory_variant_opts.chunk_size_dim,
                    key_value_spatial_dim=self.concat_window_dim,
                    query_offset=self.memory_variant_opts.chunk_size,
                )  # [W, W*N, D/H]

                energy_rel_pos = self.network.add_generic_layer(
                    f"{prefix_name}_ln_energy_rel_pos",
                    cls="dot",
                    source=[Q_H, ln_rel_pos_enc],
                    reduce=self.enc_per_head_dim,  # D/H
                    var1="auto",
                    var2="auto",
                )  # [B*C, H, W, W*N]

                energy = self.network.add_generic_layer(
                    f"{prefix_name}_ln_energy_",
                    cls="combine",
                    source=[energy, energy_rel_pos],
                    kind="add",
                    allow_broadcast_all_sources=False,
                )  # [B*C, H, W*N, W]

                if mem_paddings:
                    assert concat_mask_paddings
                    from i6_core.returnn.config import CodeWrapper

                    inf_const = self.network.add_generic_layer(
                        "inf_const",
                        cls="constant",
                        value=CodeWrapper("-float('inf')"),
                        source=None,
                        dtype="float32",
                    )
                    # for broadcasting, we need to split B and C first
                    energy_split = self.network.add_generic_layer(
                        f"{prefix_name}_ln_energy_split",
                        cls="split_batch_time",
                        source=energy,
                        base=self.memory_variant_opts.split_batch_time_base,
                    )  # [B, C, H, W*N, W]

                    energy = self.network.add_generic_layer(
                        f"{prefix_name}_ln_energy_masked_",
                        cls="switch",
                        condition=concat_mask_paddings,  # [C,W*N]
                        true_from=energy_split,
                        false_from=inf_const,
                        source=None,
                    )  # [B, C, H, W*N, W]

                    energy = self.network.add_generic_layer(
                        f"{prefix_name}_ln_energy_masked_merged",
                        cls="merge_dims",
                        source=energy,
                        axes=("B", self.memory_variant_opts.chunked_time_dim),
                    )  # [B*C, H, W*N, W]

                weights = self.network.add_generic_layer(
                    f"{prefix_name}_ln_weights",
                    cls="softmax_over_spatial",
                    source=energy,
                )  # [B*C, H, W, W*N]
                weights_drop = self.network.add_generic_layer(
                    f"{prefix_name}_ln_weights_drop",
                    cls="dropout",
                    source=weights,
                    dropout=self.att_dropout,
                    dropout_noise_shape={"*": None},
                )  # [B*C, H, W, W*N]
                att0 = self.network.add_generic_layer(
                    f"{prefix_name}_ln_att0",
                    cls="dot",
                    source=[weights_drop, V_H],
                    reduce=self.concat_window_dim,  # W*N
                    var1="auto",
                    var2="auto",
                )  # [B*C, H, W, D/H]
                mhsa_ = self.network.add_generic_layer(
                    f"{prefix_name}_ln_att_",
                    cls="merge_dims",
                    source=att0,
                    axes=(self.enc_att_num_heads_dim, self.enc_per_head_dim),
                )  # [B*C, W, D]
                mhsa = self.network.add_generic_layer(
                    f"{prefix_name}_ln_att",
                    cls="reinterpret_data",
                    source=mhsa_,
                    set_axes={"T": f"dim:{self.memory_variant_opts.chunk_size}"},
                )  # [B*C, W, D]
        else:
            mhsa = self.network.add_self_att_layer(
                name=prefix_name,
                source=ln,
                n_out=self.enc_value_dim,
                num_heads=self.att_num_heads,
                total_key_dim=self.enc_key_dim,
                att_dropout=self.att_dropout,
                forward_weights_init=self.mhsa_init,
                key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None,
                l2=self.self_att_l2,
                attention_left_only=self.use_causal_layers,
                param_variational_noise=self.weight_noise if "mhsa" in self.weight_noise_layers else None,
                param_dropout=self.mhsa_weight_drop,
            )

        mhsa_linear = self.network.add_linear_layer(
            "{}_linear".format(prefix_name),
            mhsa,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.mhsa_out_init,
            with_bias=False,
            param_dropout=self.mhsa_weight_drop,
        )

        drop = self.network.add_dropout_layer("{}_dropout".format(prefix_name), mhsa_linear, dropout=self.dropout)

        res_inputs = [drop, source]

        mhsa_res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=res_inputs, n_out=self.enc_value_dim
        )
        return mhsa_res

    def _create_convolution_module(self, prefix_name, source, layer_index, half_step=False):
        """
        Add Convolution Module:
          LN + point-wise-conv + GLU + depth-wise-conv + BN + Swish + point-wise-conv + Dropout

        :param str prefix_name: some prefix name
        :param str source: name of source layer
        :param int layer_index: index of layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = "{}_conv_mod".format(prefix_name)

        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)

        pointwise_conv1 = self.network.add_linear_layer(
            "{}_pointwise_conv1".format(prefix_name),
            ln,
            n_out=2 * self.enc_key_dim,
            activation=None,
            l2=self.l2,
            with_bias=self.ff_bias,
            forward_weights_init=self.conv_module_init,
            param_dropout=self.conv_weight_drop,
        )

        glu_act = self.network.add_gating_layer("{}_glu".format(prefix_name), pointwise_conv1)

        mem_chunks = []
        if self.memory_variant_opts is not None and self.memory_variant_opts.conv_cache:
            # glu_act: [B*C, W, D]
            # We want now to have [B, C, W, D]
            glu_act_split_chunk = self.network.add_generic_layer(
                f"{prefix_name}_glu_act_split_chunk",
                cls="split_batch_time",
                source=glu_act,
                base=self.memory_variant_opts.split_batch_time_base,
            )  # [B, C, W, D], C = chunked_time_dim
            for mem_idx in range(self.memory_variant_opts.mem_size):
                glu_act_chunk_shifted = self.network.add_generic_layer(
                    f"{prefix_name}_glu_act_chunk_shifted" + (f"_{mem_idx}" if mem_idx > 0 else ""),
                    cls="shift_axis",
                    source=glu_act_split_chunk,
                    axis=self.memory_variant_opts.chunked_time_dim,  # C
                    amount=mem_idx + 1,
                    pad=True,
                    adjust_size_info=False,  # no change in dim tag, C stays the same
                )  # [B, C, W, D]
                # Merge batch and chunk dim again.
                glu_act_chunk_shifted = self.network.add_generic_layer(
                    f"{prefix_name}_glu_act_chunk_shifted_" + (f"_{mem_idx}" if mem_idx > 0 else ""),
                    cls="merge_dims",
                    source=glu_act_chunk_shifted,
                    axes=("B", self.memory_variant_opts.chunked_time_dim),
                )  # [B*C, W, D]
                # Make sure the time_dim_axis (T) is set to the correct dim (W).
                glu_act_chunk_shifted = self.network.add_generic_layer(
                    f"{prefix_name}_glu_act_chunk_shifted__" + (f"_{mem_idx}" if mem_idx > 0 else ""),
                    cls="reinterpret_data",
                    source=glu_act_chunk_shifted,
                    set_axes={"T": "spatial"},
                )  # [B*C, W, D] but not time_dim_axis is set to W
                mem_chunks.append((glu_act_chunk_shifted, "T"))

        if mem_chunks:
            # reverse to concat left-most first
            mem_chunks.reverse()
            # Concatenate the shifted and the original tensor.
            glu_act_ = self.network.add_generic_layer(
                f"{prefix_name}_glu_act_concat",
                cls="concat",
                source=[*mem_chunks, (glu_act, "T")],
                out_dim=self.concat_window_dim,
            )  # [B*C, W*N, D]
        else:
            glu_act_ = glu_act

        if self.use_causal_conv:
            # pad to the left to make it causal
            depthwise_conv_input_padded = self.network.add_pad_layer(
                "{}_depthwise_conv_input_padded".format(prefix_name),
                glu_act_,
                axes="T",
                padding=(self.conv_kernel_size - 1, 0),
            )

            depthwise_conv = self.network.add_conv_layer(
                prefix_name + "_" + (self.conv_alternative_name or "depthwise_conv2"),
                depthwise_conv_input_padded,
                n_out=self.enc_key_dim,
                filter_size=(self.conv_kernel_size,),
                groups=self.enc_key_dim,
                l2=self.l2,
                forward_weights_init=self.conv_module_init,
                padding="valid",
                param_dropout=self.conv_weight_drop,
            )

            # Fix time dim, match with the original input
            depthwise_conv = self.network.add_reinterpret_data_layer(
                "{}_depthwise_conv2_".format(prefix_name),
                depthwise_conv,
                size_base=glu_act,
            )
        else:
            depthwise_conv = self.network.add_conv_layer(
                prefix_name + "_" + (self.conv_alternative_name or "depthwise_conv2"),
                glu_act_,
                n_out=self.enc_key_dim,
                filter_size=(self.conv_kernel_size,),
                groups=self.enc_key_dim,
                l2=self.l2,
                forward_weights_init=self.conv_module_init,
                param_dropout=self.conv_weight_drop,
            )

        if mem_chunks:
            # we apply convolution over the concatenated chunks but we only need the output of the current
            # chunk, thus, we need to slice from [B*C, W*N, D] to [B*C, W, D]
            depthwise_conv = self.network.add_generic_layer(
                f"{prefix_name}_depthwise_conv_slice",
                cls="slice",
                source=depthwise_conv,
                axis="T",
                slice_start=self.memory_variant_opts.chunk_size * self.memory_variant_opts.mem_size,
            )

        if self.use_ln:
            bn = self.network.add_layer_norm_layer("{}_layer_norm".format(prefix_name), depthwise_conv)
        else:
            bn = self.network.add_batch_norm_layer(
                "{}_bn".format(prefix_name), depthwise_conv, opts=self.batch_norm_opts
            )

        swish_act = self.network.add_activation_layer("{}_swish".format(prefix_name), bn, activation="swish")

        pointwise_conv2 = self.network.add_linear_layer(
            "{}_pointwise_conv2".format(prefix_name),
            swish_act,
            n_out=self.enc_key_dim,
            activation=None,
            l2=self.l2,
            with_bias=self.ff_bias,
            forward_weights_init=self.conv_module_init,
            param_dropout=self.conv_weight_drop,
        )

        drop = self.network.add_dropout_layer("{}_drop".format(prefix_name), pointwise_conv2, dropout=self.dropout)

        if half_step:
            drop = self.network.add_eval_layer("{}_half_step".format(prefix_name), drop, eval="0.5 * source(0)")

        res_inputs = [drop, source]

        res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=res_inputs, n_out=self.enc_key_dim
        )
        return res

    def _block_prefix_name(self, layer_index):
        if self.add_to_prefix_name:
            prefix_name = "conformer_block_%s_%02i" % (self.add_to_prefix_name, layer_index)
        else:
            prefix_name = "conformer_block_%02i" % layer_index
        return prefix_name

    def _create_conformer_block(self, i, source):
        """
        Add the whole Conformer block:
          x1 = x0 + 1/2 * FFN(x0)             (FFN module 1)
          x2 = x1 + MHSA(x1)                  (MHSA)
          x3 = x2 + Conv(x2)                  (Conv module)
          x4 = LayerNorm(x3 + 1/2 * FFN(x3))  (FFN module 2)

        :param int i: layer index
        :param str source: name of source layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = self._block_prefix_name(i)
        ff_module1 = self._create_ff_module(prefix_name, 1, source, i)

        if self.convolution_first:
            conv_module_ = self._create_convolution_module(prefix_name, ff_module1, i)
            mhsa_module = self._create_mhsa_module(prefix_name, conv_module_, i)
            ff_module2_input = mhsa_module
        else:
            if self.no_mhsa_module:
                mhsa = ff_module1  # use FF1 module output as input to conv module
            else:
                mhsa_input = ff_module1
                if self.sandwich_conv:
                    conv_module1 = self._create_convolution_module(
                        prefix_name + "_sandwich", ff_module1, i, half_step=True
                    )
                    mhsa_input = conv_module1
                mhsa = self._create_mhsa_module(prefix_name, mhsa_input, i)

            conv_module = self._create_convolution_module(prefix_name, mhsa, i, half_step=self.sandwich_conv)
            ff_module2_input = conv_module

        ff_module2 = self._create_ff_module(prefix_name, 2, ff_module2_input, i)
        res = ff_module2
        if self.block_final_norm:
            res = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), res)
        if self.subsample:
            assert 0 <= i - 1 < len(self.subsample)
            subsample_factor = self.subsample_list[i - 1]
            if subsample_factor > 1:
                res = self.network.add_pool_layer(res + "_pool{}".format(i), res, pool_size=(subsample_factor,))
        res = self.network.add_copy_layer(prefix_name, res)
        return res

    def _create_all_network_parts(self):
        """
        ConvSubsampling/LSTM -> Linear -> Dropout -> [Conformer Blocks] x N
        """
        data = self.input
        if self.specaug:
            data = self.network.add_eval_layer(
                "source",
                data,
                eval="self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
            )

        subsampled_input = None
        if self.input_layer is None:
            subsampled_input = data
        elif self.input_layer.startswith("stack"):
            stack_size = int(self.input_layer.split("-")[1])
            stack_window = self.network.add_window_layer(
                "stack_window", data, window_size=stack_size, stride=stack_size
            )  # [B,C,W,F]
            subsampled_input = self.network.add_merge_dims_layer(
                "stack_window_merge_dim",
                stack_window,
                axes=["static:0", "f"],
                keep_order=True,
            )  # [B,C,W*F]
        elif "lstm" in self.input_layer:
            sample_factor = int(self.input_layer.split("-")[1])
            pool_sizes = None
            if sample_factor == 2:
                pool_sizes = [2, 1]
            elif sample_factor == 4:
                pool_sizes = [2, 2]
            elif sample_factor == 6:
                pool_sizes = [3, 2]
            # add 2 LSTM layers with max pooling to subsample and encode positional information
            subsampled_input = self.network.add_lstm_layers(
                data,
                num_layers=2,
                lstm_dim=self.enc_key_dim,
                dropout=self.lstm_dropout,
                bidirectional=True,
                rec_weight_dropout=self.rec_weight_dropout,
                l2=self.l2,
                pool_sizes=pool_sizes,
            )
        elif self.input_layer == "conv-4":
            # conv-layer-1: 3x3x32 followed by max pool layer on feature axis (1, 2)
            # conv-layer-2: 3x3x64 with striding (2, 1) on time axis
            # conv-layer-3: 3x3x64 with striding (2, 1) on time axis

            # TODO: make this more generic

            conv_input = self.network.add_conv_block(
                "conv_out",
                data,
                hwpc_sizes=[((3, 3), (1, 2), 32)],
                l2=self.frontend_conv_l2,
                activation=self.input_layer_conv_act,
                init=self.start_conv_init,
                merge_out=False,
                param_variational_noise=self.weight_noise if "frontend_conv" in self.weight_noise_layers else None,
            )

            subsampled_input = self.network.add_conv_block(
                "conv_merged",
                conv_input,
                hwpc_sizes=[((3, 3), (2, 1), 64), ((3, 3), (2, 1), 64)],
                l2=self.frontend_conv_l2,
                activation=self.input_layer_conv_act,
                init=self.start_conv_init,
                use_striding=True,
                split_input=False,
                prefix_name="subsample_",
                merge_out_fixed=self.fix_merge_dims,
                param_variational_noise=self.weight_noise if "frontend_conv" in self.weight_noise_layers else None,
            )
        elif self.input_layer == "conv-6":
            conv_input = self.network.add_conv_block(
                "conv_out",
                data,
                hwpc_sizes=[((3, 3), (1, 2), 32)],
                l2=self.frontend_conv_l2,
                activation=self.input_layer_conv_act,
                init=self.start_conv_init,
                merge_out=False,
                param_variational_noise=self.weight_noise if "frontend_conv" in self.weight_noise_layers else None,
            )

            subsampled_input = self.network.add_conv_block(
                "conv_merged",
                conv_input,
                hwpc_sizes=[((3, 3), (3, 1), 64), ((3, 3), (2, 1), 64)],
                l2=self.frontend_conv_l2,
                activation=self.input_layer_conv_act,
                init=self.start_conv_init,
                use_striding=True,
                split_input=False,
                prefix_name="subsample_",
                merge_out_fixed=self.fix_merge_dims,
                param_variational_noise=self.weight_noise if "frontend_conv" in self.weight_noise_layers else None,
            )

        assert subsampled_input is not None

        source_linear = self.network.add_linear_layer(
            "source_linear",
            subsampled_input,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=False,
        )

        if self.add_abs_pos_enc_to_input:
            source_linear = self.network.add_pos_encoding_layer("input_abs_pos_enc", source_linear, add_to_input=True)

        if self.dropout_in:
            source_linear = self.network.add_dropout_layer("source_dropout", source_linear, dropout=self.dropout_in)

        conformer_block_src = source_linear
        for i in range(1, self.num_blocks + 1):
            conformer_block_src = self._create_conformer_block(i, conformer_block_src)

        encoder = self.network.add_copy_layer(self.output_layer_name, conformer_block_src)

        if self.with_ctc:
            default_ctc_loss_opts = {"beam_width": 1}
            if self.native_ctc:
                default_ctc_loss_opts["use_native"] = True
            else:
                self.ctc_opts.update({"ignore_longer_outputs_than_inputs": True})  # always enable
            if self.ctc_opts:
                default_ctc_loss_opts["ctc_opts"] = self.ctc_opts
            self.network.add_softmax_layer(
                "ctc",
                encoder,
                l2=self.ctc_l2,
                target=self.target,
                loss="ctc",
                dropout=self.ctc_dropout,
                loss_opts=default_ctc_loss_opts,
            )
            if self.ctc_loss_scale or self.ctc_self_align_delay:
                self.network["ctc"]["loss_scale"] = (self.ctc_loss_scale or 1.0) * (
                    (1.0 - self.ctc_self_align_scale) if self.ctc_self_align_delay else 1.0
                )

            if self.ctc_self_align_delay:
                # http://arxiv.org/abs/2105.05005
                assert self.ctc_self_align_delay > 0  # not implemented otherwise, but also not sure if meaningful
                self.network["ctc_log_prob"] = {"class": "activation", "from": "ctc", "activation": "safe_log"}
                # Cut off first N frames.
                self.network[f"ctc_log_prob_slice{self.ctc_self_align_delay}"] = {
                    "class": "slice",
                    "from": "ctc_log_prob",
                    "axis": "T",
                    "slice_start": self.ctc_self_align_delay,
                }
                # Forced alignment using that.
                self.network[f"ctc_forced_alignment_slice{self.ctc_self_align_delay}"] = {
                    "class": "forced_align",
                    "align_target": f"data:{self.target}",
                    "topology": "ctc",
                    "from": f"ctc_log_prob_slice{self.ctc_self_align_delay}",
                    "input_type": "log_prob",
                }
                # Add blanks at the end.
                self.network["_blank_idx"] = {
                    "class": "length",
                    "from": f"data:{self.target}",
                    "axis": "sparse_dim",
                    "sparse": True,
                }
                self.network[f"ctc_forced_alignment_shift{self.ctc_self_align_delay}"] = {
                    "class": "postfix_in_time",
                    "from": f"ctc_forced_alignment_slice{self.ctc_self_align_delay}",
                    "postfix": "_blank_idx",
                    "repeat": self.ctc_self_align_delay,
                }
                # Now CE loss to those targets.
                self.network[f"ctc_ce_shift{self.ctc_self_align_delay}"] = {
                    "class": "copy",
                    "from": "ctc",
                    "loss": "ce",
                    "loss_scale": (self.ctc_loss_scale or 1.0) * self.ctc_self_align_scale,
                    "target": f"layer:ctc_forced_alignment_shift{self.ctc_self_align_delay}",
                }

        return encoder

    def _create_conformer_blocks(self, input):
        if self.proj_input:
            conformer_block_src = self.network.add_linear_layer(
                "encoder_proj", input, n_out=self.enc_key_dim, activation=None, with_bias=False
            )
        else:
            conformer_block_src = input
        for i in range(1, self.num_blocks + 1):
            conformer_block_src = self._create_conformer_block(i, conformer_block_src)
        encoder = self.network.add_copy_layer(self.output_layer_name, conformer_block_src)
        return encoder

    def create_network(self):
        # create only conformer blocks without front-end, etc
        if self.create_only_blocks:
            return self._create_conformer_blocks(input=self.input)
        return self._create_all_network_parts()


@dataclass
class ConformerMemoryVariantOpts:
    split_batch_time_base: str
    chunked_time_dim: Dim
    chunk_size_dim: Dim
    chunk_size: int
    self_att_version: int  # TODO: just for testing
    mem_size: int
    mask_paddings: bool
    conv_cache: bool  # use conv cache for memory
