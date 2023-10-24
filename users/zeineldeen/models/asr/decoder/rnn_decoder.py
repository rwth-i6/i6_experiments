from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork
from i6_experiments.users.zeineldeen.modules.attention import AttentionMechanism


class RNNDecoder:
    """
    Represents RNN LSTM Attention-based decoder

    Related:
      * Single headed attention based sequence-to-sequence model for state-of-the-art results on Switchboard
        ref: https://arxiv.org/abs/2001.07263
    """

    def __init__(
        self,
        base_model,
        source=None,
        dropout=0.0,
        softmax_dropout=0.3,
        label_smoothing=0.1,
        target="bpe",
        beam_size=12,
        embed_dim=621,
        embed_dropout=0.0,
        lstm_num_units=1024,
        output_num_units=1024,
        enc_key_dim=1024,
        l2=None,
        att_dropout=None,
        rec_weight_dropout=None,
        zoneout=False,
        ff_init=None,
        add_lstm_lm=False,
        lstm_lm_dim=1024,
        loc_conv_att_filter_size=None,
        reduceout=True,
        att_num_heads=1,
        embed_weight_init=None,
        lstm_weights_init=None,
        length_normalization=True,
        coverage_threshold=None,
        coverage_scale=None,
        coverage_update="sum",
        ce_loss_scale=1.0,
        use_zoneout_output: bool = False,
        monotonic_att_weights_loss="l1",
        monotonic_att_weights_loss_scale=None,
        monotonic_att_weights_loss_scale_in_recog=None,
        att_weights_variance_loss_scale=None,
        include_eos_in_search_output=False,
    ):
        """
        :param base_model: base/encoder model instance
        :param str source: input to decoder subnetwork
        :param float dropout: dropout applied to the input of LM-like lstm
        :param float softmax_dropout: Dropout applied to the softmax input
        :param float label_smoothing: label smoothing value applied to softmax
        :param str target: target data key name
        :param int beam_size: value of the beam size
        :param int embed_dim: target embedding dimension
        :param float|None embed_dropout: dropout to be applied on the target embedding
        :param int lstm_num_units: the number of hidden units for the decoder LSTM
        :param int output_num_units: the number of hidden dimensions for the last layer before softmax
        :param int enc_key_dim: the number of hidden dimensions for the encoder key
        :param float|None l2: weight decay with l2 norm
        :param float|None att_dropout: dropout applied to attention weights
        :param float|None rec_weight_dropout: dropout applied to weight paramters
        :param bool zoneout: if set, zoneout LSTM cell is used in the decoder instead of nativelstm2
        :param str|None ff_init: feed-forward weights initialization
        :param bool add_lstm_lm: add separate LSTM layer that acts as LM-like model
          same as here: https://arxiv.org/abs/2001.07263
        :param float lstm_lm_dim: LM-like lstm dimension
        :param int|None loc_conv_att_filter_size:
        :param bool reduceout: if set to True, maxout layer is used
        :param int att_num_heads: number of attention heads
        :param str|None embed_weight_init: embedding weights initialization
        :param str|None lstm_weights_init: lstm weights initialization
        :param int lstm_lm_proj_dim: LM-like lstm projection dimension
        :param bool length_normalization: if set to True, length normalization is applied
        :param float|None coverage_threshold: threshold for coverage value used in search
        :param float|None coverage_scale: scale for coverage value
        :param str coverage_update: either cumulative sum or maximum
        :param float ce_loss_scale: scale for cross-entropy loss
        :param bool use_zoneout_output: if set, return the output h after zoneout
        """

        self.base_model = base_model

        self.source = source

        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.label_smoothing = label_smoothing

        self.enc_key_dim = enc_key_dim
        self.enc_value_dim = base_model.enc_value_dim
        self.att_num_heads = att_num_heads

        self.target = target

        self.beam_size = beam_size

        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout

        self.dec_lstm_num_units = lstm_num_units
        self.dec_output_num_units = output_num_units

        self.ff_init = ff_init

        self.decision_layer_name = None  # this is set in the end-point config

        self.l2 = l2
        self.att_dropout = att_dropout
        self.rec_weight_dropout = rec_weight_dropout
        self.dec_zoneout = zoneout

        self.add_lstm_lm = add_lstm_lm
        self.lstm_lm_dim = lstm_lm_dim

        self.loc_conv_att_filter_size = loc_conv_att_filter_size

        self.embed_weight_init = embed_weight_init
        self.lstm_weights_init = lstm_weights_init

        self.reduceout = reduceout

        self.length_normalization = length_normalization
        self.coverage_threshold = coverage_threshold
        self.coverage_scale = coverage_scale
        assert coverage_update in ["sum", "max"]
        self.coverage_update = coverage_update

        self.ce_loss_scale = ce_loss_scale

        self.use_zoneout_output = use_zoneout_output

        self.monotonic_att_weights_loss = monotonic_att_weights_loss
        self.monotonic_att_weights_loss_scale = monotonic_att_weights_loss_scale
        self.att_weights_variance_loss_scale = att_weights_variance_loss_scale
        self.monotonic_att_weights_loss_scale_in_recog = monotonic_att_weights_loss_scale_in_recog

        self.include_eos_in_search_output = include_eos_in_search_output

        self.network = ReturnnNetwork()
        self.subnet_unit = ReturnnNetwork()
        self.dec_output = None
        self.output_prob = None

    def add_decoder_subnetwork(self, subnet_unit: ReturnnNetwork):
        subnet_unit.add_compare_layer("end", source="output", value=0)  # sentence end token

        # target embedding
        subnet_unit.add_linear_layer(
            "target_embed0",
            "output",
            n_out=self.embed_dim,
            initial_output=0,
            with_bias=False,
            l2=self.l2,
            forward_weights_init=self.embed_weight_init,
        )

        subnet_unit.add_dropout_layer(
            "target_embed", "target_embed0", dropout=self.embed_dropout, dropout_noise_shape={"*": None}
        )

        # attention
        att = AttentionMechanism(
            enc_key_dim=self.enc_key_dim,
            att_num_heads=self.att_num_heads,
            att_dropout=self.att_dropout,
            l2=self.l2,
            loc_filter_size=self.loc_conv_att_filter_size,
            loc_num_channels=self.enc_key_dim,
        )
        subnet_unit.update(att.create())

        monotonic_att_weights_penalty = None

        if self.monotonic_att_weights_loss_scale or self.att_weights_variance_loss_scale:
            enc_len_range = self.network.add_range_in_axis_layer(
                "enc_len_range", "encoder", axis="T", dtype="float32"
            )  # [B]
            expected_att_weights_pos = self.subnet_unit.add_combine_layer(
                "expected_att_weights_pos",
                ["att_weights", "base:" + enc_len_range],
                kind="mul",
                allow_broadcast_all_sources=True,
            )  # [B,1,T]
            expected_att_weights_pos_reduce = self.subnet_unit.add_reduce_layer(
                "expected_att_weights_pos_reduce",
                expected_att_weights_pos,
                mode="sum",
                axes=["T"],
                keep_dims=False,
                initial_output=0,
            )  # [B,1]
            if self.monotonic_att_weights_loss_scale:
                #  delta_i = E_i - E_{i-1} and E_i = sum_{t=0}^{t-1} alpha(t|i) * t
                expected_att_weights_pos_delta = self.subnet_unit.add_combine_layer(
                    "expected_att_weights_pos_delta",
                    [expected_att_weights_pos_reduce, "prev:" + expected_att_weights_pos_reduce],  # E_j - E_{j-1}
                    kind="sub",
                )  # [B,1]

                if self.monotonic_att_weights_loss == "l1":
                    # L = |delta_i - 1| - (delta_i - 1)
                    monotonic_loss_str = f"tf.math.abs(source(0) - 1) - (source(0) - 1)"
                elif self.monotonic_att_weights_loss == "l2":
                    # L = (|delta_i - 1| - (delta_i - 1))^2
                    monotonic_loss_str = f"(tf.math.abs(source(0) - 1) - (source(0) - 1)) * (tf.math.abs(source(0) - 1) - (source(0) - 1))"
                else:
                    raise NotImplementedError(f"monotonic_att_weights_loss={self.monotonic_att_weights_loss!r}")

                self.subnet_unit.add_eval_layer(
                    "monotonic_att_weights_loss",
                    source=expected_att_weights_pos_delta,
                    eval=monotonic_loss_str,
                    loss="as_is",  # register as loss
                    loss_scale=self.monotonic_att_weights_loss_scale,
                )  # [B,1]

                if self.monotonic_att_weights_loss_scale_in_recog:
                    monotonic_att_weights_penalty = self.subnet_unit.add_eval_layer(
                        "monotonic_att_weights_penalty",
                        source=expected_att_weights_pos_delta,
                        eval=monotonic_loss_str,
                    )
                    monotonic_att_weights_penalty = self.subnet_unit.add_squeeze_layer(
                        "monotonic_att_weights_penalty_squeeze", monotonic_att_weights_penalty, axis="except_batch"
                    )

            if self.att_weights_variance_loss_scale:
                # L = sum_{t=0}^{T-1} alpha(t|i) * (t - E_i)^2
                att_weights_variance_ = self.subnet_unit.add_combine_layer(
                    "att_weighst_variance_",
                    [expected_att_weights_pos_reduce, "base:" + enc_len_range],
                    kind="sub",
                )  # [B,1,T]
                att_weights_variance = self.subnet_unit.add_eval_layer(
                    "att_weights_variance", att_weights_variance_, eval="source(0) ** 2"
                )  # [B,1,T]
                att_weights_variance_loss_input = self.subnet_unit.add_combine_layer(
                    "att_weights_variance_loss_",
                    ["att_weights", att_weights_variance],
                    kind="mul",
                )  # [B,1,T]
                self.subnet_unit.add_reduce_layer(
                    "att_weights_variance_loss",
                    att_weights_variance_loss_input,
                    mode="sum",
                    axes=["T"],
                    keep_dims=False,
                    loss="as_is",
                    loss_scale=self.att_weights_variance_loss_scale,
                )  # [B,1]

        # LM-like component same as here https://arxiv.org/pdf/2001.07263.pdf
        lstm_lm_component_proj = None
        if self.add_lstm_lm:
            lstm_lm_component = subnet_unit.add_rec_layer(
                "lm_like_s",
                "prev:target_embed",
                n_out=self.lstm_lm_dim,
                l2=self.l2,
                unit="NativeLSTM2",
                rec_weight_dropout=self.rec_weight_dropout,
                weights_init=self.lstm_weights_init,
            )
            if self.lstm_lm_dim != self.enc_value_dim:
                lstm_lm_component_proj = subnet_unit.add_linear_layer(
                    "lm_like_s_proj", lstm_lm_component, n_out=self.enc_value_dim, l2=self.l2, dropout=self.dropout
                )
            else:
                lstm_lm_component_proj = lstm_lm_component

        if self.add_lstm_lm:
            # front-lstm + prev:context
            lstm_inputs = subnet_unit.add_combine_layer(
                "add_embed_ctx", [lstm_lm_component_proj, "prev:att"], kind="add", n_out=self.enc_value_dim
            )
        else:
            lstm_inputs = ["prev:target_embed", "prev:att"]

        # LSTM decoder (or decoder state)
        if self.dec_zoneout:
            zoneout_unit_opts = {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05}
            if self.use_zoneout_output:
                zoneout_unit_opts["use_zoneout_output"] = True
            subnet_unit.add_rnn_cell_layer(
                "s",
                lstm_inputs,
                n_out=self.dec_lstm_num_units,
                l2=self.l2,
                weights_init=self.lstm_weights_init,
                unit="zoneoutlstm",
                unit_opts=zoneout_unit_opts,
            )
        else:
            subnet_unit.add_rec_layer(
                "s",
                lstm_inputs,
                n_out=self.dec_lstm_num_units,
                l2=self.l2,
                unit="NativeLSTM2",
                rec_weight_dropout=self.rec_weight_dropout,
                weights_init=self.lstm_weights_init,
            )

        if self.add_lstm_lm:
            # s_transformed (query) has 1024 dim
            s_proj = subnet_unit.add_linear_layer(
                "s_proj", "s_transformed", n_out=self.enc_value_dim, l2=self.l2, dropout=self.dropout
            )
            # back-lstm (query) + context
            readout_in_src = subnet_unit.add_combine_layer(
                "add_s_att", [s_proj, "att"], kind="add", n_out=self.enc_value_dim
            )
        else:
            readout_in_src = ["s", "prev:target_embed", "att"]

        subnet_unit.add_linear_layer("readout_in", readout_in_src, n_out=self.dec_output_num_units, l2=self.l2)

        if self.reduceout:
            subnet_unit.add_reduceout_layer("readout", "readout_in")
        else:
            subnet_unit.add_copy_layer("readout", "readout_in")

        ce_loss_opts = {"label_smoothing": self.label_smoothing}
        if self.ce_loss_scale != 1.0:
            ce_loss_opts["scale"] = self.ce_loss_scale

        self.output_prob = subnet_unit.add_softmax_layer(
            "output_prob",
            "readout",
            l2=self.l2,
            loss="ce",
            loss_opts=ce_loss_opts,
            target=self.target,
            dropout=self.softmax_dropout,
        )

        if self.coverage_scale and self.coverage_threshold:
            assert self.att_num_heads == 1, "Not supported for multi-head attention."  # TODO: just average the heads?
            if self.coverage_update == "sum":
                accum_w = self.subnet_unit.add_eval_layer(
                    "accum_w", source=["prev:att_weights", "att_weights"], eval="source(0) + source(1)"
                )  # [B,enc-T,H=1]
            else:
                assert self.coverage_update == "max"
                accum_w = self.subnet_unit.add_combine_layer(
                    "accum_w", ["prev:att_weights", "att_weights"], kind="maximum"
                )

            assert self.att_num_heads == 1, "Not supported for multi-head attention."  # TODO: just average the heads?
            merge_accum_w = self.subnet_unit.add_merge_dims_layer(
                "merge_accum_w", accum_w, axes="except_batch"
            )  # [B,enc-T]
            coverage_mask = self.subnet_unit.add_compare_layer(
                "coverage_mask", merge_accum_w, kind="greater", value=self.coverage_threshold
            )  # [B,enc-T]
            float_coverage_mask = self.subnet_unit.add_cast_layer(
                "float_coverage_mask", coverage_mask, dtype="float32"
            )  # [B,enc-T]

            accum_coverage = self.subnet_unit.add_reduce_layer(
                "accum_coverage", float_coverage_mask, mode="sum", axes=-1
            )  # [B]

            accum_coverage = self.subnet_unit.add_combine_layer(
                "diff_accum_coverage",
                [accum_coverage, "prev:" + accum_coverage],
                kind="sub",
            )  # [B]

            self.output_prob = self.subnet_unit.add_eval_layer(
                "output_prob_coverage",
                source=[self.output_prob, accum_coverage],
                eval=f"source(0) + {self.coverage_scale} * source(1)",
            )

        if self.monotonic_att_weights_loss_scale_in_recog:
            assert self.monotonic_att_weights_loss_scale, "monotonic loss scale for training must be set."
            assert monotonic_att_weights_penalty
            self.output_prob = self.subnet_unit.add_eval_layer(
                "output_prob_monotonic_penalty",
                source=[self.output_prob, monotonic_att_weights_penalty],
                eval=f"source(0) - {self.monotonic_att_weights_loss_scale_in_recog} * source(1)",
            )

        if self.length_normalization:
            subnet_unit.add_choice_layer(
                "output", self.output_prob, target=self.target, beam_size=self.beam_size, initial_output=0
            )
        else:
            subnet_unit.add_choice_layer(
                "output",
                self.output_prob,
                target=self.target,
                beam_size=self.beam_size,
                initial_output=0,
                length_normalization=False,
            )

        # recurrent subnetwork
        dec_output = self.network.add_subnet_rec_layer(
            "output",
            unit=subnet_unit.get_net(),
            target=self.target,
            source=self.source,
            include_eos=self.include_eos_in_search_output,
        )

        return dec_output

    def create_network(self):
        self.dec_output = self.add_decoder_subnetwork(self.subnet_unit)

        # Add to Base/Encoder network

        if hasattr(self.base_model, "enc_proj_dim") and self.base_model.enc_proj_dim:
            self.base_model.network.add_copy_layer("enc_ctx", "encoder_proj")
            self.base_model.network.add_split_dim_layer(
                "enc_value", "encoder_proj", dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads)
            )
        else:
            self.base_model.network.add_linear_layer(
                "enc_ctx", "encoder", with_bias=True, n_out=self.enc_key_dim, l2=self.base_model.l2
            )
            self.base_model.network.add_split_dim_layer(
                "enc_value", "encoder", dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads)
            )

        self.base_model.network.add_linear_layer(
            "inv_fertility", "encoder", activation="sigmoid", n_out=self.att_num_heads, with_bias=False
        )

        decision_layer_name = self.base_model.network.add_decide_layer("decision", self.dec_output, target=self.target)
        self.decision_layer_name = decision_layer_name

        return self.dec_output
