"""
Model, based on Mohammads code.
"""


from __future__ import annotations
from typing import Optional, Callable

from returnn.tf.util.data import Dim, SpatialDim, FeatureDim

from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork
from i6_experiments.users.zeineldeen.modules.abs_module import AbsModule


class AttentionMechanism(AbsModule):
    """
    Single-head or Multi-head attention mechanism
    """

    def __init__(
        self,
        enc_key_dim,
        att_num_heads,
        att_dropout,
        l2,
        loc_filter_size,
        loc_num_channels,
    ):
        super().__init__()
        self.enc_key_dim = enc_key_dim
        assert isinstance(att_num_heads, Dim)
        self.att_num_heads = att_num_heads

        self.att_dropout = att_dropout
        self.l2 = l2

        self.loc_filter_size = loc_filter_size
        self.loc_num_channels = loc_num_channels

        self.select_base_enc: Optional[Callable[[str], str]] = None
        self.enc_time_dim = None

    def create(self):
        out_net = ReturnnNetwork()

        out_net.add_linear_layer(
            "s_transformed", "s", n_out=self.enc_key_dim, with_bias=False, l2=self.l2
        )  # project query

        enc_ctx = "base:enc_ctx"
        if self.select_base_enc:
            enc_ctx = self.select_base_enc(enc_ctx)
        out_net.add_combine_layer(
            "energy_in",
            [enc_ctx, "s_transformed"],
            kind="add",
            n_out=self.enc_key_dim,
        )

        # compute energies
        out_net.add_activation_layer("energy_tanh", "energy_in", activation="tanh")
        energy = out_net.add_linear_layer(
            "energy",
            "energy_tanh",
            n_out=self.att_num_heads.dimension,
            out_dim=self.att_num_heads,
            with_bias=False,
            l2=self.l2,
        )

        att_sm_opts = {}
        if self.enc_time_dim:
            att_sm_opts["axis"] = self.enc_time_dim
        if self.att_dropout:
            att_weights0 = out_net.add_softmax_over_spatial_layer(
                "att_weights0", energy, **att_sm_opts
            )
            att_weights = out_net.add_dropout_layer(
                "att_weights",
                att_weights0,
                dropout=self.att_dropout,
                dropout_noise_shape={"*": None},
            )
        else:
            att_weights = out_net.add_softmax_over_spatial_layer(
                "att_weights", energy, **att_sm_opts
            )

        enc_value = "base:enc_value"
        if self.select_base_enc:
            enc_value = self.select_base_enc(enc_value)
        if self.enc_time_dim:
            att0 = out_net.add_dot_layer(
                "att0",
                [att_weights, enc_value],
                reduce=self.enc_time_dim,
                var1="auto",
                var2="auto",
            )
        else:
            att0 = out_net.add_generic_att_layer(
                "att0", weights=att_weights, base=enc_value
            )
        self.name = out_net.add_merge_dims_layer("att", att0, axes="static")

        return out_net.get_net()


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
        loc_conv_att_num_channels=None,
        reduceout=True,
        att_num_heads=1,
        embed_weight_init=None,
        lstm_weights_init=None,
        lstm_lm_proj_dim=1024,
        length_normalization=True,
        coverage_threshold=None,
        coverage_scale=None,
        enc_chunks_dim: Optional[Dim] = None,
        enc_time_dim: Optional[Dim] = None,
        eos_id=0,
        search_type: Optional[str] = None,
    ):
        """
        :param base_model: base/encoder model instance
        :param str source: input to decoder subnetwork
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
        :param float lstm_lm_dim:
        :param int|None loc_conv_att_filter_size:
        :param int|None loc_conv_att_num_channels:
        :param bool reduceout: if set to True, maxout layer is used
        :param int att_num_heads: number of attention heads
        :param enc_chunks_dim:
        :param enc_time_dim:
        :param int eos_id: end of sentence id. or end-of-chunk if chunking is used
        :param search_type:
            None -> use RETURNN default handling via search flag (i.e. disabled in training, enabled in search mode).
            "end-of-chunk" -> assume given targets without EOC, and search for EOC.
        """

        self.base_model = base_model

        self.source = source

        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.label_smoothing = label_smoothing

        self.enc_key_dim = enc_key_dim
        self.enc_value_dim = base_model.enc_value_dim
        if isinstance(att_num_heads, int):
            att_num_heads = SpatialDim("dec-att-num-heads", att_num_heads)
        assert isinstance(att_num_heads, Dim)
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
        self.lstm_lm_proj_dim = lstm_lm_proj_dim

        self.loc_conv_att_filter_size = loc_conv_att_filter_size
        self.loc_conv_att_num_channels = loc_conv_att_num_channels

        self.embed_weight_init = embed_weight_init
        self.lstm_weights_init = lstm_weights_init

        self.reduceout = reduceout

        self.length_normalization = length_normalization
        self.coverage_threshold = coverage_threshold
        self.coverage_scale = coverage_scale

        self.enc_chunks_dim = enc_chunks_dim
        self.enc_time_dim = enc_time_dim
        self.eos_id = eos_id
        self.search_type = search_type

        self.network = ReturnnNetwork()
        self.subnet_unit = ReturnnNetwork()
        self.dec_output = None
        self.output_prob = None

    def add_decoder_subnetwork(self, subnet_unit: ReturnnNetwork):

        if self.enc_chunks_dim:  # use chunking
            subnet_unit["new_label_pos"] = {
                "class": "eval",
                "from": ["output", "prev:new_label_pos"],
                "eval": f"tf.where(tf.equal(source(0), {self.eos_id}), source(1), source(1) + 1)",
                "out_type": {
                    "dtype": "int32",
                    "dim": None,
                    "sparse_dim": self.enc_chunks_dim,
                },
                "initial_output": 0,
            }

            subnet_unit["label_pos"] = {"class": "copy", "from": "prev:new_label_pos"}

            subnet_unit["label_pos_reached_end"] = {
                "class": "compare",
                "from": ["label_pos", "ground_truth_label_seq_len"],
                "kind": "greater_equal",
            }

            subnet_unit["chunk_idx_reached_last"] = {
                "class": "compare",
                "from": ["chunk_idx", "last_chunk_idx"],
                "kind": "equal",
            }

            subnet_unit["chunk_idx_can_be_finished"] = {
                "class": "eval",
                "from": ["chunk_idx_reached_last", "label_pos_reached_end"],
                "eval": "tf.logical_or(tf.logical_and(source(0), source(1)), tf.logical_not(source(0)))",
            }

            subnet_unit["ground_truth_label"] = {
                "class": "gather",
                "from": f"base:data:{self.target}",
                "axis": "T",
                "position": "label_pos",
                "clip_to_valid": True,
            }

            subnet_unit["ground_truth_label_seq_len"] = {
                "class": "length",
                "from": f"base:data:{self.target}",
                "axis": "T",
            }

            subnet_unit["ground_truth_last_label_pos"] = {
                "class": "eval",
                "from": "ground_truth_label_seq_len",
                "eval": "source(0) - 1",
            }

            subnet_unit["new_chunk_idx"] = {
                "class": "eval",
                "from": ["output", "prev:new_chunk_idx"],
                "eval": f"tf.where(tf.equal(source(0), {self.eos_id}), source(1) + 1, source(1))",
                "out_type": {
                    "dtype": "int32",
                    "dim": None,
                    "sparse_dim": self.enc_chunks_dim,
                },
                "initial_output": 0,
            }

            subnet_unit["chunk_idx"] = {"class": "copy", "from": "prev:new_chunk_idx"}

            subnet_unit["num_chunks"] = {
                "class": "length",
                "from": "base:encoder",
                "axis": self.enc_chunks_dim,
            }

            subnet_unit["last_chunk_idx"] = {
                "class": "eval",
                "from": "num_chunks",
                "eval": "source(0) - 1",
            }

            subnet_unit["end"] = {
                "class": "compare",
                "from": ["new_chunk_idx", "num_chunks"],
                "kind": "greater_equal",
            }

        else:  # no chunking
            subnet_unit.add_compare_layer(
                "end", source="output", value=self.eos_id
            )  # sentence end token

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
            "target_embed",
            "target_embed0",
            dropout=self.embed_dropout,
            dropout_noise_shape={"*": None},
        )

        # attention
        att = AttentionMechanism(
            enc_key_dim=self.enc_key_dim,
            att_num_heads=self.att_num_heads,
            att_dropout=self.att_dropout,
            l2=self.l2,
            loc_filter_size=self.loc_conv_att_filter_size,
            loc_num_channels=self.loc_conv_att_num_channels,
        )
        if self.enc_chunks_dim:

            def _gather_chunk(source: str) -> str:
                name = source.replace("base:", "")
                subnet_unit[name + "_gather"] = {
                    "class": "gather",
                    "from": source,
                    "position": "chunk_idx",
                    "axis": self.enc_chunks_dim,
                    "clip_to_valid": True,
                }
                subnet_unit[name + "_set_time"] = {
                    "class": "reinterpret_data",
                    "from": name + "_gather",
                    "set_axes": {"T": self.enc_time_dim},
                }
                return name + "_set_time"

            assert self.enc_time_dim
            att.select_base_enc = _gather_chunk
            att.enc_time_dim = self.enc_time_dim
        subnet_unit.update(att.create())

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
            lstm_lm_component_proj = subnet_unit.add_linear_layer(
                "lm_like_s_proj",
                lstm_lm_component,
                n_out=self.lstm_lm_proj_dim,
                l2=self.l2,
                with_bias=False,
                dropout=self.dropout,
            )

        lstm_inputs = []
        if lstm_lm_component_proj:
            lstm_inputs += [lstm_lm_component_proj]
        else:
            lstm_inputs += ["prev:target_embed"]
        lstm_inputs += ["prev:att"]

        if self.add_lstm_lm:
            # element-wise addition is applied instead of concat
            lstm_inputs = subnet_unit.add_combine_layer(
                "add_embed_ctx", lstm_inputs, kind="add", n_out=self.lstm_lm_proj_dim
            )

        # LSTM decoder (or decoder state)
        if self.dec_zoneout:
            subnet_unit.add_rnn_cell_layer(
                "s",
                lstm_inputs,
                n_out=self.dec_lstm_num_units,
                l2=self.l2,
                weights_init=self.lstm_weights_init,
                unit="zoneoutlstm",
                unit_opts={"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05},
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

        s_name = "s"
        if self.add_lstm_lm:
            s_name = subnet_unit.add_linear_layer(
                "s_proj",
                "s",
                n_out=self.lstm_lm_proj_dim,
                with_bias=False,
                dropout=self.dropout,
                l2=self.l2,
            )

        if self.add_lstm_lm:
            readout_in_src = subnet_unit.add_combine_layer(
                "add_s_att", [s_name, "att"], kind="add", n_out=self.lstm_lm_proj_dim
            )
        else:
            readout_in_src = [s_name, "prev:target_embed", "att"]

        subnet_unit.add_linear_layer(
            "readout_in", readout_in_src, n_out=self.dec_output_num_units, l2=self.l2
        )

        if self.reduceout:
            subnet_unit.add_reduceout_layer("readout", "readout_in")
        else:
            subnet_unit.add_copy_layer("readout", "readout_in")

        out_prob_opts = {}
        loss_ext = dict(
            loss="ce",
            loss_opts={"label_smoothing": self.label_smoothing},
        )
        if not self.search_type:
            out_prob_opts.update(loss_ext)
        self.output_prob = subnet_unit.add_softmax_layer(
            "output_prob",
            "readout",
            l2=self.l2,
            target=f"layer:base:data:{self.target}"
            if self.search_type == "end-of-chunk"
            else self.target,
            dropout=self.softmax_dropout,
            **out_prob_opts,
        )
        if self.search_type == "end-of-chunk":
            subnet_unit["_label_indices"] = {
                "class": "range_in_axis",
                "from": f"base:data:{self.target}",
                "axis": "sparse_dim",
            }
            subnet_unit["_label_indices_eq_eoc"] = {
                "class": "compare",
                "from": "_label_indices",
                "value": self.eos_id,
                "kind": "equal",
            }
            subnet_unit["_label_indices_eq_eoc_"] = {
                "class": "switch",
                "condition": "chunk_idx_can_be_finished",
                "true_from": "_label_indices_eq_eoc",
                "false_from": False,
            }
            subnet_unit["_label_indices_eq_true_label"] = {
                "class": "compare",
                "from": ["_label_indices", "ground_truth_label"],
                "kind": "equal",
            }
            subnet_unit["_label_indices_eq_true_label_"] = {
                "class": "switch",
                "condition": "label_pos_reached_end",
                "true_from": False,
                "false_from": "_label_indices_eq_true_label",
            }
            subnet_unit["eoc_label_mask"] = {
                "class": "combine",
                "kind": "logical_or",
                "from": ["_label_indices_eq_eoc_", "_label_indices_eq_true_label_"],
            }
            subnet_unit["output_prob_filter_eoc"] = {
                "class": "switch",
                "condition": "eoc_label_mask",
                "true_from": "output_prob",
                "false_from": 1e-20,
            }
            self.output_prob = "output_prob_filter_eoc"

            subnet_unit["output_logits"] = {
                "class": "eval",
                "from": "output_prob",
                "collocate_with": "output_prob",
                "eval": "source(0, as_layer=True).output_before_activation.x",
            }
            loss_ext["loss_opts"]["input_type"] = "logits"
            subnet_unit["output_prob_loss"] = {
                "class": "copy",
                "from": "output_logits",
                "target": "layer:output",
                "extra_deps": "output",
                "collocate_with": "output",
                **loss_ext,
            }

        if self.coverage_scale and self.coverage_threshold:
            assert (
                self.att_num_heads.dimension == 1
            ), "Not supported for multi-head attention."  # TODO: just average the heads?
            accum_w = self.subnet_unit.add_eval_layer(
                "accum_w",
                source=["prev:att_weights", "att_weights"],
                eval="source(0) + source(1)",
            )  # [B,enc-T,H=1]
            merge_accum_w = self.subnet_unit.add_merge_dims_layer(
                "merge_accum_w", accum_w, axes="except_batch"
            )  # [B,enc-T]
            coverage_mask = self.subnet_unit.add_compare_layer(
                "coverage_mask",
                merge_accum_w,
                kind="greater",
                value=self.coverage_threshold,
            )  # [B,enc-T]
            float_coverage_mask = self.subnet_unit.add_cast_layer(
                "float_coverage_mask", coverage_mask, dtype="float32"
            )  # [B,enc-T]
            accum_coverage = self.subnet_unit.add_reduce_layer(
                "accum_coverage",
                float_coverage_mask,
                mode="sum",
                axes=-1,
                keep_dims=True,
            )  # [B,1]

            self.output_prob = self.subnet_unit.add_eval_layer(
                "output_prob_coverage",
                source=[self.output_prob, accum_coverage],
                eval=f"source(0) * (source(1) ** {self.coverage_scale})",
            )

        choice_opts = dict(target=self.target)
        if not self.length_normalization:
            choice_opts["length_normalization"] = False
        if not self.search_type:
            pass
        elif self.search_type == "end-of-chunk":
            choice_opts["search"] = True
            choice_opts["target"] = None
        else:
            raise ValueError(f"Unknown search type: {self.search_type!r}")
        subnet_unit.add_choice_layer(
            "output",
            self.output_prob,
            beam_size=self.beam_size,
            initial_output=0,
            **choice_opts,
        )

        # recurrent subnetwork
        rec_opts = dict(target=self.target)
        if self.search_type == "end-of-chunk":
            # search_flag is False in training, but we anyway want to search, and we don't want the seq len
            # from the ground truth labels (without EOC labels), so we must not use the target here.
            rec_opts["target"] = None
        if self.enc_chunks_dim:
            assert self.enc_time_dim and self.enc_time_dim.dimension is not None
            rec_opts["include_eos"] = True
            rec_opts[
                "max_seq_len"
            ] = f"max_len_from('base:encoder') * {self.enc_time_dim.dimension}"
        dec_output = self.network.add_subnet_rec_layer(
            "output", unit=subnet_unit.get_net(), source=self.source, **rec_opts
        )

        return dec_output

    def create_network(self):
        self.dec_output = self.add_decoder_subnetwork(self.subnet_unit)

        # Add to Base/Encoder network

        if hasattr(self.base_model, "enc_proj_dim") and self.base_model.enc_proj_dim:
            self.base_model.network.add_copy_layer("enc_ctx", "encoder_proj")
            self.base_model.network.add_split_dim_layer(
                "enc_value",
                "encoder_proj",
                dims=(
                    self.att_num_heads,
                    FeatureDim(
                        "val", self.enc_value_dim // self.att_num_heads.dimension
                    ),
                ),
            )
        else:
            self.base_model.network.add_linear_layer(
                "enc_ctx",
                "encoder",
                with_bias=True,
                n_out=self.enc_key_dim,
                l2=self.base_model.l2,
            )
            self.base_model.network.add_split_dim_layer(
                "enc_value",
                "encoder",
                dims=(
                    self.att_num_heads,
                    FeatureDim(
                        "val", self.enc_value_dim // self.att_num_heads.dimension
                    ),
                ),
            )

        self.base_model.network.add_linear_layer(
            "inv_fertility",
            "encoder",
            activation="sigmoid",
            n_out=self.att_num_heads,
            with_bias=False,
        )

        self.base_model.network["out_best"] = {
            "class": "decide",
            "from": self.dec_output,
        }

        self.base_model.network["enc_seq_len"] = {"class": "length", "from": "encoder"}
        self.base_model.network["targets_seq_len"] = {
            "class": "length",
            "from": f"data:{self.target}",
        }
        self.base_model.network["out_best_seq_len"] = {
            "class": "length",
            "from": "out_best",
        }
        if False:
            for name in [
                "enc_seq_len",
                "out_best",
                "out_best_seq_len",
                f"data:{self.target}",
                "targets_seq_len",
            ]:
                name_ = name.replace("data:", "")
                self.base_model.network[f"debug_print_{name_}"] = {
                    "class": "print",
                    "from": name,
                    "is_output_layer": True,
                }

        # TODO fix this
        # decision_layer_name = self.base_model.network.add_decide_layer(
        #    "decision", self.dec_output, target=self.target
        # )
        self.decision_layer_name = "decision"  # decision_layer_name

        return self.dec_output
