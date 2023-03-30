"""
Model, based on Mohammads code.
"""


from __future__ import annotations
from typing import Optional, Callable
import tensorflow as tf

from returnn.util.basic import NotSpecified
from returnn.tf.util.data import Data, Dim, SpatialDim, FeatureDim, single_step_dim
from returnn.tf.layers.basic import LayerBase

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
            att_weights0 = out_net.add_softmax_over_spatial_layer("att_weights0", energy, **att_sm_opts)
            att_weights = out_net.add_dropout_layer(
                "att_weights",
                att_weights0,
                dropout=self.att_dropout,
                dropout_noise_shape={"*": None},
            )
        else:
            att_weights = out_net.add_softmax_over_spatial_layer("att_weights", energy, **att_sm_opts)

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
            att0 = out_net.add_generic_att_layer("att0", weights=att_weights, base=enc_value)
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
        enable_check_align=True,
        masked_computation_blank_idx: Optional[int] = None,
        full_sum_simple_approx: bool = False,
        prev_target_embed_direct: bool = False,
    ):
        """
        :param base_model: base/encoder model instance
        :param str|None source: input to decoder subnetwork
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
        :param enable_check_align: if set, the targets are checked whether M + U = T
        :param masked_computation_blank_idx: if set, it uses masked computation for the LSTM/prev:target,
            and the mask is for all non-blank indices
        :param full_sum_simple_approx: if enabled, it creates a 4D tensor [B, M, U+1, V] via a simple approximation
            by only attending to one fixed chunk in M for the whole sequence U+1, and then it uses the RNN-T loss.
            The decoder gets only the non-blank labels as input in this case, including BOS (U+1).
            This makes only sense in training. Then in recog, you do align-sync search,
            and you should set masked_computation_blank_idx to get consistent behavior,
            i.e. that blank labels are not used.
        :param prev_target_embed_direct: if False, uses "prev:target_embed",
            otherwise "prev_target_embed" uses "prev:output". should be like "apply(0)" as initial_output.
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

        self.enable_check_align = enable_check_align
        self.masked_computation_blank_idx = masked_computation_blank_idx
        self.full_sum_simple_approx = full_sum_simple_approx
        if full_sum_simple_approx:
            assert enc_chunks_dim is not None, "full_sum_simple_approx requires enc_chunks_dim"
        self.prev_target_embed_direct = prev_target_embed_direct

    def add_decoder_subnetwork(
        self,
        subnet_unit: ReturnnNetwork,
        target: str = NotSpecified,
        search_type: Optional[str] = NotSpecified,
        rec_layer_name: Optional[str] = None,
        rec_layer_opts: Optional[dict] = None,
    ):
        if target is NotSpecified:
            target = self.target
        if search_type is NotSpecified:
            search_type = self.search_type
        if rec_layer_opts is None:
            rec_layer_opts = {}
        if not rec_layer_name:
            if search_type == "end-of-chunk":
                rec_layer_name = "output_align"
                rec_layer_opts.setdefault("name_scope", "output/rec")
            else:
                rec_layer_name = "output"

        if self.full_sum_simple_approx:
            pass
        elif self.enc_chunks_dim:  # use chunking
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
                "from": f"base:data:{target}",
                "axis": "T",
                "position": "label_pos",
                "clip_to_valid": True,
            }

            subnet_unit["ground_truth_label_seq_len"] = {
                "class": "length",
                "from": f"base:data:{target}",
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
            subnet_unit.add_compare_layer("end", source="output", value=self.eos_id)  # sentence end token

        if self.masked_computation_blank_idx is not None:
            subnet_unit["masked_comp_mask"] = {
                "class": "compare",
                "from": "output",
                "kind": "not_equal",
                "value": self.masked_computation_blank_idx,
                "initial_output": True,
            }

        prev_output = "prev:output"
        if self.full_sum_simple_approx:
            assert self.prev_target_embed_direct
            assert not self.source
            prev_output = "data:source"

        # target embedding
        _name = subnet_unit.add_linear_layer(
            "prev_target_embed0" if self.prev_target_embed_direct else "target_embed0",
            prev_output if self.prev_target_embed_direct else "output",
            n_out=self.embed_dim,
            with_bias=False,
            l2=self.l2,
            forward_weights_init=self.embed_weight_init,
            initial_output=self.eos_id,
        )
        if self.masked_computation_blank_idx is not None:
            target_embed_layer_dict = subnet_unit[_name]
            target_embed_layer_dict = {
                "class": "masked_computation",
                "unit": target_embed_layer_dict,
                "mask": "prev:masked_comp_mask" if self.prev_target_embed_direct else "masked_comp_mask",
                "initial_output": self.eos_id,
                "from": target_embed_layer_dict["from"],
            }
            target_embed_layer_dict["unit"]["from"] = "data"
            subnet_unit[_name] = target_embed_layer_dict
        if self.prev_target_embed_direct:
            subnet_unit[_name]["name_scope"] = "target_embed0"  # params compatible

        subnet_unit.add_dropout_layer(
            "prev_target_embed" if self.prev_target_embed_direct else "target_embed",
            _name,
            dropout=self.embed_dropout,
            dropout_noise_shape={"*": None},
        )
        prev_target_embed = "prev_target_embed" if self.prev_target_embed_direct else "prev:target_embed"

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
            att.enc_time_dim = self.enc_time_dim
            if self.full_sum_simple_approx:
                pass
            else:

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
        subnet_unit.update(att.create())

        # LM-like component same as here https://arxiv.org/pdf/2001.07263.pdf
        lstm_lm_component_proj = None
        if self.add_lstm_lm:
            assert self.masked_computation_blank_idx is None  # not implemented...
            lstm_lm_component = subnet_unit.add_rec_layer(
                "lm_like_s",
                prev_target_embed,
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
            lstm_inputs += [prev_target_embed]
        lstm_inputs += ["prev:att"]

        if self.add_lstm_lm:
            # element-wise addition is applied instead of concat
            lstm_inputs = subnet_unit.add_combine_layer(
                "add_embed_ctx", lstm_inputs, kind="add", n_out=self.lstm_lm_proj_dim
            )

        # LSTM decoder (or decoder state)
        if self.dec_zoneout and not self.full_sum_simple_approx:
            # It's bad to use rnn_cell here... Just annoying to keep this just to preserve hash...
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
                unit="zoneoutlstm" if self.dec_zoneout else "NativeLSTM2",
                rec_weight_dropout=self.rec_weight_dropout,
                weights_init=self.lstm_weights_init,
            )
            if self.dec_zoneout:
                subnet_unit["s"].setdefault("unit_opts", {}).update(
                    {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05}
                )
        if self.full_sum_simple_approx:
            subnet_unit["s"]["axis"] = single_step_dim
        if self.masked_computation_blank_idx is not None:
            subnet_unit["_s_input"] = {"class": "copy", "from": lstm_inputs}
            layer_dict = subnet_unit["s"]
            subnet_unit["s"] = {
                "class": "masked_computation",
                "unit": layer_dict,
                "from": "_s_input",
                "mask": "prev:masked_comp_mask",
            }
            layer_dict["from"] = "data"

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

            readout_in_src = subnet_unit.add_combine_layer(
                "add_s_att", [s_name, "att"], kind="add", n_out=self.lstm_lm_proj_dim
            )
        else:
            readout_in_src = [s_name, prev_target_embed, "att"]

        subnet_unit.add_linear_layer("readout_in", readout_in_src, n_out=self.dec_output_num_units, l2=self.l2)

        if self.reduceout:
            subnet_unit.add_reduceout_layer("readout", "readout_in")
        else:
            subnet_unit.add_copy_layer("readout", "readout_in")

        out_prob_opts = {}
        if not search_type and not self.full_sum_simple_approx:
            out_prob_opts.update(
                dict(
                    loss="ce",
                    loss_opts={"label_smoothing": self.label_smoothing},
                )
            )
        self.output_prob = subnet_unit.add_softmax_layer(
            "output_prob",
            "readout",
            l2=self.l2,
            target=f"layer:base:data:{target}"
            if (search_type == "end-of-chunk" or self.full_sum_simple_approx)
            else target,
            dropout=self.softmax_dropout,
            **out_prob_opts,
        )

        if self.full_sum_simple_approx:
            assert self.enc_chunks_dim
            subnet_unit["output_log_prob"] = {
                "class": "activation",
                "from": "output_prob",
                "activation": "safe_log",
            }
            subnet_unit["full_sum_simple_approx_loss"] = {
                "class": "eval",
                "from": ["output_log_prob", f"base:data:{target}"],
                # Pickling/serialization of the func ref should work when this is a global function of this module.
                # But depending on your setup, there might anyway not be any serialization.
                "eval": _rnnt_full_sum_log_prob_eval_layer_func,
                "eval_locals": {
                    "blank_index": self.eos_id,
                    "input_spatial_dim": self.enc_chunks_dim,
                },
                "out_type": _rnnt_full_sum_log_prob_eval_layer_out,
                "loss": "as_is",
            }

        if search_type == "end-of-chunk":
            subnet_unit["_label_indices"] = {
                "class": "range_in_axis",
                "from": f"base:data:{target}",
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

        if self.full_sum_simple_approx:
            subnet_unit.add_copy_layer("output", "data:source")
        else:
            choice_opts = dict(target=target)
            if not self.length_normalization:
                choice_opts["length_normalization"] = False
            if not search_type:
                pass
            elif search_type == "end-of-chunk":
                choice_opts["search"] = True
                choice_opts["target"] = None
            else:
                raise ValueError(f"Unknown search type: {search_type!r}")
            subnet_unit.add_choice_layer(
                "output",
                self.output_prob,
                beam_size=self.beam_size,
                initial_output=0,
                **choice_opts,
            )

        # recurrent subnetwork
        rec_opts = dict(target=target)
        if search_type == "end-of-chunk":
            # search_flag is False in training, but we anyway want to search, and we don't want the seq len
            # from the ground truth labels (without EOC labels), so we must not use the target here.
            rec_opts["target"] = None
        if self.full_sum_simple_approx:
            assert self.prev_target_embed_direct
            self.network["_targets_with_bos"] = {
                "class": "prefix_in_time",
                "from": f"data:{target}",
                "prefix": self.eos_id,
            }
            rec_opts["source"] = "_targets_with_bos"
            rec_opts["target"] = None
        elif self.source:
            rec_opts["source"] = self.source
        if self.enc_chunks_dim:
            assert self.enc_time_dim and self.enc_time_dim.dimension is not None
            rec_opts["include_eos"] = True
            # TODO warning this is wrong, needs to be larger,
            #   but we can't easily change it now because it changes the hash
            rec_opts["max_seq_len"] = f"max_len_from('base:encoder') * {self.enc_time_dim.dimension}"
        if rec_layer_opts:
            rec_opts.update(rec_layer_opts)
        dec_output = self.network.add_subnet_rec_layer(rec_layer_name, unit=subnet_unit.get_net(), **rec_opts)

        return dec_output

    def create_network(self):
        self.dec_output = self.add_decoder_subnetwork(self.subnet_unit)
        target = self.target
        if self.search_type == "end-of-chunk":
            self.base_model.network["_02_alignment_on_the_fly"] = {
                "class": "copy",
                "from": "out_best",
                "register_as_extern_data": "alignment_on_the_fly",
            }
            target = "alignment_on_the_fly"
            # Add another output layer for potential training.
            subnet_unit = ReturnnNetwork()
            self.add_decoder_subnetwork(subnet_unit, search_type=None, target=target)

        # Add to Base/Encoder network

        if hasattr(self.base_model, "enc_proj_dim") and self.base_model.enc_proj_dim:
            self.base_model.network.add_copy_layer("enc_ctx", "encoder_proj")
            self.base_model.network.add_split_dim_layer(
                "enc_value",
                "encoder_proj",
                dims=(
                    self.att_num_heads,
                    FeatureDim("val", self.enc_value_dim // self.att_num_heads.dimension),
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
                    FeatureDim("val", self.enc_value_dim // self.att_num_heads.dimension),
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

        # Filter blank / EOS / EOC
        if not self.full_sum_simple_approx:

            self.base_model.network["out_best_non_blank_mask"] = {
                "class": "compare",
                "from": "out_best",
                "value": self.eos_id,
                "kind": "not_equal",
            }

            self.base_model.network["out_best_wo_blank"] = {
                "class": "masked_computation",
                "mask": "out_best_non_blank_mask",
                "from": "out_best",
                "unit": {"class": "copy"},
            }
            self.decision_layer_name = "out_best_wo_blank"

            self.base_model.network["edit_distance"] = {
                "class": "copy",
                "from": "out_best_wo_blank",
                "only_on_search": True,
                "loss": "edit_distance",
                "target": self.target,
            }

            if self.enc_chunks_dim and self.enable_check_align:
                self.base_model.network["_check_alignment"] = {
                    "class": "eval",
                    "from": "out_best_wo_blank",
                    "eval": _check_alignment,
                    "eval_locals": {"target": target},  # with blank
                    "is_output_layer": True,
                }

        return self.dec_output


# noinspection PyShadowingNames
def _check_alignment(source, self, target, **_kwargs):
    import tensorflow as tf
    from returnn.tf.util.data import Data

    out_wo_blank = source(0, as_data=True)
    assert isinstance(out_wo_blank, Data)
    if not self.network.eval_flag:
        # Targets are not available during recognition.
        return out_wo_blank.placeholder
    out_with_blank = self.network.get_layer(f"data:{target}").output
    assert isinstance(out_with_blank, Data)
    encoder = self.network.get_layer("encoder").output
    assert isinstance(encoder, Data)
    num_chunks = encoder.get_sequence_lengths()
    num_labels_wo_blank = out_wo_blank.get_sequence_lengths()
    num_labels_w_blank = out_with_blank.get_sequence_lengths()
    deps = [
        tf.Assert(
            tf.reduce_all(tf.equal(num_labels_wo_blank + num_chunks, num_labels_w_blank)),
            [
                "num labels wo blank, num chunks, with blank:",
                num_labels_wo_blank,
                num_chunks,
                num_labels_w_blank,
                "labels wo blank, with blank:",
                out_wo_blank.placeholder,
                out_with_blank.placeholder,
            ],
            summarize=100,
        ),
    ]
    self.network.register_post_control_dependencies(deps)
    with tf.control_dependencies(deps):
        return tf.identity(out_wo_blank.placeholder)


# Taken from returnn_common, adopted.
def _rnnt_full_sum_log_prob_eval_layer_func(
    *,
    self: LayerBase,
    source,
    input_spatial_dim: Dim,
    blank_index: int,
) -> tf.Tensor:
    from returnn.tf.util.data import Data
    from returnn.tf.layers.basic import LayerBase
    from returnn.extern.HawkAaronWarpTransducer import rnnt_loss

    assert isinstance(self, LayerBase)
    log_probs = source(0, auto_convert=False, as_data=True)
    labels = source(1, auto_convert=False, as_data=True)
    assert isinstance(log_probs, Data) and isinstance(labels, Data)
    assert labels.batch_ndim == 2 and labels.have_batch_axis() and labels.have_time_axis()
    labels_spatial_dim = labels.get_time_dim_tag()
    prev_labels_spatial_dim = 1 + labels_spatial_dim
    batch_dims = list(self.output.dim_tags)
    feat_dim = log_probs.feature_dim_or_sparse_dim
    if blank_index < 0:
        blank_index += feat_dim.dimension
    assert 0 <= blank_index < feat_dim.dimension
    assert labels.sparse_dim.dimension <= feat_dim.dimension
    # Move axes into the right order (no-op if they already are).
    log_probs = log_probs.copy_compatible_to(
        Data("log_probs", dim_tags=batch_dims + [input_spatial_dim, prev_labels_spatial_dim, feat_dim]),
        check_dtype=False,
    )
    labels = labels.copy_compatible_to(
        Data("labels", dim_tags=batch_dims + [labels_spatial_dim], sparse_dim=labels.sparse_dim), check_dtype=False
    )
    input_lengths = input_spatial_dim.get_dyn_size_ext_for_batch_ctx(
        log_probs.batch, log_probs.control_flow_ctx
    ).copy_compatible_to(Data("input_lengths", dim_tags=batch_dims), check_dtype=False)
    label_lengths = labels_spatial_dim.get_dyn_size_ext_for_batch_ctx(
        log_probs.batch, log_probs.control_flow_ctx
    ).copy_compatible_to(Data("label_lengths", dim_tags=batch_dims), check_dtype=False)

    return rnnt_loss(
        acts=log_probs.placeholder,
        labels=labels.placeholder,
        input_lengths=input_lengths.placeholder,
        label_lengths=label_lengths.placeholder,
        blank_label=blank_index,
    )


def _rnnt_full_sum_log_prob_eval_layer_out(
    *,
    name: str,
    **_kwargs,
) -> Data:
    from returnn.tf.util.data import Data, batch_dim

    return Data("%s_output" % name, dim_tags=[batch_dim])
