from i6_core.returnn.config import CodeWrapper
from typing import Optional, Dict, List
import copy


def get_source_layers(from_layer: str):
  return {
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('%s')(source(0, as_data=True), network=self.network)" % ("transform"),
      "from": from_layer
    },
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)
  }


def get_on_the_fly_feature_extraction(
        stft_frame_shift: int,
        stft_frame_size: int,
        stft_fft_size: int,
        mel_filterbank_fft_size: int,
        mel_filterbank_nr_of_filters: int,
        mel_filterbank_n_out: int,
        mel_filterbank_sampling_rate: int,
        mel_filterbank_f_min: int,
        mel_filterbank_f_max: int,
        log10_type: int,
):
  feature_extraction_dict = {
    "stft": {
      "class": "stft",
      "frame_shift": stft_frame_shift,
      "frame_size": stft_frame_size,
      "fft_size": stft_fft_size,
      "from": "data:data",
    },
    "abs": {"class": "activation", "from": "stft", "activation": "abs"},
    "power": {"class": "eval", "from": "abs", "eval": "source(0) ** 2"},
    "mel_filterbank": {
      "class": "mel_filterbank",
      "from": "power",
      "fft_size": mel_filterbank_fft_size,
      "nr_of_filters": mel_filterbank_nr_of_filters,
      "n_out": mel_filterbank_n_out,
      "sampling_rate": mel_filterbank_sampling_rate,
      "f_min": mel_filterbank_f_min,
      "f_max": mel_filterbank_f_max
    },
    "log": {
      "from": "mel_filterbank",
      "class": "activation",
      "activation": "safe_log",
      "opts": {"eps": 1e-10},
    },
    "log_mel_features": {"class": "copy", "from": "log10"},
  }

  if log10_type == 1:
    feature_extraction_dict.update({
      "log10": {"from": "log", "class": "eval", "eval": "source(0) / 2.3026"}
    })
  else:
    assert log10_type == 2
    feature_extraction_dict.update({
      "log10": {
        "class": "eval",
        "from": ["log10_", "global_mean", "global_stddev"],
        "eval": "(source(0) - source(1)) / source(2)",
      },
      "global_mean": {
        "class": "eval",
        "eval": "exec('import numpy') or "
                "numpy.loadtxt('/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.faDEtM4G5zug/output/mean', "
                "dtype='float32') + (source(0) - source(0))",
        "from": "log10_",
      },
      "global_stddev": {
        "class": "eval",
        "eval": "exec('import numpy') or "
                "numpy.loadtxt('/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.faDEtM4G5zug/output/std_dev', "
                "dtype='float32') + (source(0) - source(0))",
        "from": "log10_",
      },
      "log10_": {"from": "log", "class": "eval", "eval": "source(0) / 2.3026"},
    })

  return feature_extraction_dict


def get_conformer_encoder(
        num_blocks: int,
        source_linear_n_out: int,
        conformer_big_dim: int,
        conformer_small_dim: int,
        conformer_rel_pos_enc_dim: int,
        conformer_self_att_num_heads: int,
        conformer_depthwise_conv_filter_size: int
):
  conformer_dict = {
    "conv0": {
      "class": "conv",
      "from": "source0",
      "padding": "same",
      "filter_size": (3, 3),
      "n_out": 32,
      "activation": "relu",
      "with_bias": True,
    },
    "conv0p": {
      "class": "pool",
      "from": "conv0",
      "pool_size": (1, 2),
      "mode": "max",
      "trainable": False,
      "padding": "same",
    },
    "conv_out": {"class": "copy", "from": "conv0p"},
    "subsample_conv0": {
      "class": "conv",
      "from": "conv_out",
      "padding": "same",
      "filter_size": (3, 3),
      "n_out": 64,
      "activation": "relu",
      "with_bias": True,
      "strides": (3, 1),
    },
    "subsample_conv1": {
      "class": "conv",
      "from": "subsample_conv0",
      "padding": "same",
      "filter_size": (3, 3),
      "n_out": 64,
      "activation": "relu",
      "with_bias": True,
      "strides": (2, 1),
    },
    "conv_merged": {"class": "merge_dims", "from": "subsample_conv1", "axes": "static"},
    "source_linear": {
      "class": "linear",
      "activation": None,
      "with_bias": False,
      "from": "conv_merged",
      "n_out": source_linear_n_out,
      "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    },
  }

  src_layer = "source_linear"
  for i in range(1, num_blocks+1):
    conformer_dict.update({
      "conformer_block_01_ffmod_1_ln": {"class": "layer_norm", "from": src_layer},
      "conformer_block_01_ffmod_1_ff1": {
          "class": "linear",
          "activation": None,
          "with_bias": True,
          "from": "conformer_block_01_ffmod_1_ln",
          "n_out": conformer_big_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
      },
      "conformer_block_01_ffmod_1_relu": {
          "class": "activation",
          "activation": "relu",
          "from": "conformer_block_01_ffmod_1_ff1",
      },
      "conformer_block_01_ffmod_1_square_relu": {
          "class": "eval",
          "eval": "source(0) ** 2",
          "from": "conformer_block_01_ffmod_1_relu",
      },
      "conformer_block_01_ffmod_1_drop1": {
          "class": "dropout",
          "from": "conformer_block_01_ffmod_1_square_relu",
          "dropout": 0.0,
      },
      "conformer_block_01_ffmod_1_ff2": {
          "class": "linear",
          "activation": None,
          "with_bias": True,
          "from": "conformer_block_01_ffmod_1_drop1",
          "n_out": conformer_small_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
      },
      "conformer_block_01_ffmod_1_drop2": {
          "class": "dropout",
          "from": "conformer_block_01_ffmod_1_ff2",
          "dropout": 0.0,
      },
      "conformer_block_01_ffmod_1_half_step": {
          "class": "eval",
          "eval": "0.5 * source(0)",
          "from": "conformer_block_01_ffmod_1_drop2",
      },
      "conformer_block_01_ffmod_1_res": {
          "class": "combine",
          "kind": "add",
          "from": ["conformer_block_01_ffmod_1_half_step", "source_linear"],
          "n_out": conformer_small_dim,
      },
      "conformer_block_01_self_att_ln": {
          "class": "layer_norm",
          "from": "conformer_block_01_ffmod_1_res",
      },
      "conformer_block_01_self_att_ln_rel_pos_enc": {
          "class": "relative_positional_encoding",
          "from": "conformer_block_01_self_att_ln",
          "n_out": conformer_rel_pos_enc_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
          "scale=1.0)",
          "clipping": 16,
      },
      "conformer_block_01_self_att": {
          "class": "self_attention",
          "from": "conformer_block_01_self_att_ln",
          "n_out": conformer_small_dim,
          "num_heads": conformer_self_att_num_heads,
          "total_key_dim": conformer_small_dim,
          "key_shift": "conformer_block_01_self_att_ln_rel_pos_enc",
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)",
      },
      "conformer_block_01_self_att_linear": {
          "class": "linear",
          "activation": None,
          "with_bias": False,
          "from": "conformer_block_01_self_att",
          "n_out": conformer_small_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
      },
      "conformer_block_01_self_att_dropout": {
          "class": "dropout",
          "from": "conformer_block_01_self_att_linear",
          "dropout": 0.0,
      },
      "conformer_block_01_self_att_res": {
          "class": "combine",
          "kind": "add",
          "from": [
              "conformer_block_01_self_att_dropout",
              "conformer_block_01_ffmod_1_res",
          ],
          "n_out": conformer_small_dim,
      },
      "conformer_block_01_conv_mod_ln": {
          "class": "layer_norm",
          "from": "conformer_block_01_self_att_res",
      },
      "conformer_block_01_conv_mod_pointwise_conv1": {
          "class": "linear",
          "activation": None,
          "with_bias": True,
          "from": "conformer_block_01_conv_mod_ln",
          "n_out": conformer_small_dim * 2,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
          "scale=1.0)",
      },
      "conformer_block_01_conv_mod_glu": {
          "class": "gating",
          "from": "conformer_block_01_conv_mod_pointwise_conv1",
          "activation": "identity",
      },
      "conformer_block_01_conv_mod_depthwise_conv2": {
          "class": "conv",
          "from": "conformer_block_01_conv_mod_glu",
          "padding": "same",
          "filter_size": (conformer_depthwise_conv_filter_size,),
          "n_out": conformer_small_dim,
          "activation": None,
          "with_bias": True,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
          "scale=1.0)",
          "groups": conformer_small_dim,
      },
      "conformer_block_01_conv_mod_bn": {
          "class": "batch_norm",
          "from": "conformer_block_01_conv_mod_depthwise_conv2",
          "momentum": 0.1,
          "epsilon": 0.001,
          "update_sample_only_in_training": True,
          "delay_sample_update": True,
      },
      "conformer_block_01_conv_mod_swish": {
          "class": "activation",
          "activation": "swish",
          "from": "conformer_block_01_conv_mod_bn",
      },
      "conformer_block_01_conv_mod_pointwise_conv2": {
          "class": "linear",
          "activation": None,
          "with_bias": True,
          "from": "conformer_block_01_conv_mod_swish",
          "n_out": conformer_small_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', "
          "scale=1.0)",
      },
      "conformer_block_01_conv_mod_drop": {
          "class": "dropout",
          "from": "conformer_block_01_conv_mod_pointwise_conv2",
          "dropout": 0.0,
      },
      "conformer_block_01_conv_mod_res": {
          "class": "combine",
          "kind": "add",
          "from": ["conformer_block_01_conv_mod_drop", "conformer_block_01_self_att_res"],
          "n_out": conformer_small_dim,
      },
      "conformer_block_01_ffmod_2_ln": {
          "class": "layer_norm",
          "from": "conformer_block_01_conv_mod_res",
      },
      "conformer_block_01_ffmod_2_ff1": {
          "class": "linear",
          "activation": None,
          "with_bias": True,
          "from": "conformer_block_01_ffmod_2_ln",
          "n_out": conformer_big_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
      },
      "conformer_block_01_ffmod_2_relu": {
          "class": "activation",
          "activation": "relu",
          "from": "conformer_block_01_ffmod_2_ff1",
      },
      "conformer_block_01_ffmod_2_square_relu": {
          "class": "eval",
          "eval": "source(0) ** 2",
          "from": "conformer_block_01_ffmod_2_relu",
      },
      "conformer_block_01_ffmod_2_drop1": {
          "class": "dropout",
          "from": "conformer_block_01_ffmod_2_square_relu",
          "dropout": 0.0,
      },
      "conformer_block_01_ffmod_2_ff2": {
          "class": "linear",
          "activation": None,
          "with_bias": True,
          "from": "conformer_block_01_ffmod_2_drop1",
          "n_out": conformer_small_dim,
          "forward_weights_init": "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
      },
      "conformer_block_01_ffmod_2_drop2": {
          "class": "dropout",
          "from": "conformer_block_01_ffmod_2_ff2",
          "dropout": 0.0,
      },
      "conformer_block_01_ffmod_2_half_step": {
          "class": "eval",
          "eval": "0.5 * source(0)",
          "from": "conformer_block_01_ffmod_2_drop2",
      },
      "conformer_block_01_ffmod_2_res": {
          "class": "combine",
          "kind": "add",
          "from": [
              "conformer_block_01_ffmod_2_half_step",
              "conformer_block_01_conv_mod_res",
          ],
          "n_out": conformer_small_dim,
      },
      "conformer_block_01_ln": {
          "class": "layer_norm",
          "from": "conformer_block_01_ffmod_2_res",
      },
      "conformer_block_01": {"class": "copy", "from": "conformer_block_01_ln"},
    })

    src_layer = "conformer_block_%02d" % i

  return conformer_dict


def get_length_model_unit_dict(task: str):
  length_model_unit_dict = {
    "am": {"class": "copy", "from": "data:source"},
    "blank_log_prob": {
      "class": "eval",
      "eval": "tf.math.log_sigmoid(-source(0))",
      "from": "emit_prob0",
    },
    "const1": {"class": "constant", "value": 1},
    "emit_log_prob": {
      "activation": "log_sigmoid",
      "class": "activation",
      "from": "emit_prob0",
    },
    "emit_prob0": {
      "activation": None,
      "class": "linear",
      "from": "s",
      "is_output_layer": True,
      "n_out": 1,
    },
    "output_emit": {
      "class": "compare",
      "from": "output",
      "initial_output": True,
      "kind": "not_equal",
      "value": 1030,
    },
    "prev_out_embed": {
      "activation": None,
      "class": "linear",
      "from": "prev:output",
      "n_out": 128,
    },
    "s": {
      "L2": 0.0001,
      "class": "rec",
      "dropout": 0.3,
      "from": ["am", "prev_out_embed"],
      "n_out": 128,
      "unit": "nativelstm2",
      "unit_opts": {"rec_weight_dropout": 0.3},
    },
  }

  if task == "train":
    length_model_unit_dict.update({
      "emit_blank_log_prob": {
        "class": "copy",
        "from": ["blank_log_prob", "emit_log_prob"],
      },
      "emit_blank_prob": {
        "activation": "exp",
        "class": "activation",
        "from": "emit_blank_log_prob",
        "loss": "ce",
        "loss_opts": {"focal_loss_factor": 0.0},
        "target": "emit_ground_truth",
      },
      "output": {
        "beam_size": 4,
        "cheating": "exclusive",
        "class": "choice",
        "from": "data",
        "initial_output": 0,
        "input_type": "log_prob",
        "target": "targetb",
      },
    })

  return length_model_unit_dict


def get_output_log_prob(output_prob_layer_name: str):
  return {
    "class": "eval",
    "from": output_prob_layer_name,
    "eval": "tf.math.log(source(0))"
  }


def add_is_last_frame_condition(network: Dict, rec_layer_name: str):
  network[rec_layer_name]["unit"].update({
    "encoder_len": {
      "class": "length",
      "from": "base:encoder"
    },
    "is_last_frame": {
      "class": "eval",
      "from": [":i", "encoder_len"],
      "eval": "tf.math.equal(source(0), source(1) - 1)",
      "out_type": {"dtype": "bool"},
    },
  })


def add_center_positions(network: Dict, segment_lens_starts_layer_name: str):
  """
  For center-window models, add a layer returning the center position of the windows.
  This is defined as "segment_starts1" + "segment_lens1".

  If this is used with a normal segmental model, then the "center_positions" layer will correspond to the last frame of
  the segments.
  :param network:
  :return:
  """
  network[segment_lens_starts_layer_name]["unit"].update({
    "center_positions": {
      "class": "eval",
      "from": ["segment_starts1", "segment_lens1"],
      "eval": "source(0) + source(1)",
      "is_output_layer": True,
    }
  })

  if "label_model" in network:
    network.update({
      "center_positions_masked": {
        "class": "masked_computation",
        "from": "%s/center_positions" % segment_lens_starts_layer_name,
        "mask": "is_label",
        "unit": {"class": "copy", "from": "data"},
      },
    })
    network["label_model"]["unit"].update({
      "center_positions": {
        "axis": "t",
        "class": "gather",
        "from": "base:center_positions_masked",
        "position": ":i",
      },
    })


def add_abs_segment_positions(network: Dict, rec_layer_name: str, att_t_dim_tag_code_wrapper: CodeWrapper):
  """
  For segmental models, add a layer which computes the absolute (w.r.t. T) positions of frames within a segment.
  E.g. if a segment is 5 frames long and starts at time 4, the layer outputs [4, 5, 6, 7, 8]

  :param network:
  :return:
  """

  # make sure the network looks like we expect
  assert "att_weights" in network[rec_layer_name]["unit"]

  network[rec_layer_name]["unit"].update({
    "segment_abs_positions0": {
      "class": "range_in_axis",
      "from": "att_weights",
      "axis": att_t_dim_tag_code_wrapper,
    },
    "segment_abs_positions": {
      "class": "eval",
      "from": ["segment_abs_positions0", "segment_starts"],
      "eval": "source(0) + source(1)"
    },
  })


def get_explicit_lstm(layer_name: str, n_out: int, from_: List[str]):
  input_ = "%s_input" % layer_name
  input_gate_ = "%s_input_gate" % layer_name
  forget_gate_ = "%s_forget_gate" % layer_name
  output_gate_ = "%s_output_gate" % layer_name
  cell_in_ = "%s_cell_in" % layer_name
  c_ = "%s_c" % layer_name
  output_ = layer_name

  lstm_dict = {
    input_: {
      "class": "copy",
      "from": ["prev:%s" % output_] + from_
    },
    input_gate_: {
      "class": "linear",
      "from": input_,
      "activation": "sigmoid",
      "n_out": n_out
    },
    forget_gate_: {
      "class": "linear",
      "from": input_,
      "activation": "sigmoid",
      "n_out": n_out
    },
    output_gate_: {
      "class": "linear",
      "from": input_,
      "activation": "sigmoid",
      "n_out": n_out
    },
    cell_in_: {
      "class": "linear",
      "from": input_,
      "activation": "tanh",
      "n_out": n_out
    },
    c_: {
      "class": "eval",
      "from": [input_gate_, cell_in_, forget_gate_, "prev:%s" % c_],
      "eval": "source(0) * source(1) + source(2) * source(3)"
    },
    output_: {
      "class": "eval",
      "from": [output_gate_, c_],
      "eval": "source(0) * source(1)"
    },
  }

  return lstm_dict


def add_att_weights_center_of_gravity(
        network: Dict,
        rec_layer_name: str,
        att_t_dim_tag_code_wrapper: Optional[CodeWrapper] = None,
        global_att: bool = False
):
  """
  Add a layer which computes the center of gravity of the attention weights.
  I.e. sum_t [alpha_t * t], where the t are absolute encoder positions (i.e. t \in [0, T))
  :param network:
  :param rec_layer_name:
  :param att_t_dim_tag_code_wrapper:
  :param global_att:
  :return:
  """
  # make sure the network looks like we expect
  assert "att_weights" in network[rec_layer_name]["unit"]
  if global_att:
    network.update({
      "encoder_range": {
        "class": "range_in_axis",
        "from": "encoder",
        "axis": "t"
      }
    })
    att_weight_position_layer = "base:encoder_range"
    att_weight_reduce_axis = "t"
  else:
    add_abs_segment_positions(network, rec_layer_name, att_t_dim_tag_code_wrapper)
    att_weight_position_layer = "segment_abs_positions"
    att_weight_reduce_axis = att_t_dim_tag_code_wrapper

  network[rec_layer_name]["unit"].update({
    "weighted_segment_abs_positions": {
      "class": "eval",
      "from": [att_weight_position_layer, "att_weights"],
      "eval": "tf.cast(source(0), tf.float32) * source(1)"
    },
    "att_weights_center_of_gravity": {
      "class": "reduce",
      "mode": "sum",
      "from": "weighted_segment_abs_positions",
      "axis": att_weight_reduce_axis
    },
  })


def modify_decoder(
        version: int,
        net_dict: Dict,
        rec_layer_name: str,
        target_num_labels: int,
        masked_computation: bool,
        train: bool,
        python_prolog: Optional[List] = None,
):
  """
  Modify the decoder part of the network.
  V1: Simply remove the att vector from the LSTM input.
  V2: Like https://arxiv.org/abs/2404.01716 but without CE loss on non-blank predictor.
  V3: Like V3 but with CE loss on non-blank predictor.
  :param version:
  :param net_dict:
  :param rec_layer_name:
  :param target_num_labels:
  :param masked_computation:
  :param train:
  :param python_prolog:
  :return:
  """

  if masked_computation:
    assert not train, "expected train=False for masked computation"
    net_dict[rec_layer_name]["unit"]["s_masked"]["unit"]["subnetwork"]["s"]["name_scope"] = "/output/rec/s_wo_att/rec"
    s_layer = net_dict[rec_layer_name]["unit"]["s_masked"]["unit"]["subnetwork"]["s"]
  else:
    if train:
      net_dict[rec_layer_name]["unit"]["s_wo_att"] = copy.deepcopy(net_dict[rec_layer_name]["unit"]["s"])
      del net_dict[rec_layer_name]["unit"]["s"]
      s_layer = net_dict[rec_layer_name]["unit"]["s_wo_att"]
      net_dict[rec_layer_name]["unit"]["s"] = {
        "class": "copy",
        "from": "s_wo_att"
      }
    else:
      net_dict[rec_layer_name]["unit"]["s"]["name_scope"] = "/output/rec/s_wo_att/rec"
      s_layer = net_dict[rec_layer_name]["unit"]["s"]

  if version in (1, 2, 3):
    if masked_computation:
      # before: ["data", "prev:att"]
      s_layer["from"] = ["data"]
    else:
      # before: ["prev:target_embed", "prev:att"]
      s_layer["from"] = ["prev:target_embed"]
    # before: "except_batch" -> does not work since att layer is optimized out of the loop now
    net_dict[rec_layer_name]["unit"]["att"]["axes"] = "except_time"
  if version in (2, 3):
    ####### Label Model #######
    # project directly to the target labels
    s_layer["n_out"] = target_num_labels
    # project att to the target labels and apply log_softmax to att and LSTM
    net_dict[rec_layer_name]["unit"].update({
      "s_log_softmax": {
        "class": "activation",
        "activation": "log_softmax",
        "from": "s"
      },
      "att_log_softmax": {
        "class": "linear",
        "from": "att",
        "n_out": target_num_labels,
        "activation": "log_softmax"
      }
    })
    # add the output (log) prob layer
    if rec_layer_name == "label_model":
      net_dict[rec_layer_name]["unit"]["output_prob"] = {
        "class": "eval",
        "from": ["s_log_softmax", "att_log_softmax"],
        "eval": "source(0) + source(1)",
        "activation": "softmax",
        "loss": "ce",
        "loss_opts": {"label_smoothing": 0.1},
        "target": net_dict[rec_layer_name]["unit"]["output_prob"]["target"]
      }
    else:
      assert rec_layer_name == "output"
      net_dict[rec_layer_name]["unit"]["label_log_prob"] = {
        "class": "eval",
        "from": ["s_log_softmax", "att_log_softmax"],
        "eval": "source(0) + source(1)",
        "activation": "log_softmax",
      }
    # readout layer not needed anymore
    del net_dict[rec_layer_name]["unit"]["readout_in"]
    del net_dict[rec_layer_name]["unit"]["readout"]
    ####### Length Model #######
    net_dict["output"]["unit"]["s_length_model"]["from"] = ["prev:target_embed_length_model"]
    net_dict["output"]["unit"]["s_length_model"]["n_out"] = 512
    net_dict["output"]["unit"].update({
      "s_length_model_plus_encoder": {
        "class": "eval",
        "from": ["s_length_model", "am"],
        "eval": "source(0) + source(1)"
      }
    })
    net_dict["output"]["unit"]["emit_prob0"]["from"] = "s_length_model_plus_encoder"
  if version == 3 and train:
    net_dict[rec_layer_name]["unit"]["s_softmax"] = {
      "class": "activation",
      "from": "s_log_softmax",
      "activation": "exp",
      "loss": "ce",
      "target": net_dict[rec_layer_name]["unit"]["output_prob"]["target"]
    }
  if version not in (1, 2, 3):
    raise ValueError("Invalid version %d" % version)


def add_length_model_pos_probs(
        network: Dict,
        rec_layer_name: str,
        use_normalization: bool,
        att_t_dim_tag: CodeWrapper,
        blank_log_prob_dim_tag: CodeWrapper,
        segment_lens_starts_layer_name: str,
):
  """
  For segmental models, add a layer which, for each frame in a segment, computes the length model probability of this
  frame being an "emit" frame.
  E.g. p(t_s = t) = [ \prod_{t' = t_{s-1} + 1}^{t-1} p(t' = "blank") ] * p(t = "emit")

  :param network:
  :return:
  """
  # for now, this is only implemented for the case that the length model is independent of alignment/label context
  # otherwise this cannot be implemented in an efficient way
  if network["output"]["unit"]["s_length_model"]["from"] != ["am"]:
    raise NotImplementedError

  add_abs_segment_positions(network, rec_layer_name, att_t_dim_tag_code_wrapper=att_t_dim_tag)
  add_center_positions(network, segment_lens_starts_layer_name=segment_lens_starts_layer_name)

  network["output"]["unit"]["blank_log_prob"]["is_output_layer"] = True
  network["output"]["unit"]["emit_log_prob"]["is_output_layer"] = True
  network[rec_layer_name]["unit"]["segments"]["out_spatial_dim"] = att_t_dim_tag

  if rec_layer_name == "label_model":
    network[rec_layer_name]["unit"].update({
      # get all the blank log probs from the prev center pos to the second last frame of the segment
      "blank_log_prob_size": {
        "class": "eval",
        "from": ["segment_starts", "segment_lens", "prev:center_positions"],
        "eval": "source(0) + source(1) - source(2)"
      },
      "blank_log_prob0": {
        "class": "slice_nd",
        "from": "base:output/blank_log_prob",
        "out_spatial_dim": blank_log_prob_dim_tag,
        "size": "blank_log_prob_size",
        "start": "prev:center_positions"
      },
      # set the first of these blank log probs to 0.0
      "blank_log_prob_mask_range": {
        "class": "range_in_axis",
        "from": "blank_log_prob0",
        "axis": blank_log_prob_dim_tag
      },
      "blank_log_prob_mask": {
        "class": "compare",
        "from": "blank_log_prob_mask_range",
        "value": 0,
        "kind": "greater"
      },
      "blank_log_prob": {
        "class": "switch",
        "condition": "blank_log_prob_mask",
        "true_from": "blank_log_prob0",
        "false_from": 0.0
      },
      # cumulatively sum of all of the blank log probs
      # this corresponds to the first product of the formula in the function documentation
      # the initial 0.0 in the blank log probs corresponds to the case that the first frame after the last boundary
      # is an "emit" frame
      "blank_log_prob_cum_sum0": {
        "class": "cumsum",
        "from": "blank_log_prob",
        "axis": blank_log_prob_dim_tag
      },
      # get the cum blank log probs corresponding to the current segment
      # in case the segment is overlapping with the last one, we slice from positions which are left of the actual start
      # of the blank_log_prob_cum_sum0 layer. in this case, the slice-nd layer clips the gather indices to 0. later, we
      # then set the log probs of these "invalid" indices to -inf
      "blank_log_prob_cum_sum_left_bound": {
        "class": "eval",
        "from": ["segment_starts", "prev:center_positions"],
        "eval": "source(0) - source(1) - 1"
      },
      "blank_log_prob_cum_sum": {
        "class": "slice_nd",
        "from": "blank_log_prob_cum_sum0",
        "axis": blank_log_prob_dim_tag,
        "out_spatial_dim": att_t_dim_tag,
        "size": "segment_lens",
        "start": "blank_log_prob_cum_sum_left_bound"
      },
      # get the emit log probs corresponding to the current segment
      "emit_log_prob": {
        "class": "slice_nd",
        "from": "base:output/emit_log_prob",
        "out_spatial_dim": att_t_dim_tag,
        "size": "segment_lens",
        "start": "segment_starts"
      },
      # get the final length model probs by adding the emit log prob with the cum blank log probs
      "label_sync_pos_log_prob0": {
        "class": "eval",
        "from": ["blank_log_prob_cum_sum", "emit_log_prob"],
        "eval": "source(0) + source(1)"
      },
      # set the prob for the invalid positions to 0.0
      "label_sync_pos_valid_start_idx": {
        "class": "eval",
        "from": ["segment_starts", "segment_lens", "prev:center_positions"],
        "eval": "source(1) - (source(0) + source(1) - source(2) - 1)"
      },
      "label_sync_pos_range": {
        "class": "range_in_axis",
        "from": "label_sync_pos_log_prob0",
        "axis": att_t_dim_tag
      },
      "label_sync_pos_valid_mask": {
        "class": "compare",
        "from": ["label_sync_pos_range", "label_sync_pos_valid_start_idx"],
        "kind": "greater_equal"
      },
      "label_sync_pos_log_prob": {
        "class": "switch",
        "condition": "label_sync_pos_valid_mask",
        "true_from": "label_sync_pos_log_prob0",
        "false_from": CodeWrapper('float("-inf")')
      },
    })

    if use_normalization:
      # normalize with softmax
      network[rec_layer_name]["unit"].update({
        "label_sync_pos_prob": {
          "class": "softmax_over_spatial",
          "from": "label_sync_pos_log_prob",
          "axis": att_t_dim_tag
        }
      })
    else:
      # just use exp without normalization
      network[rec_layer_name]["unit"].update({
        "label_sync_pos_prob": {
          "class": "activation",
          "from": "label_sync_pos_log_prob",
          "activation": "exp"
        }
      })
  else:
    assert rec_layer_name == "output"
    network["length_model_probs"] = {
      "back_prop": False,
      "class": "rec",
      "from": "encoder",
      "include_eos": True,
      "max_seq_len": "max_len_from('base:encoder')",
      "size_target": None,
      "target": "targets",
      "is_output_layer": True,
      "unit": {
        "am": {"class": "copy", "from": "data:source"},
        "blank_log_prob": {
          "class": "eval",
          "eval": "tf.math.log_sigmoid(-source(0))",
          "from": "emit_prob0",
          "is_output_layer": True
        },
        "emit_log_prob": {
          "activation": "log_sigmoid",
          "class": "activation",
          "from": "emit_prob0",
          "is_output_layer": True
        },
        "emit_prob0": {
          "activation": None,
          "class": "linear",
          "from": "s_length_model",
          "n_out": 1,
          "name_scope": "/output/rec/emit_prob0"
        },
        "output": {
          "class": "copy",
          "from": "am"
        },
        "s_length_model": {
          "L2": 0.0001,
          "class": "rec",
          "dropout": 0.3,
          "from": ["am"],
          "n_out": 128,
          "unit": "nativelstm2",
          "unit_opts": {"rec_weight_dropout": 0.3},
          "name_scope": "/output/rec/s_length_model/rec"
        },
      }
    }
    network[rec_layer_name]["unit"]["blank_log_prob"] = {
      "class": "gather",
      "from": "base:length_model_probs/blank_log_prob",
      "position": ":i",
      "axis": "t"
    }
    network[rec_layer_name]["unit"]["emit_log_prob"] = {
      "class": "gather",
      "from": "base:length_model_probs/emit_log_prob",
      "position": ":i",
      "axis": "t"
    }
    del network[rec_layer_name]["unit"]["emit_prob0"]
    network[rec_layer_name]["unit"].update({
      # get all the blank log probs from the prev center pos to the second last frame of the segment
      "accum_blank_log_prob_size": {
        "class": "eval",
        "from": ["segment_starts", "segment_lens", "prev:center_positions"],
        "eval": "source(0) + source(1) - source(2)"
      },
      "accum_blank_log_prob0": {
        "class": "slice_nd",
        "from": "base:length_model_probs/blank_log_prob",
        "out_spatial_dim": blank_log_prob_dim_tag,
        "size": "accum_blank_log_prob_size",
        "start": "prev:center_positions"
      },
      # set the first of these blank log probs to 0.0
      "accum_blank_log_prob_mask_range": {
        "class": "range_in_axis",
        "from": "accum_blank_log_prob0",
        "axis": blank_log_prob_dim_tag
      },
      "accum_blank_log_prob_mask": {
        "class": "compare",
        "from": "accum_blank_log_prob_mask_range",
        "value": 0,
        "kind": "greater"
      },
      "accum_blank_log_prob": {
        "class": "switch",
        "condition": "accum_blank_log_prob_mask",
        "true_from": "accum_blank_log_prob0",
        "false_from": 0.0
      },
      # cumulatively sum of all of the blank log probs
      # this corresponds to the first product of the formula in the function documentation
      # the initial 0.0 in the blank log probs corresponds to the case that the first frame after the last boundary
      # is an "emit" frame
      "blank_log_prob_cum_sum0": {
        "class": "cumsum",
        "from": "accum_blank_log_prob",
        "axis": blank_log_prob_dim_tag
      },
      # get the cum blank log probs corresponding to the current segment
      # in case the segment is overlapping with the last one, we slice from positions which are left of the actual start
      # of the blank_log_prob_cum_sum0 layer. in this case, the slice-nd layer clips the gather indices to 0. later, we
      # then set the log probs of these "invalid" indices to -inf
      "blank_log_prob_cum_sum_left_bound": {
        "class": "eval",
        "from": ["segment_starts", "prev:center_positions"],
        "eval": "source(0) - source(1) - 1"
      },
      "blank_log_prob_cum_sum": {
        "class": "slice_nd",
        "from": "blank_log_prob_cum_sum0",
        "axis": blank_log_prob_dim_tag,
        "out_spatial_dim": att_t_dim_tag,
        "size": "segment_lens",
        "start": "blank_log_prob_cum_sum_left_bound"
      },
      # get the emit log probs corresponding to the current segment
      "accum_emit_log_prob": {
        "class": "slice_nd",
        "from": "base:length_model_probs/emit_log_prob",
        "out_spatial_dim": att_t_dim_tag,
        "size": "segment_lens",
        "start": "segment_starts"
      },
      # get the final length model probs by adding the emit log prob with the cum blank log probs
      "label_sync_pos_log_prob0": {
        "class": "eval",
        "from": ["blank_log_prob_cum_sum", "accum_emit_log_prob"],
        "eval": "source(0) + source(1)"
      },
      # set the prob for the invalid positions to 0.0
      "label_sync_pos_valid_start_idx": {
        "class": "eval",
        "from": ["segment_starts", "segment_lens", "prev:center_positions"],
        "eval": "source(1) - (source(0) + source(1) - source(2) - 1)"
      },
      "label_sync_pos_range": {
        "class": "range_in_axis",
        "from": "label_sync_pos_log_prob0",
        "axis": att_t_dim_tag
      },
      "label_sync_pos_valid_mask": {
        "class": "compare",
        "from": ["label_sync_pos_range", "label_sync_pos_valid_start_idx"],
        "kind": "greater_equal"
      },
      "label_sync_pos_log_prob": {
        "class": "switch",
        "condition": "label_sync_pos_valid_mask",
        "true_from": "label_sync_pos_log_prob0",
        "false_from": CodeWrapper('float("-inf")')
      },
    })

    if use_normalization:
      # normalize with softmax
      network[rec_layer_name]["unit"].update({
        "label_sync_pos_prob": {
          "class": "softmax_over_spatial",
          "from": "label_sync_pos_log_prob",
          "axis": att_t_dim_tag
        }
      })
    else:
      # just use exp without normalization
      network[rec_layer_name]["unit"].update({
        "label_sync_pos_prob": {
          "class": "activation",
          "from": "label_sync_pos_log_prob",
          "activation": "exp"
        }
      })


def add_att_weight_interpolation(
        network: Dict, rec_layer_name: str, interpolation_layer_name: str, interpolation_scale: float):
  # just to make sure the network looks as we expect
  assert "att_weights0" not in network[rec_layer_name]["unit"]
  assert network[rec_layer_name]["unit"]["att_weights"]["class"] == "softmax_over_spatial"

  network[rec_layer_name]["unit"].update({
    "att_weights0": copy.deepcopy(network[rec_layer_name]["unit"]["att_weights"]),
    "att_weights": {
      "class": "eval",
      "from": ["att_weights0", interpolation_layer_name],
      "eval": "{interpolation_scale} * source(1) + (1 - {interpolation_scale}) * source(0)".format(
        interpolation_scale=interpolation_scale)
    },
  })


def add_ctc_shallow_fusion(network: Dict, rec_layer_name: str, ctc_scale: float, target_num_labels_w_blank: int):

  assert "s_length_model" in network["output"]["unit"], "This function is only supported for our segmental model for now"
  assert network["output"]["unit"]["output_log_prob"]["from"] == [
    "label_log_prob_plus_emit", "blank_log_prob"], "output_log_prob layer does not look as expected"
  assert 0 <= ctc_scale <= 1, "CTC scale must be in [0, 1]"

  del network["ctc"]["loss"]
  del network["ctc"]["loss_opts"]
  del network["ctc"]["loss_scale"]
  del network["ctc"]["target"]
  network["ctc"]["n_out"] = target_num_labels_w_blank

  network["output"]["unit"]["output_log_prob0"] = copy.deepcopy(network["output"]["unit"]["output_log_prob"])

  network[rec_layer_name]["unit"].update({
    "gather_ctc_prob": {
      "class": "gather",
      "from": "base:ctc",
      "position": ":i",
      "axis": "t"
    },
    "output_log_prob": {
      "class": "eval",
      "from": ["output_log_prob0", "gather_ctc_prob"],
      "eval": f"{1 - ctc_scale} * source(0) + {ctc_scale} * tf.math.log(source(1))"
    }
  })


def get_segment_starts_and_lengths(segment_center_window_size: Optional[int]):
  if segment_center_window_size is None:
    return {
      "segment_lens": {
        "class": "combine",
        "from": ["segment_lens0", "const1"],
        "is_output_layer": True,
        "kind": "add",
      },
      "segment_lens0": {
        "class": "combine",
        "from": [":i", "segment_starts"],
        "kind": "sub",
      },
      "segment_starts": {
        "class": "switch",
        "condition": "prev:output_emit",
        "false_from": "prev:segment_starts",
        "initial_output": 0,
        "is_output_layer": True,
        "true_from": ":i",
      },
    }
  else:
    if segment_center_window_size % 2 == 1:
      window_half_size = (segment_center_window_size - 1) // 2
      return {
        "segment_ends": {
          "class": "switch",
          "condition": "seq_end_too_far",
          "false_from": "segment_ends1",
          "true_from": "seq_lens",
        },
        "segment_ends1": {
          "class": "eval",
          "eval": "source(0) + source(1) + %d" % window_half_size,
          "from": ["segment_starts1", "segment_lens1"],
        },
        "segment_lens": {
          "class": "eval",
          "eval": "source(0) - source(1) + 1",
          "from": ["segment_ends", "segment_starts"],
          "is_output_layer": True,
        },
        "segment_lens1": {
          "class": "combine",
          "from": [":i", "segment_starts1"],
          "kind": "sub",
        },
        "segment_starts": {
          "class": "switch",
          "condition": "seq_start_too_far",
          "false_from": "segment_starts2",
          "true_from": 0,
        },
        "segment_starts1": {
          "class": "switch",
          "condition": "prev:output_emit",
          "false_from": "prev:segment_starts1",
          "initial_output": 0,
          "is_output_layer": True,
          "true_from": ":i",
        },
        "segment_starts2": {
          "class": "eval",
          "eval": "source(0) + source(1) - %d" % window_half_size,
          "from": ["segment_starts1", "segment_lens1"],
          "is_output_layer": True,
        },
        "seq_end_too_far": {
          "class": "compare",
          "from": ["segment_ends1", "seq_lens"],
          "kind": "greater",
        },
        "seq_lens": {"class": "length", "from": "base:encoder"},
        "seq_start_too_far": {
          "class": "compare",
          "from": ["segment_starts2"],
          "kind": "less",
          "value": 0,
        },
      }
    else:
      window_half_size = segment_center_window_size // 2
      return {
        "segment_ends": {
          "class": "switch",
          "condition": "seq_end_too_far",
          "false_from": "segment_ends1",
          "true_from": "seq_lens",
        },
        "segment_ends1": {
          "class": "eval",
          "eval": "source(0) + source(1) + %d" % window_half_size,
          "from": ["segment_starts1", "segment_lens1"],
        },
        "segment_lens": {
          "class": "eval",
          "eval": "source(0) - source(1)",
          "from": ["segment_ends", "segment_starts"],
          "is_output_layer": True,
        },
        "segment_lens0": {
          "class": "combine",
          "from": [":i", "segment_starts1"],
          "kind": "sub",
        },
        "segment_lens1": {
          "class": "combine",
          "from": ["segment_lens0", "const1"],
          "is_output_layer": True,
          "kind": "add",
        },
        "segment_starts": {
          "class": "switch",
          "condition": "seq_start_too_far",
          "false_from": "segment_starts2",
          "true_from": 0,
        },
        "segment_starts1": {
          "class": "switch",
          "condition": "prev:output_emit",
          "false_from": "prev:segment_starts1",
          "initial_output": 0,
          "is_output_layer": True,
          "true_from": ":i",
        },
        "segment_starts2": {
          "class": "eval",
          "eval": "source(0) + source(1) - %d" % window_half_size,
          "from": ["segment_starts1", "segment_lens1"],
          "is_output_layer": True,
        },
        "seq_end_too_far": {
          "class": "compare",
          "from": ["segment_ends1", "seq_lens"],
          "kind": "greater",
        },
        "seq_lens": {"class": "length", "from": "base:encoder"},
        "seq_start_too_far": {
          "class": "compare",
          "from": ["segment_starts2"],
          "kind": "less",
          "value": 0,
        },
      }


def get_label_model_unit_dict(global_attention: bool, task: str):
  label_model_unit_dict = {
    "att": {"axes": "except_batch", "class": "merge_dims", "from": "att0"},
    "att_energy": {
      "activation": None,
      "class": "linear",
      "from": ["att_energy_tanh"],
      "n_out": 1,
      "with_bias": False,
    },
    "att_query": {
      "activation": None,
      "class": "linear",
      "from": "lm",
      # "is_output_layer": False,
      "n_out": 1024,
      "with_bias": False,
    },
    "att_energy_tanh": {
      "activation": "tanh",
      "class": "activation",
      "from": ["att_energy_in"],
    },
    "readout": {
      "class": "reduce_out",
      "from": ["readout_in"],
      "mode": "max",
      "num_pieces": 2,
    },
    "readout_in": {
      "activation": None,
      "class": "linear",
      "from": ["lm", "att"],
      "n_out": 1000,
    },
    "target_embed": {
      "activation": None,
      "class": "linear",
      "from": "output",
      "n_out": 621,
      "initial_output": 0,
      "with_bias": False,
    },
    "label_log_prob": {
      "activation": "log_softmax",
      "class": "linear",
      "dropout": 0.3,
      "from": "readout",
      "n_out": 1030,
    },
    "att_weights": {
      "class": "dropout",
      "dropout": 0.1,
      "dropout_noise_shape": {"*": None},
      "from": "att_weights0",
    },
  }

  if task == "train" or global_attention:
    label_model_unit_dict.update({
      "lm": {
        "class": "rec",
        "from": ["prev:att", "prev:target_embed"],
        "n_out": 1024,
        "unit": "nativelstm2",
      },
      "label_prob": {
        "activation": "exp",
        "class": "activation",
        "from": "label_log_prob",
        "is_output_layer": True,
        "loss": "ce",
        "loss_opts": {"focal_loss_factor": 0.0, "label_smoothing": 0.1},
        "target": "target_w_eos" if global_attention else "label_ground_truth",
      },
    })

  if global_attention:
    label_model_unit_dict.update({
      "att0": {
        "base": "base:enc_value",
        "class": "generic_attention",
        "weights": "att_weights",
      },
      "att_weights0": {
        "class": "softmax_over_spatial",
        "energy_factor": 0.03125,
        "from": ["att_energy"],
      },
      "end": {"class": "compare", "from": ["output"], "value": 0},
      "output": {
        "beam_size": 12,
        "class": "choice",
        "from": ["label_prob"],
        "initial_output": 0,
        "target": "target_w_eos" if task == "train" else "bpe",
      },
      "att_energy_in": {
        "class": "combine",
        "from": ["base:enc_ctx", "att_query"],
        "kind": "add",
        "n_out": 1024,
      },
    })
  else:
    label_model_unit_dict.update({
      "att0": {
        "base": "att_val",
        "class": "generic_attention",
        "weights": "att_weights",
      },
      # "att_val0": {"class": "copy", "from": "segments"},
      # "att_val_split": {
      #   "class": "reinterpret_data",
      #   "from": "att_val_split0",
      #   "set_dim_tags": {
      #     "dim:1": DimensionTag(
      #       kind=DimensionTag.Types.Spatial, description="att_heads", dimension=1
      #     )
      #   },
      # },
      "att_val": {
        "axis": "f",
        "class": "split_dims",
        "dims": (1, -1),
        "from": "segments",
      },
      "att_ctx": {
        "L2": None,
        "activation": None,
        "class": "linear",
        "dropout": 0.0,
        "from": "segments",
        "n_out": 1024,
        "with_bias": False,
        "name_scope": "/enc_ctx"
      },
      "att_weights0": {
        "axis": "stag:att_t",
        "class": "softmax_over_spatial",
        "energy_factor": 0.03125,
        "from": "att_energy",
      },
      "att_energy_in": {
        "class": "combine",
        "from": ["att_ctx", "att_query"],
        "kind": "add",
        "n_out": 1024,
      },
      "segments": {
        "class": "reinterpret_data",
        "from": "segments0",
        "set_dim_tags": {
          "stag:sliced-time:segments": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t")')
        },
      },
      "segments0": {
        "class": "slice_nd",
        "from": "base:encoder",
        "size": "segment_lens",
        "start": "segment_starts",
      },
    })

    if task == "train":
      label_model_unit_dict.update({
        "segment_lens": {
          "axis": "t",
          "class": "gather",
          "from": "base:data:segment_lens_masked",
          "position": ":i",
        },
        "segment_starts": {
          "axis": "t",
          "class": "gather",
          "from": "base:data:segment_starts_masked",
          "position": ":i",
        },
        "output": {
          "beam_size": 12,
          "class": "choice",
          "from": "label_prob",
          "initial_output": 0,
          "target": "label_ground_truth",
        },
      })
    else:
      label_model_unit_dict.update({
        "label_log_prob0": {
          "class": "combine",
          "from": ["label_log_prob", "emit_log_prob"],
          "kind": "add",
        },
        "lm_masked": {
          "class": "masked_computation",
          "from": "prev:target_embed",
          "mask": "prev:output_emit",
          "unit": {
            "class": "subnetwork",
            "from": "data",
            "subnetwork": {
              "lm": {
                "class": "rec",
                "from": ["base:prev:att", "data"],
                "name_scope": "/output/rec/lm",
                "n_out": 1024,
                "unit": "nativelstm2",
              },
              "output": {"class": "copy", "from": "lm"},
            },
          },
        },
        "output_log_prob": {
          "class": "copy",
          "from": ["label_log_prob0", "blank_log_prob"],
        },
        "output": {
          "beam_size": 12,
          "class": "choice",
          "from": "output_log_prob",
          "initial_output": 0,
          "input_type": "log_prob",
          "length_normalization": False,
          "target": "targets",
        },
      })

  return label_model_unit_dict


def get_ctc_loss(global_att: bool):
  return {
    "ctc": {
      "class": "copy",
      "from": "ctc_out_scores",
      "loss": "ctc",
      "loss_opts": {
        "beam_width": 1,
        "ctc_opts": {"logits_normalize": False},
        "output_in_log_space": True,
        "use_native": True,
      },
      "target": "targets" if global_att else "label_ground_truth",
    },
    "ctc_out": {
      "class": "softmax",
      "from": "encoder",
      "n_out": 1031,
      "with_bias": False,
    },
    "ctc_out_scores": {
      "class": "eval",
      "eval": "safe_log(source(0))",
      "from": ["ctc_out"],
    },
  }


def get_ctc_forced_align_hdf_dump(align_target: str, filename: str):
  return {
    "ctc_forced_align": {
      "align_target": align_target,
      "class": "forced_align",
      "from": "ctc",
      "input_type": "prob",
      "topology": "rna",
    },
    "ctc_forced_align_dump": {
      "class": "hdf_dump",
      "filename": filename,
      "from": "ctc_forced_align",
      "is_output_layer": True,
    },
  }


def get_info_layer(global_att: bool):
  if global_att:
    return {
      "#info": {"att_num_heads": 1, "enc_val_per_head": 2048}
    }
  else:
    return {
      "#info": {
        "l2": 0.0001,
        "learning_rate": 0.001,
        "lstm_dim": 1024,
        "time_red": [3, 2],
      }
    }


def get_enc_ctx_and_val():
  return {
    "enc_value": {
      "axis": "F",
      "class": "split_dims",
      "dims": (1, -1),
      "from": ["encoder"],
    },
    "enc_ctx": {
      "activation": None,
      "class": "linear",
      "from": ["encoder"],
      "n_out": 1024,
      "with_bias": False,
    },
  }


def get_target_w_eos():
  return {
    "target_with_eos": {
      "class": "postfix_in_time",
      "from": "data:targets",
      "postfix": 0,
      "register_as_extern_data": "target_w_eos",
      "repeat": 1,
    }
  }


def get_existing_alignment_layer():
  return {
    "existing_alignment": {
      "class": "reinterpret_data",
      "from": "data:targets",
      "set_sparse": True,
      "set_sparse_dim": 1031,
      "size_base": "encoder",
    },
  }


def get_is_label_layer():
  return {
    "is_label": {
      "class": "compare",
      "from": "existing_alignment",
      "kind": "not_equal",
      "value": 1030,
    },
  }


def get_label_ground_truth_target():
  return {
    "label_ground_truth_masked": {
      "class": "reinterpret_data",
      "enforce_batch_major": True,
      "from": "label_ground_truth_masked0",
      "register_as_extern_data": "label_ground_truth",
      "set_sparse_dim": 1030,
    },
    "label_ground_truth_masked0": {
      "class": "masked_computation",
      "from": "existing_alignment",
      "mask": "is_label",
      "unit": {"class": "copy", "from": "data"},
    },
  }


def get_emit_ground_truth_target():
  return {
    "const0": {"class": "constant", "value": 0, "with_batch_dim": True},
    "const1": {"class": "constant", "value": 1, "with_batch_dim": True},
    "emit_ground_truth": {
      "class": "reinterpret_data",
      "from": "emit_ground_truth0",
      "is_output_layer": True,
      "register_as_extern_data": "emit_ground_truth",
      "set_sparse": True,
      "set_sparse_dim": 2,
    },
    "emit_ground_truth0": {
      "class": "switch",
      "condition": "is_label",
      "false_from": "const0",
      "true_from": "const1",
    },
  }


def get_targetb_target():
  return {
    "labels_with_blank_ground_truth": {
      "class": "copy",
      "from": "existing_alignment",
      "register_as_extern_data": "targetb",
    },
  }


def get_masked_segment_starts_and_lengths():
  return {
    "segment_lens_masked": {
      "class": "masked_computation",
      "from": "output/segment_lens",
      "mask": "is_label",
      "out_spatial_dim": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="label-axis")'),
      "register_as_extern_data": "segment_lens_masked",
      "unit": {"class": "copy", "from": "data"},
    },
    "segment_starts_masked": {
      "class": "masked_computation",
      "from": "output/segment_starts",
      "mask": "is_label",
      "out_spatial_dim": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="label-axis")'),
      "register_as_extern_data": "segment_starts_masked",
      "unit": {"class": "copy", "from": "data"},
    },
  }


def get_decision_layer(global_att: bool):
  if global_att:
    return {
      "decision": {
        "class": "decide",
        "from": "output",
        "loss": "edit_distance",
      }
    }
  else:
    return {
      "output_non_blank": {
        "class": "compare",
        "from": "output",
        "kind": "not_equal",
        "value": 1030,
      },
      "output_wo_b": {
        "class": "reinterpret_data",
        "from": "output_wo_b0",
        "set_sparse_dim": 1030,
      },
      "output_wo_b0": {
        "class": "masked_computation",
        "from": "output",
        "mask": "output_non_blank",
        "unit": {"class": "copy"},
      },
      "decision": {
        "class": "decide",
        "from": "output_wo_b",
        "loss": "edit_distance",
        "target": "targets",
      },
    }


def get_blstm_encoder():
  lstm_dim = 1024
  time_reduction = [3, 2]
  src_layer = "conv_merged"
  num_lstm_layers = 6
  l2 = 0.0001

  encoder_net_dict = {}
  encoder_net_dict.update({
    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv",
      "from": "source0",
      "padding": "same",
      "filter_size": (3, 3),
      "n_out": 32,
      "activation": None,
      "with_bias": True,
      "auto_use_channel_first": False
    },  # (T,40,32)
    "conv0p": {
      "class": "pool",
      "mode": "max",
      "padding": "same",
      "pool_size": (1, 2),
      "from": "conv0",
      "use_channel_first": False
    },  # (T,20,32)
    "conv1": {
      "class": "conv",
      "from": "conv0p",
      "padding": "same",
      "filter_size": (3, 3),
      "n_out": 32,
      "activation": None,
      "with_bias": True,
      "auto_use_channel_first": False
    },  # (T,20,32)
    "conv1p": {
      "class": "pool",
      "mode": "max",
      "padding": "same",
      "pool_size": (1, 2),
      "from": "conv1",
      "use_channel_first": False
    },  # (T,10,32)
    "conv_merged": {
      "class": "merge_dims",
      "from": "conv1p",
      "axes": "static"
    },  # (T,320)
    "encoder": {"class": "copy", "from": "encoder0"},
  })
  # Add encoder BLSTM stack.

  if num_lstm_layers >= 1:
    encoder_net_dict.update({
      "lstm0_fw": {
        "class": "rec",
        "unit": "nativelstm2",
        "n_out": lstm_dim,
        "L2": l2,
        "direction": 1,
        "from": src_layer,
        # "trainable": True
      },
      "lstm0_bw": {
        "class": "rec",
        "unit": "nativelstm2",
        "n_out": lstm_dim,
        "L2": l2,
        "direction": -1,
        "from": src_layer,
        # "trainable": True
      }
    })
    src_layer = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    encoder_net_dict.update({
      "lstm%i_pool" % (i - 1): {
        "class": "pool",
        "mode": "max",
        "padding": "same",
        "pool_size": (red,),
        "from": src_layer
      }
    })
    src_layer = "lstm%i_pool" % (i - 1)
    encoder_net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec",
        "unit": "nativelstm2",
        "n_out": lstm_dim,
        "L2": l2,
        "direction": 1,
        "from": src_layer,
        "dropout": 0.3,
        # "trainable": True
      },
      "lstm%i_bw" % i: {
        "class": "rec",
        "unit": "nativelstm2",
        "n_out": lstm_dim,
        "L2": l2,
        "direction": -1,
        "from": src_layer,
        "dropout": 0.3,
        # "trainable": True
      }
    })
    src_layer = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  encoder_net_dict["encoder0"] = {"class": "copy", "from": src_layer}

  return encoder_net_dict
