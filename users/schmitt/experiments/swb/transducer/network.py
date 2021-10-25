

def get_alignment_net_dict(pretrain_idx):
  """
  :param int|None pretrain_idx: starts at 0. note that this has a default repetition factor of 6
  :return: net_dict or None if pretrain should stop
  :rtype: dict[str,dict[str]|int]|None
  """
  import numpy
  # Note: epoch0 is 0-based here! I.e. in contrast to elsewhere, where it is 1-based.
  # Also, we never use #repetition here, such that this is correct.
  # This is important because of sub-epochs and storing the HDF files,
  # to know exactly which HDF files cover the dataset completely.
  epoch0 = pretrain_idx
  net_dict = {}

  # network
  # (also defined by num_inputs & num_outputs)
  # EncKeyTotalDim = 200
  # AttNumHeads = 1  # must be 1 for hard-att
  # AttentionDropout = 0.1
  # l2 = 0.0001
  # EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
  # EncValueTotalDim = 2048
  # EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
  # LstmDim = EncValueTotalDim // 2

  if pretrain_idx is not None:
    net_dict["#config"] = {}

    # Do this in the very beginning.
    # lr_warmup = [0.0] * eval("epoch_split")  # first collect alignments with existing model, no training
    lr_warmup = list(numpy.linspace(eval("learning_rate") * 0.1, eval("learning_rate"), num=10))
    if pretrain_idx < len(lr_warmup):
      net_dict["#config"]["learning_rate"] = lr_warmup[pretrain_idx]

  # We import the model, thus no growing.
  start_num_lstm_layers = 2
  final_num_lstm_layers = 6
  num_lstm_layers = final_num_lstm_layers
  if pretrain_idx is not None:
    pretrain_idx = max(pretrain_idx, 0) // 5  # Repeat a bit.
    num_lstm_layers = pretrain_idx + start_num_lstm_layers
    pretrain_idx = num_lstm_layers - final_num_lstm_layers
    num_lstm_layers = min(num_lstm_layers, final_num_lstm_layers)

  if final_num_lstm_layers > start_num_lstm_layers:
    start_dim_factor = 0.5
    grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
    dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
  else:
    dim_frac = 1.

  time_reduction = [3, 2] if num_lstm_layers >= 3 else [6]

  if pretrain_idx is not None and pretrain_idx <= 1 and "learning_rate" not in net_dict["#config"]:
    # Fixed learning rate for the beginning.
    net_dict["#config"]["learning_rate"] = eval("learning_rate")

  net_dict["#info"] = {
    "epoch0": epoch0,  # Set this here such that a new construction for every pretrain idx is enforced in all cases.
    "num_lstm_layers": num_lstm_layers, "dim_frac": dim_frac, }

  # We use this pretrain construction during the whole training time (epoch0 > num_epochs).
  if pretrain_idx is not None and epoch0 % eval("epoch_split") == 0 and epoch0 > eval("num_epochs"):
    # Stop pretraining now.
    return None

  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "trainable": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True, "trainable": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    # Encoder LSTMs added below, resulting in "encoder0".

    "encoder": {"class": "copy", "from": "encoder0"},
    # "enc_ctx0": {
    #   "class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim, "L2": l2,
    #   "dropout": 0.2},
    # "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": 5},  # [B,T,W,D]
    # "enc_val": {"class": "copy", "from": "encoder"},
    # "enc_val_win": {"class": "window", "from": "enc_val", "window_size": 5},  # [B,T,W,D]

    # "enc_ctx": {
    #   "class": "linear", "from": "encoder", "activation": "tanh", "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},

    "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},

    # for task "search" / search_output_layer
    "output_wo_b0": {
      "class": "masked_computation", "unit": {"class": "copy"}, "from": "output", "mask": "output/output_emit"},
    "output_wo_b": {"class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": eval("target_num_labels")},
    "decision": {
      "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": eval("target"), 'only_on_search':
      True}, })

  # Add encoder BLSTM stack.
  src = "conv_merged"
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "trainable": True},
      "lstm0_bw": {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "trainable": True}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "dropout": 0.3 * dim_frac, "trainable": True}, "lstm%i_bw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

  # This is used for training.
  net_dict["lm_input0"] = {"class": "copy", "from": "data:%s" % eval("target")}
  net_dict["lm_input1"] = {"class": "prefix_in_time", "from": "lm_input0", "prefix": eval("targetb_blank_idx")}
  net_dict["lm_input"] = {"class": "copy", "from": "lm_input1"}

  def get_output_dict(train, search, targetb, beam_size=eval("beam_size")):
    return {
      "class": "rec", "from": "encoder",  # time-sync
      "include_eos": True, "back_prop": (eval("task") == "train") and train, "unit": {
        "am": {"class": "copy", "from": "data:source"},

        # "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
        # "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
        # "att_query": {
        #   "class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
        # 'att_energy': {
        #   "class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
        #   "from": ['enc_ctx_win', 'att_query']},  # (B, W)
        # 'att_weights0': {
        #   "class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
        #   "energy_factor": EncKeyPerHeadDim ** -0.5},  # (B, W)
        # 'att_weights1': {
        #   "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0',
        #   "dropout": AttentionDropout},
        # "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
        # 'att': {
        #   "class": "dot", "from": ['att_weights', 'enc_val_win'], "red1": "static:0", "red2": "static:0",
        #   "var1": None, "var2": "f"},  # (B, V)

        "prev_out_non_blank": {
          "class": "reinterpret_data", "from": "prev:output", "set_sparse_dim": eval("target_num_labels"),
          "set_sparse": True},
        "lm_masked": {
          "class": "masked_computation", "mask": "prev:output_emit", "from": "prev_out_non_blank",  # in decoding
          "masked_from": "base:lm_input" if eval("task") == "train" else None,  # enables optimization if used

          "unit": {
            "class": "subnetwork", "from": "data", "trainable": True, "subnetwork": {
              "input_embed": {
                "class": "linear", "n_out": 256, "activation": "identity", "trainable": True, "L2": l2,
                "from": "data"}, "lstm0": {
                "class": "rec", "unit": "nativelstm2", "dropout": 0.2, "n_out": 1024, "L2": l2, "from": "input_embed",
                "trainable": True}, "output": {"class": "copy", "from": "lstm0"}
            }}},
        "lm": {"class": "copy", "from": "lm_embed_unmask"},  # [B,L]

        # joint network: (W_enc h_{enc,t} + W_pred * h_{pred,u} + b)
        # train : (T-enc, B, F|2048) ; (U+1, B, F|256)
        # search: (B, F|2048) ; (B, F|256)
        "readout_in": {
          "class": "linear", "from": ["am", "att", "lm_masked"], "activation": None, "n_out": 1000, "L2": l2,
          "dropout": 0.2, "out_type": {
            "batch_dim_axis": 2 if eval("task") == "train" else 0,
            "shape": (None, None, 1000) if eval("task") == "train" else (1000,),
            "time_dim_axis": 0 if eval("task") == "train" else None}},  # (T, U+1, B, 1000)

        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

        "label_log_prob": {
          "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3,
          "n_out": eval("target_num_labels")},  # (B, T, U+1, 1030)
        "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1, "is_output_layer": True},
        # (B, T, U+1, 1)
        "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
        "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.math.log_sigmoid(-source(0))"},
        # (B, T, U+1, 1)
        "label_emit_log_prob": {"class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},
        # (B, T, U+1, 1), scaling factor in log-space
        "output_log_prob": {"class": "copy", "from": ["blank_log_prob", "label_emit_log_prob"]},  # (B, T, U+1, 1031)

        "output_prob": {
          "class": "eval", "from": ["output_log_prob", "base:data:" + eval("target"), "base:encoder"],
          "eval": eval("rna_loss"), "out_type": eval("rna_loss_out"), "loss": "as_is", },

        # this only works when the loop has been optimized, i.e. log-probs are (B, T, U, V)
        "rna_alignment": {
          "class": "eval", "from": ["output_log_prob", "base:data:" + eval("target"), "base:encoder"],
          "eval": eval("rna_alignment"),
          "out_type": eval("rna_alignment_out"), "is_output_layer": True} if eval("task") == "train"  # (B, T)
        else {"class": "copy", "from": "output_log_prob"},

        # During training   : targetb = "target"  (RNA-loss)
        # During recognition: targetb = "targetb"
        'output': {
          'class': 'choice', 'target': targetb, 'beam_size': beam_size, 'from': "output_log_prob",
          "input_type": "log_prob", "initial_output": 0, "cheating": "exclusive" if eval("task") == "train" else None,
          "explicit_search_sources": ["prev:out_str", "prev:output"] if eval("task") == "search" else None,
          "custom_score_combine": eval("targetb_recomb_recog") if eval("task") == "search" else None},

        "out_str": {
          "class": "eval", "from": ["prev:out_str", "output_emit", "output"], "initial_output": None,
          "out_type": {"shape": (), "dtype": "string"}, "eval": eval("out_str")},

        "output_is_not_blank": {
          "class": "compare", "from": "output", "value": eval("targetb_blank_idx"), "kind": "not_equal",
          "initial_output": True},

        # initial state=True so that we are consistent to the training and the initial state is correctly set.
        "output_emit": {
          "class": "copy", "from": "output_is_not_blank", "initial_output": True, "is_output_layer": True},

        "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt"]},
        "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt"]},

      },
    }

  if eval("task") == "train":
    net_dict["output"] = eval("get_output_dict(train=True, search=False, targetb=target)")
  else:
    net_dict["output"] = eval("get_output_dict(train=True, search=True, targetb='targetb')")

  if eval("task") in ("train", "forward"):
    net_dict["rna_alignment"] = {"class": "copy", "from": ["output/rna_alignment"]}  # (B, T)

  return net_dict

def get_extended_net_dict(pretrain_idx):
  """
  :param int|None pretrain_idx: starts at 0. note that this has a default repetition factor of 6
  :return: net_dict or None if pretrain should stop
  :rtype: dict[str,dict[str]|int]|None
  """
  # Note: epoch0 is 0-based here! I.e. in contrast to elsewhere, where it is 1-based.
  # Also, we never use #repetition here, such that this is correct.
  # This is important because of sub-epochs and storing the HDF files,
  # to know exactly which HDF files cover the dataset completely.
  epoch0 = pretrain_idx
  net_dict = {}

  # network
  # (also defined by num_inputs & num_outputs)
  # EncKeyTotalDim = 200
  # AttNumHeads = 1  # must be 1 for hard-att
  # AttentionDropout = 0.1
  # EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
  # EncValueTotalDim = 2048
  # EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
  # LstmDim = EncValueTotalDim // 2
  # l2 = 0.0001

  have_existing_align = True  # only in training, and only in pretrain, and only after the first epoch
  if pretrain_idx is not None:
    net_dict["#config"] = {}

    # Do this in the very beginning.
    # lr_warmup = [0.0] * eval("epoch_split")  # first collect alignments with existing model, no training
    lr_warmup = list(numpy.linspace(learning_rate * 0.1, learning_rate, num=10))
    # lr_warmup += [learning_rate] * 20
    if pretrain_idx < len(lr_warmup):
      net_dict["#config"]["learning_rate"] = lr_warmup[
        pretrain_idx]  # if pretrain_idx >= eval("epoch_split") + eval("epoch_split") // 2:  #    net_dict["#config"][
      # "param_variational_noise"]
      # = 0.1  # pretrain_idx -= len(lr_warmup)

  use_targetb_search_as_target = False  # not have_existing_align or epoch0 < StoreAlignmentUpToEpoch
  keep_linear_align = False  # epoch0 is not None and epoch0 < eval("epoch_split") * 2

  # We import the model, thus no growing.
  start_num_lstm_layers = 2
  final_num_lstm_layers = 6
  num_lstm_layers = final_num_lstm_layers
  if pretrain_idx is not None:
    pretrain_idx = max(pretrain_idx, 0) // 6  # Repeat a bit.
    num_lstm_layers = pretrain_idx + start_num_lstm_layers
    pretrain_idx = num_lstm_layers - final_num_lstm_layers
    num_lstm_layers = min(num_lstm_layers, final_num_lstm_layers)

  if final_num_lstm_layers > start_num_lstm_layers:
    start_dim_factor = 0.5
    grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
    dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
  else:
    dim_frac = 1.

  time_reduction = [3, 2] if num_lstm_layers >= 3 else [6]

  if pretrain_idx is not None and pretrain_idx <= 1 and "learning_rate" not in net_dict["#config"]:
    # Fixed learning rate for the beginning.
    net_dict["#config"]["learning_rate"] = learning_rate

  net_dict["#info"] = {
    "epoch0": epoch0,  # Set this here such that a new construction for every pretrain idx is enforced in all cases.
    "num_lstm_layers": num_lstm_layers, "dim_frac": dim_frac, "have_existing_align": have_existing_align,
    "use_targetb_search_as_target": use_targetb_search_as_target, "keep_linear_align": keep_linear_align, }

  # We use this pretrain construction during the whole training time (epoch0 > num_epochs).
  if pretrain_idx is not None and epoch0 % eval("epoch_split") == 0 and epoch0 > num_epochs:
    # Stop pretraining now.
    return None

  net_dict.update({
    "source": {
      "class": "eval",
      "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {
      "class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {
      "class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None,
      "with_bias": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    # Encoder LSTMs added below, resulting in "encoder0".

    # "encoder": {"class": "postfix_in_time", "postfix": 0.0, "from": "encoder0"},
    "encoder": {"class": "linear", "from": "encoder0", "n_out": 256, "activation": None},
    # "enc_ctx0": {
    #   "class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
    # "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": 5},  # [B,T,W,D]
    # "enc_val": {"class": "copy", "from": "encoder"},
    # "enc_val_win": {"class": "window", "from": "enc_val", "window_size": 5},  # [B,T,W,D]

    "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},

    # for task "search" / search_output_layer
    "output_wo_b0": {
      "class": "masked_computation", "unit": {"class": "copy"}, "from": "output", "mask": "output/output_emit"},
    "output_wo_b": {"class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": target_num_labels},
    "decision": {
      "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target, 'only_on_search': True},

    "targetb_linear": {
      "class": "eval", "from": ["data:%s" % target, "encoder"], "eval": targetb_linear,
      "out_type": targetb_linear_out},

    # Target for decoder ('output') with search ("extra.search") in training.
    # The layer name must be smaller than "t_target" such that this is created first.
    "1_targetb_base": {
      "class": "copy", "from": "existing_alignment",  # if have_existing_align else "targetb_linear",
      "register_as_extern_data": "targetb_base" if task == "train" else None},

    "2_targetb_target": {
      "class": "eval", "from": "targetb_search_or_fallback" if use_targetb_search_as_target else "data:targetb_base",
      "eval": "source(0)", "register_as_extern_data": "targetb" if task == "train" else None},

    "ctc_out": {"class": "softmax", "from": "encoder", "with_bias": False, "n_out": targetb_num_labels},
    "ctc_out_scores": {
      "class": "eval", "from": ["ctc_out"], "eval": "safe_log(source(0))", },

    "_target_masked": {
      "class": "masked_computation", "mask": "output/output_emit", "from": "output", "unit": {"class": "copy"}},
    "3_target_masked": {
      "class": "reinterpret_data", "from": "_target_masked", "set_sparse_dim": target_num_labels,
      # we masked blank away
      "enforce_batch_major": True,  # ctc not implemented otherwise...
      "register_as_extern_data": "targetb_masked" if task == "train" else None},

    "ctc": {
      "class": "copy", "from": "ctc_out_scores", "loss": "ctc" if task == "train" else None,
      "target": "targetb_masked" if task == "train" else None, "loss_opts": {
        "beam_width": 1, "use_native": True, "output_in_log_space": True,
        "ctc_opts": {"logits_normalize": False}} if task == "train" else None}, })

  if have_existing_align:
    net_dict.update({
      # This should be compatible to t_linear or t_search.
      "existing_alignment": {
        "class": "reinterpret_data", "from": "data:alignment", "set_sparse": True,  # not sure what the HDF gives us
        "set_sparse_dim": targetb_num_labels, "size_base": "encoder",  # for RNA...
      }, })

  # Add encoder BLSTM stack.
  src = "conv_merged"
  if num_lstm_layers >= 1:
    net_dict.update({
      "lstm0_fw": {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "trainable": True}, "lstm0_bw": {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "trainable": True}})
    src = ["lstm0_fw", "lstm0_bw"]
  for i in range(1, num_lstm_layers):
    red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
    net_dict.update({
      "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
    src = "lstm%i_pool" % (i - 1)
    net_dict.update({
      "lstm%i_fw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1,
        "from": src, "dropout": 0.3 * dim_frac, "trainable": True}, "lstm%i_bw" % i: {
        "class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1,
        "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
    src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
  net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

  def get_output_dict(train, search, targetb, beam_size=beam_size):
    return {
      "class": "rec", "from": "encoder", "include_eos": True, "back_prop": (task == "train") and train, "unit": {
        # "am": {"class": "gather_nd", "from": "base:encoder", "position": "prev:t"},  # [B,D]
        "am": {"class": "copy", "from": "data:source"}, # could make more efficient...
        # "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
        # "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
        # "att_query": {
        #   "class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
        # 'att_energy': {
        #   "class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
        #   "from": ['enc_ctx_win', 'att_query']},  # (B, W)
        # 'att_weights0': {
        #   "class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
        #   "energy_factor": EncKeyPerHeadDim ** -0.5},  # (B, W)
        # 'att_weights1': {
        #   "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0',
        #   "dropout": AttentionDropout},
        # "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
        # 'att': {
        #   "class": "dot", "from": ['att_weights', 'enc_val_win'], "red1": "static:0", "red2": "static:0",
        #   "var1": None, "var2": "f"},  # (B, V)

        "prev_out_non_blank": {
          "class": "reinterpret_data", "from": "prev:output_", "set_sparse_dim": target_num_labels,
          "set_sparse": True}, "lm_masked": {
          "class": "masked_computation", "mask": "prev:output_emit", "from": "prev_out_non_blank",  # in decoding

          "unit": {
            "class": "subnetwork", "from": "data", "subnetwork": {
              "input_embed": {
                "class": "linear", "activation": None, "with_bias": False, "from": "data", "n_out": 621},
              "lstm0": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "from": ["input_embed", "base:att"]},
              "output": {"class": "copy", "from": "lstm0"}}}},
        "lm_embed_masked": {"class": "copy", "from": "lm_masked"},
        "lm_embed_unmask": {"class": "unmask", "from": "lm_embed_masked", "mask": "prev:output_emit"},
        "lm": {"class": "copy", "from": "lm_embed_unmask"},  # [B,L]

        "prev_label_masked": {
          "class": "masked_computation", "mask": "prev:output_emit", "from": "prev_out_non_blank",  # in decoding
          "unit": {"class": "linear", "activation": None, "n_out": 256}},
        "prev_label_unmask": {"class": "unmask", "from": "prev_label_masked", "mask": "prev:output_emit"},

        "prev_out_embed": {"class": "linear", "from": "prev:output_", "activation": None, "n_out": 128}, "s": {
          "class": "rec", "unit": "nativelstm2", "from": ["am", "prev_out_embed", "lm"], "n_out": 128, "L2": l2,
          "dropout": 0.3, "unit_opts": {"rec_weight_dropout": 0.3}},

        "readout_in": {"class": "linear", "from": ["s", "att", "lm"], "activation": None, "n_out": 1000},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

        "label_log_prob": {
          "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3,
          "n_out": target_num_labels}, "label_prob": {
          "class": "activation", "from": "label_log_prob", "activation": "exp"},
        "emit_prob0": {"class": "linear", "from": "s", "activation": None, "n_out": 1, "is_output_layer": True},
        "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},
        "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.math.log_sigmoid(-source(0))"},
        "label_emit_log_prob": {"class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},
        # 1 gets broadcasted
        "output_log_prob": {"class": "copy", "from": ["label_emit_log_prob", "blank_log_prob"]}, "output_prob": {
          "class": "activation", "from": "output_log_prob", "activation": "exp", "target": targetb, "loss": "ce",
          "loss_opts": {"focal_loss_factor": 2.0}},

        # "output_ce": {
        #    "class": "loss", "from": "output_prob", "target_": "layer:output", "loss_": "ce", "loss_opts_": {"label_smoothing": 0.1},
        #    "loss": "as_is" if train else None, "loss_scale": 0 if train else None},
        # "output_err": {"class": "copy", "from": "output_ce/error", "loss": "as_is" if train else None, "loss_scale": 0 if train else None},
        # "output_ce_blank": {"class": "eval", "from": "output_ce", "eval": "source(0) * 0.03"},  # non-blank/blank factor
        # "loss": {"class": "switch", "condition": "output_is_blank", "true_from": "output_ce_blank", "false_from": "output_ce", "loss": "as_is" if train else None},

        'output': {
          'class': 'choice', 'target': targetb, 'beam_size': beam_size, 'from': "output_log_prob",
          "input_type": "log_prob", "initial_output": 0, "cheating": "exclusive" if task == "train" else None,
          # "explicit_search_sources": ["prev:u"] if task == "train" else None,
          # "custom_score_combine": targetb_recomb_train if task == "train" else None
          "explicit_search_sources": ["prev:out_str", "prev:output"] if task == "search" else None,
          "custom_score_combine": targetb_recomb_recog if task == "search" else None}, "output_": {
          "class": "eval", "from": "output", "eval": switchout_target,
          "initial_output": 0, } if task == "train" else {"class": "copy", "from": "output", "initial_output": 0},

        "out_str": {
          "class": "eval", "from": ["prev:out_str", "output_emit", "output"], "initial_output": None,
          "out_type": {"shape": (), "dtype": "string"}, "eval": out_str},

        "output_is_not_blank": {
          "class": "compare", "from": "output_", "value": targetb_blank_idx, "kind": "not_equal",
          "initial_output": True},

        # This "output_emit" is True on the first label but False otherwise, and False on blank.
        "output_emit": {
          "class": "copy", "from": "output_is_not_blank", "initial_output": True, "is_output_layer": True},

        "const0": {"class": "constant", "value": 0, "collocate_with": "du"},
        "const1": {"class": "constant", "value": 1, "collocate_with": "du"},

        # pos in target, [B]
        "du": {"class": "switch", "condition": "output_emit", "true_from": "const1", "false_from": "const0"},
        "u": {"class": "combine", "from": ["prev:u", "du"], "kind": "add", "initial_output": 0},

        # "end": {"class": "compare", "from": ["t", "base:enc_seq_len"], "kind": "greater_equal"},

      }, "target": targetb, "size_target": targetb if task == "train" else None,
      "max_seq_len": "max_len_from('base:encoder') * 2"}

  net_dict["output"] = get_output_dict(train=True, search=(task != "train"), targetb="targetb")

  return net_dict


def custom_construction_algo(idx, net_dict):
  # For debugging, use: python3 ./crnn/Pretrain.py config...
  net_dict = get_net_dict(pretrain_idx=idx)
  if net_dict is not None:
    add_attention(net_dict, _attention_type)
  return net_dict
