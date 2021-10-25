from enum import Enum


class AttentionTypes(Enum):
  """The order matters as the functions below use indices for the attention types"""
  NO = 0
  LOCAL_WIN_SIZE_5 = 1
  LOCAL_WIN_SIZE_10 = 2
  SEG = 3
  SEG_PLUS_WIN = 4
  SEG_PLUS_RIGHT_WIN = 5
  SEG_PLUS_LEFT_WIN = 6
  SEG_EXCEPT_FIRST = 7
  SEG_ONLY_NON_BLANK = 8


def add_attention(net_dict, attention_type):
  """

  :param net_dict:
  :param attention_type: int in [0, 1, 2]
  :return: dict net_dict with added attention mechanism
  """

  if attention_type == 0:
    net_dict["output"]["unit"]["att"] = {"class": "copy", "from": "am"}
  if attention_type in (1, 2):
    if attention_type == 1:
      win_size = 5
    else:
      win_size = 10
    net_dict.update({
      "enc_ctx0": {
        "class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim"),
        "L2": eval("l2"),
        "dropout": 0.2},
      "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": win_size},  # [B,T,W,D]
      "enc_val": {"class": "copy", "from": "encoder"},
      "enc_val_win": {"class": "window", "from": "enc_val", "window_size": win_size},  # [B,T,W,D]
    })
    net_dict["output"]["unit"].update({
      "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
      "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
      "att_query": {
        "class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
      'att_energy': {
        "class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
        "from": ['enc_ctx_win', 'att_query']},  # (B, W)
      'att_weights0': {
        "class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
        "energy_factor": eval("EncKeyPerHeadDim") ** -0.5},  # (B, W)
      'att_weights1': {
        "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0',
        "dropout": eval("AttentionDropout")},
      "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
      'att': {
        "class": "dot", "from": ['att_weights', 'enc_val_win'], "red1": "static:0", "red2": "static:0",
        "var1": None, "var2": "f"},  # (B, V)
    })
  if attention_type == 3:
    net_dict["output"]["unit"].update({
      "const1": {"class": "constant", "value": 1, "is_output_layer": True},
      "segment_starts": {  # (B,)
        "class": "switch", "condition": "prev:output_is_not_blank", "true_from": ":i",
        "false_from": "prev:segment_starts", "initial_output": 0},
      "segment_lens0": {"class": "combine", "kind": "sub", "from": [":i", "segment_starts"]},
      "segment_lens": {"class": "combine", "kind": "add", "from": ["segment_lens0", "const1"]},  # (B,)

      "segments": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},

      "att_query": {"class": "linear", "from": "am", "activation": None, "with_bias": False,
                    "n_out": eval("EncKeyTotalDim")},
      # [B, D]

      "enc_ctx_seg0": {"class": "copy", "from": "segments"}, "enc_ctx_seg": {  # [B,D]
        "class": "linear", "from": "enc_ctx_seg0", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
      "enc_val_seg": {"class": "copy", "from": "segments"},

      'att_energy': {  # [B,t_sliced,1]
        "class": "dot", "red1": "f", "red2": "f", "var1": "dyn:-1", "var2": None, "from": ['enc_ctx_seg', 'att_query']},
      'att_weights0': {  # [B,t_sliced,1]
        "class": "softmax_over_spatial", "axis": "dyn:-1", "from": 'att_energy',
        "energy_factor": eval("EncKeyPerHeadDim") ** -0.5},
      'att_weights': {
        "class": "dropout", "dropout_noise_shape": {"*": None},  # [B,t_sliced,1]
        "from": 'att_weights0', "dropout": eval("AttentionDropout")},
      'att0': {
        "class": "dot", "from": ['att_weights', 'enc_val_seg'], "red1": "dyn:-1", "red2": "dyn:-1", "var1": "f",
        "var2": "f"},  # (B, 1, V)
      "att": {"class": "merge_dims", "from": "att0", "axes": ["static:0", "static:1"]},  # [B,V]
    })
  if attention_type in (4, 5, 6):
    """
    segmental attention + fixed local window
    """
    win_size = 5
    net_dict["output"]["unit"].update({
      "const1": {"class": "constant", "value": 1, "is_output_layer": True},
      "const_win_size": {
        "class": "constant",
        "value": win_size},

      "segments": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},

      "att_query": {"class": "linear", "from": "am", "activation": None, "with_bias": False,
                    "n_out": eval("EncKeyTotalDim")},
      # [B, D]

      "enc_ctx_seg0": {"class": "copy", "from": "segments"}, "enc_ctx_seg": {  # [B,D]
        "class": "linear", "from": "enc_ctx_seg0", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
      "enc_val_seg": {"class": "copy", "from": "segments"},

      'att_energy': {  # [B,t_sliced,1]
        "class": "dot", "red1": "f", "red2": "f", "var1": "dyn:-1", "var2": None, "from": ['enc_ctx_seg', 'att_query']},
      'att_weights0': {  # [B,t_sliced,1]
        "class": "softmax_over_spatial", "axis": "dyn:-1", "from": 'att_energy',
        "energy_factor": eval("EncKeyPerHeadDim") ** -0.5},
      'att_weights': {
        "class": "dropout", "dropout_noise_shape": {"*": None},  # [B,t_sliced,1]
        "from": 'att_weights0', "dropout": eval("AttentionDropout")},
      'att0': {
        "class": "dot", "from": ['att_weights', 'enc_val_seg'], "red1": "dyn:-1", "red2": "dyn:-1", "var1": "f",
        "var2": "f"},  # (B, 1, V)
      "att": {"class": "merge_dims", "from": "att0", "axes": ["static:0", "static:1"]},  # [B,V]
    })

    if attention_type == 4:
      net_dict["output"]["unit"].update({
        "segment_starts0": {  # (B,)
          "class": "switch",
          "condition": "prev:output_is_not_blank",
          "true_from": ":i",
          "false_from": "prev:segment_starts0",
          "initial_output": 0},
        "segment_starts": {
          "class": "combine",
          "from": ["segment_starts0", "const_win_size"],
          "kind": "sub"},
        "segment_lens0": {
          "class": "combine",
          "kind": "sub",
          "from": [":i", "segment_starts"]},
        "segment_lens1": {
          "class": "combine",
          "kind": "add",
          "from": ["segment_lens0", "const1"]},  # (B,)
        "segment_lens": {
          "class": "combine",
          "kind": "add",
          "from": ["segment_lens1", "const_win_size"]},
      })
    elif attention_type == 5:
      net_dict["output"]["unit"].update({
        "segment_starts": {  # (B,)
          "class": "switch",
          "condition": "prev:output_is_not_blank",
          "true_from": ":i",
          "false_from": "prev:segment_starts",
          "initial_output": 0},
        "segment_lens0": {
          "class": "combine",
          "kind": "sub",
          "from": [":i", "segment_starts"]},
        "segment_lens1": {
          "class": "combine",
          "kind": "add",
          "from": ["segment_lens0", "const1"]},  # (B,)
        "segment_lens": {
          "class": "combine",
          "kind": "add",
          "from": ["segment_lens1", "const_win_size"]},
      })
    elif attention_type == 6:
      net_dict["output"]["unit"].update({
        "segment_starts0": {  # (B,)
          "class": "switch",
          "condition": "prev:output_is_not_blank",
          "true_from": ":i",
          "false_from": "prev:segment_starts0",
          "initial_output": 0},
        "segment_starts": {
          "class": "combine",
          "from": ["segment_starts0", "const_win_size"],
          "kind": "sub"},
        "segment_lens0": {
          "class": "combine",
          "kind": "sub",
          "from": [":i", "segment_starts"]},
        "segment_lens": {
          "class": "combine",
          "kind": "add",
          "from": ["segment_lens0", "const1"]},  # (B,)
      })
  if attention_type == 7:
    win_size = 1
    net_dict["output"]["unit"].update({
      "const1": {"class": "constant", "value": 1, "is_output_layer": True},
      "const_win_size": {"class": "constant", "value": win_size},
      "segment_starts": {  # (B,)
        "class": "switch", "condition": "prev:output_is_not_blank", "true_from": ":i",
        "false_from": "prev:segment_starts", "initial_output": 0},
      # check if start is 0 - in this case, we are still in the first segment (start is 0 until we arrive at the
      # next segment)
      "start_is_0": {"class": "compare", "from": "prev:segment_starts", "kind": "equal", "value": 0},

      # segment: from last border to current position
      "segment_lens_a": {"class": "combine", "kind": "sub", "from": [":i", "segment_starts"]},
      # current position - window size
      "segment_lens_b00": {"class": "combine", "kind": "sub", "from": [":i", "const_win_size"]},
      # + 1 (in case that window size is equal to current position)
      "segment_lens_b0": {"class": "combine", "kind": "add", "from": ["const1", "segment_lens_b00"]},
      # window size - current position (in case that window size is bigger than current position)
      "segment_lens_b1": {"class": "combine", "kind": "sub", "from": ["const_win_size", ":i"]},
      "len_smaller_0": {"class": "compare", "from": "segment_lens_b00", "kind": "less", "value": 0},
      # if window is larger than current position, use all previous frames, otherwise use window size
      "segment_lens_b": {
        "class": "switch", "condition": "len_smaller_0", "true_from": ":i", "false_from": "segment_lens_b0"},
      # for the first segment (beginning at 0), use fix left windows instead of the whole segment
      "segment_lens0": {
        "class": "switch", "condition": "start_is_0", "true_from": "segment_lens_b", "false_from": "segment_lens_a"},
      "segment_lens": {"class": "combine", "kind": "add", "from": ["segment_lens0", "const1"]},  # (B,)

      "segments": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},

      "att_query": {
        "class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
      # [B, D]

      "enc_ctx_seg0": {"class": "copy", "from": "segments"}, "enc_ctx_seg": {  # [B,D]
        "class": "linear", "from": "enc_ctx_seg0", "activation": None, "with_bias": False,
        "n_out": eval("EncKeyTotalDim")}, "enc_val_seg": {"class": "copy", "from": "segments"},

      'att_energy': {  # [B,t_sliced,1]
        "class": "dot", "red1": "f", "red2": "f", "var1": "dyn:-1", "var2": None,
        "from": ['enc_ctx_seg', 'att_query']}, 'att_weights0': {  # [B,t_sliced,1]
        "class": "softmax_over_spatial", "axis": "dyn:-1", "from": 'att_energy',
        "energy_factor": eval("EncKeyPerHeadDim") ** -0.5}, 'att_weights': {
        "class": "dropout", "dropout_noise_shape": {"*": None},  # [B,t_sliced,1]
        "from": 'att_weights0', "dropout": eval("AttentionDropout")}, 'att0': {
        "class": "dot", "from": ['att_weights', 'enc_val_seg'], "red1": "dyn:-1", "red2": "dyn:-1", "var1": "f",
        "var2": "f"},  # (B, 1, V)
      "att": {"class": "merge_dims", "from": "att0", "axes": ["static:0", "static:1"]},  # [B,V]
    })
  if attention_type == 8:
    net_dict["output"]["unit"].update({
      "const1": {"class": "constant", "value": 1, "is_output_layer": True},
      "segment_starts": {  # (B,)
        "class": "switch", "condition": "prev:output_is_not_blank", "true_from": ":i",
        "false_from": "prev:segment_starts", "initial_output": 0},
      "segment_lens0": {"class": "combine", "kind": "sub", "from": [":i", "segment_starts"]},
      "segment_lens": {"class": "combine", "kind": "add", "from": ["segment_lens0", "const1"]},  # (B,)

      "segments": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},

      "att_query": {"class": "linear", "from": "am", "activation": None, "with_bias": False,
                    "n_out": eval("EncKeyTotalDim")},
      # [B, D]

      "enc_ctx_seg0": {"class": "copy", "from": "segments"}, "enc_ctx_seg": {  # [B,D]
        "class": "linear", "from": "enc_ctx_seg0", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
      "enc_val_seg": {"class": "copy", "from": "segments"},

      'att_energy': {  # [B,t_sliced,1]
        "class": "dot", "red1": "f", "red2": "f", "var1": "dyn:-1", "var2": None, "from": ['enc_ctx_seg', 'att_query']},
      'att_weights0': {  # [B,t_sliced,1]
        "class": "softmax_over_spatial", "axis": "dyn:-1", "from": 'att_energy',
        "energy_factor": eval("EncKeyPerHeadDim") ** -0.5},
      'att_weights': {
        "class": "dropout", "dropout_noise_shape": {"*": None},  # [B,t_sliced,1]
        "from": 'att_weights0', "dropout": eval("AttentionDropout")},
      'att0': {
        "class": "dot", "from": ['att_weights', 'enc_val_seg'], "red1": "dyn:-1", "red2": "dyn:-1", "var1": "f",
        "var2": "f"},  # (B, 1, V)
      "att1": {"class": "merge_dims", "from": "att0", "axes": ["static:0", "static:1"]},  # [B,V]

      "att": {"class": "switch", "condition": "prev:output_is_not_blank", "false_from": "am", "true_from": "prev:att1"}
    })