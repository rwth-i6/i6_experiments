from enum import Enum
from recipe.i6_core.returnn.config import CodeWrapper
import copy
import math
# from returnn.tf.util.data import Dim

# global use_attention
# global Dim
# global att_seg_emb_size
# global att_seg_use_emb
# global att_win_size
# global task
# global EncValueTotalDim
# global EncValueDecFactor
# global att_weight_feedback
# global att_type
# global att_seg_clamp_size
# global att_seg_left_size
# global att_seg_right_size
# global att_area
# global att_query_in
# global att_seg_emb_query
# global AttNumHeads
# global EncValuePerHeadDim


def add_attention(
  net_dict, att_seg_emb_size, att_seg_use_emb, att_win_size, task, EncValueTotalDim, EncValueDecFactor, EncKeyTotalDim,
  att_weight_feedback, att_type, att_seg_clamp_size, att_seg_left_size, att_seg_right_size, att_area, att_ctx_reg,
  AttNumHeads, EncValuePerHeadDim, l2, EncKeyPerHeadDim, AttentionDropout, att_query, ctx_with_bias,
  segment_center_window_size=None):
  """This function expects a network dictionary of an "extended transducer" model and adds a self-attention mechanism
  according to some parameters set in the returnn config.

  The parameters are currently set inside the config in order to not modify the header of this function, which would
  lead to a different hash value of the ReturnnConfig object. The currently supported parameters are:

  :param att_type:
  :param att_area:
  :param att_win_size:
  :param att_seg_left_size:
  :param att_seg_right_size:
  :param att_seg_clamp_size:
  :param mask_att:
  :param att_locations:
  :param att_seg_use_emb:

  :param net_dict:
  :param attention_type: int in [0, 1, 2, ...]
  :return: dict net_dict with added attention mechanism
  """

  # assert not (att_area == "win" and att_win_size == "full"), "Currently not implemented since readout is masked"

  net_dict = copy.deepcopy(net_dict)

  # during training, the readout layer and the attention are masked to be only calculated in emit steps to save time and
  # memory
  # net_dict["label_model"]["unit"].update({
  #   "att_masked": {
  #     "class": "masked_computation", "mask": "is_label" if task == "train" else "const_true",
  #     "from": "prev_non_blank_embed", "unit": {
  #       "class": "subnetwork", "from": "data", "subnetwork": {
  #         "output": {"class": "copy", "from": "att"}}}},
  #   "att_unmask": {"class": "unmask", "from": "att_masked", "mask": "is_label" if task == "train" else "const_true"},
  #   "att": {"class": "copy", "from": "att_unmask"},  # [B,L]
  # })
  if task == "train":
    readout_level_dict = net_dict["label_model"]["unit"]
  else:
    readout_level_dict = net_dict["output"]["unit"]

  att_heads_tag = CodeWrapper(
    "Dim(kind=Dim.Types.Spatial, description='att_heads', dimension=%d)" % AttNumHeads)

  def add_segment_information():
    # define segment as all frames since the last non-blank output
    # the last non-blank frame is excluded; the current frame is included
    net_dict["output"]["unit"].update({
      "const1": {"class": "constant", "value": 1},
      "segment_starts": {  # (B,)
        "class": "switch", "condition": "prev:output_emit", "true_from": ":i",
        "false_from": "prev:segment_starts", "initial_output": 0,
        "is_output_layer": False if att_win_size == "full" and task != "train" else True},
      "segment_lens0": {"class": "combine", "kind": "sub", "from": [":i", "segment_starts"]},
      "segment_lens": {
        "class": "combine", "kind": "add", "from": ["segment_lens0", "const1"],
        "is_output_layer": False if att_win_size == "full" and task != "train" else True},  # (B,)
    })

    if segment_center_window_size is not None:
      assert type(segment_center_window_size) == int or segment_center_window_size == 0.5
      net_dict["output"]["unit"]["segment_starts1"] = net_dict["output"]["unit"]["segment_starts"].copy()
      net_dict["output"]["unit"]["segment_lens1"] = net_dict["output"]["unit"]["segment_lens"].copy()
      net_dict["output"]["unit"]["segment_starts1"]["false_from"] = "prev:segment_starts1"
      net_dict["output"]["unit"]["segment_lens0"]["from"][1] = "segment_starts1"
      net_dict["output"]["unit"].update({
        "segment_starts2": {
          "class": "eval", "from": ["segment_starts1", "segment_lens1"], "is_output_layer": True,
          "eval": "source(0) + source(1) - %d" % segment_center_window_size
        },
        "segment_ends1": {
          "class": "eval", "from": ["segment_starts1", "segment_lens1"],
          "eval": "source(0) + source(1) + %d" % math.ceil(segment_center_window_size)
        },
        "segment_lens": {
          "class": "eval", "from": ["segment_ends", "segment_starts"], "is_output_layer": True,
          "eval": "source(0) - source(1)"
        },
        "seq_lens": {"class": "length", "from": "base:encoder"},
        "seq_end_too_far": {"class": "compare", "from": ["segment_ends1", "seq_lens"], "kind": "greater"},
        "seq_start_too_far": {"class": "compare", "from": ["segment_starts2"], "value": 0, "kind": "less"},
        "segment_ends": {"class": "switch", "condition": "seq_end_too_far", "true_from": "seq_lens",
                         "false_from": "segment_ends1"},
        "segment_starts": {"class": "switch", "condition": "seq_start_too_far", "true_from": 0,
                           "false_from": "segment_starts2"},
      })

  def add_segmental_embedding_vector():
    def get_one_hot_embedding(nd, idx):
      return {"class": "copy", "from": ["const0.0" if i != idx else "const1.0" for i in range(nd)]}

    def get_conditions():
      if att_seg_emb_size >= 2:
        conditions = {
          "is_in_segment": {
            "class": "compare", "from": ["segment_left_index", "segment_indices", "segment_right_index"],
            "kind": "less_equal"}, }
      if att_seg_emb_size >= 3:
        conditions.update({
          "left_of_segment": {
            "class": "compare", "from": ["segment_left_index", "segment_indices"], "kind": "greater"}})
      if att_seg_emb_size == 4:
        conditions = dict({
          "is_cur_step": {
            "class": "compare", "from": ["segment_indices", "segment_right_index"], "kind": "equal"}, **conditions})

      return conditions

    if att_seg_use_emb and att_seg_emb_size:
      conditions = get_conditions()
      net_dict["output"]["unit"].update(conditions)

      for i in range(att_seg_emb_size):
        net_dict["output"]["unit"]["emb" + str(i)] = get_one_hot_embedding(att_seg_emb_size, i)

      # TODO: this condition is due to some legacy models which still need to do search
      # the legacy model used 2D embedding and used [0,1] in case that the frame was in the segment
      # this new model uses [1,0]
      if att_seg_emb_size > 2:
        for i, cond in enumerate(conditions):
          if i == len(conditions) - 1:
            net_dict["output"]["unit"].update({
              "embedding" + str(i): {
                "class": "switch", "condition": cond, "true_from": "emb" + str(i), "false_from": "emb" + str(i + 1)}, })
          else:
            net_dict["output"]["unit"].update({
              "embedding" + str(i): {
                "class": "switch", "condition": cond, "true_from": "emb" + str(i),
                "false_from": "embedding" + str(i + 1)}, })
      else:
        net_dict["output"]["unit"].update({
          "embedding0": {
            "class": "switch", "condition": "is_in_segment", "true_from": "emb1", "false_from": "emb0"}, })

  def add_local_window():
    """Local window attention"""
    att_time_tag = CodeWrapper(
      'Dim(kind=Dim.Types.Spatial, description="att_t", dimension=%d)' % att_win_size)
    # In this case, the local window has a given fixed size

    # extract the window inside the encoder (more efficient than doing in inside the decoder)
    net_dict.update({
      "enc_ctx0": {
        "class": "linear", "from": "encoder", "activation": None, "with_bias": False,
        "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},  # (B,T,D)
      "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": att_win_size},  # [B,T,W,D]
      "enc_val": {"class": "copy", "from": "encoder"},  # (B,T,V)
      "enc_val_win": {"class": "window", "from": "enc_val", "window_size": att_win_size},  # [B,T,W,V]
    })
    # context and value are just the windows at the corresponding step in the decoder
    net_dict["output"]["unit"].update({
      "att_ctx0": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
      "att_ctx": {
        "class": "reinterpret_data", "from": "att_ctx0", "set_dim_tags": {"stag:enc_ctx_win:window": att_time_tag}},
      "att_val0": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,V],
      "att_val": {
        "class": "reinterpret_data", "from": "att_val0",
        "set_dim_tags": {"stag:enc_val_win:window": att_time_tag}}})
    readout_level_dict.update({
      "att_ctx": {"class": "copy", "from": "base:att_ctx"}, "att_val": {"class": "copy", "from": "base:att_val"}
    })

  def add_global_window():
    def add_global_with_weight_feedback():
      net_dict.update({
        "inv_fertility": {
          "class": "linear", "activation": "sigmoid", "with_bias": False, "from": "encoder", "n_out": 1}, })
      net_dict["output"]["unit"].update({
        "weight_feedback": {
          "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
          "n_out": EncKeyTotalDim},
        "accum_att_weights": {
          "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
          "eval": "source(0) + source(1) * source(2) * 0.5", "out_type": {"dim": 1, "shape": (None, 1)}}, })
      net_dict["output"]["unit"]["att_energy_in"]["from"].append("weight_feedback")

      if not (att_seg_use_emb and att_seg_emb_size):
        net_dict.update({
          "enc_ctx": {  # (B, T, D)
            "class": "linear", "from": "encoder", "activation": None, "with_bias": False,
            "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},
          "enc_val": {"class": "copy", "from": ["encoder"]}, })
        net_dict["output"]["unit"]["att_energy_in"]["from"] = ["base:enc_ctx" if item == "att_ctx" else item for item
                                                               in net_dict["output"]["unit"]["att_energy_in"]["from"]]
        # see explanation above on the "att" layer
        if task == "train":
          net_dict["output"]["unit"]['att']["base"] = "base:enc_val"
        else:
          net_dict["output"]["unit"]['att0']["base"] = "base:enc_val"
      else:
        raise NotImplementedError
        # net_dict.update({
        #   "segment_indices": {"class": "range_in_axis", "from": "encoder", "axis": "t"}, })
        #
        # net_dict["output"]["unit"].update({
        #   "segment_left_index": {"class": "copy", "from": ["segment_starts"]},
        #   "segment_right_index": {"class": "combine", "from": ["segment_starts", "segment_lens0"], "kind": "add"},
        #   "att_val": {"class": "copy", "from": ["base:encoder", "embedding0"]}, "att_ctx": {  # (B, T, D)
        #     "class": "linear", "from": ["base:encoder", "embedding0"], "activation": None, "with_bias": False,
        #     "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2}})
        # for cond in ["is_in_segment", "left_of_segment", "is_cur_step"]:
        #   if cond in net_dict["output"]["unit"]:
        #     net_dict["output"]["unit"][cond]["from"] = ["base:segment_indices" if item == "segment_indices" else item
        #                                                 for item in net_dict["output"]["unit"][cond]["from"]]

    def add_global_without_weight_feedback():
      net_dict.update({
        "encoder_new": {"class": "reinterpret_data", "from": "encoder", "set_dim_tags": {"t": att_time_tag}}})
      readout_level_dict.update({
        "att_ctx0": {  # (B, T, D)
          "class": "linear", "from": ["base:encoder_new"], "activation": None, "with_bias": False,
          "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},
        "att_ctx": {
          "class": "copy", "from": ["att_ctx0"]},
        "att_val": {  # (B,T,V)
          "class": "copy", "from": ["base:encoder_new"]}})

      if att_seg_use_emb and att_seg_emb_size:
        net_dict["output"]["unit"]["att_val0"] = net_dict["output"]["unit"]["att_val"].copy()
        net_dict["output"]["unit"]["att_ctx"] = net_dict["output"]["unit"]["att_ctx0"].copy()
        net_dict["output"]["unit"]["att_ctx"]["from"] = ["att_val0", "embedding0"]
        net_dict["output"]["unit"].update({
          "segment_indices": {"class": "range_in_axis", "from": "att_val0", "axis": "t"},
          "segment_left_index": {"class": "copy", "from": ["segment_starts"]},
          "segment_right_index": {"class": "combine", "from": ["segment_starts", "segment_lens0"], "kind": "add"},
          # "att_val1": {"class": "copy", "from": ["att_val0", "embedding0"]},
          "att_val": {  # (B,T,V)
            "class": "reinterpret_data", "from": ["att_val1"], "set_axes": {
              "t": "time"}},
          "att_val1": {  # (B, T, D)
            "class": "linear", "from": ["att_val0", "embedding0"], "activation": None, "with_bias": False,
            "n_out": EncValueTotalDim // EncValueDecFactor, "L2": l2, "dropout": 0.2},
        })

    att_time_tag = CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_t")')

    add_global_without_weight_feedback()

  def add_segmental_window():
    def add_clamping():
      if att_seg_clamp_size is not None:
        # in this case, we clamp all segments to the specified size
        # this might make sense since the first segment is often much longer than the other segments
        assert type(att_seg_clamp_size) == int
        net_dict["output"]["unit"]["segment_starts0"] = net_dict["output"]["unit"]["segment_starts"].copy()
        net_dict["output"]["unit"]["segment_starts0"]["false_from"] = "prev:segment_starts0"
        net_dict["output"]["unit"]["segment_lens0_copy"] = net_dict["output"]["unit"]["segment_lens0"].copy()
        net_dict["output"]["unit"]["segment_lens0_copy"]["from"] = [":i", "segment_starts0"]
        net_dict["output"]["unit"]["segment_lens1_copy"] = net_dict["output"]["unit"]["segment_lens"].copy()
        net_dict["output"]["unit"]["segment_lens1_copy"]["from"] = ["segment_lens0_copy", "const1"]
        net_dict["output"]["unit"].update({
          "clamp_size": {"class": "constant", "value": att_seg_clamp_size},
          "clamp_mask": {"class": "compare", "from": ["segment_lens1_copy", "clamp_size"], "kind": "greater"},
          "clamped_diff": {"class": "combine", "from": ["segment_lens1_copy", "clamp_size"], "kind": "sub"},
          "clamped_start": {"class": "combine", "from": ["segment_starts0", "clamped_diff"], "kind": "add"},
          "segment_starts": {
            "class": "switch", "condition": "clamp_mask", "true_from": "clamped_start",
            "false_from": "segment_starts0"}})

    def add_left_window():
      if att_seg_left_size is not None:
        # in this case, we add a specified number of frames on the left side of the segment
        if att_seg_clamp_size is None:
          idx = 0
        else:
          idx = 1
        net_dict["output"]["unit"]["segment_starts" + str(idx)] = net_dict["output"]["unit"]["segment_starts"].copy()
        if att_seg_clamp_size is None:
          net_dict["output"]["unit"]["segment_starts" + str(idx)]["false_from"] = "prev:segment_starts" + str(idx)

        if type(att_seg_left_size) == int:
          net_dict["output"]["unit"].update({
            "const_left_win_size": {"class": "constant", "value": att_seg_left_size}, })
        elif att_seg_left_size == "full":
          net_dict["output"]["unit"].update({
            "const_left_win_size": {"class": "copy", "from": "segment_starts" + str(idx)}, })
        net_dict["output"]["unit"].update({
          "segment_starts" + str(idx + 1): {
            "class": "combine", "from": ["segment_starts" + str(idx), "const_left_win_size"], "kind": "sub"},
          "less_than_0": {"class": "compare", "from": "segment_starts" + str(idx + 1), "value": 0, "kind": "less"},
          "segment_starts": {
            "class": "switch", "condition": "less_than_0", "true_from": 0,
            "false_from": "segment_starts" + str(idx + 1)}})

    def add_right_window():
      if att_seg_right_size is not None:
        # in this case, we add a specified number of frames on the right side of the segment
        net_dict["output"]["unit"]["segment_lens1"] = net_dict["output"]["unit"]["segment_lens"].copy()
        net_dict["output"]["unit"]["seq_lens"] = {"class": "length", "from": "base:encoder"}
        if type(att_seg_right_size) == int:
          net_dict["output"]["unit"].update({
            "const_right_win_size": {"class": "constant", "value": att_seg_right_size}, "segment_lens2": {
              "class": "combine", "kind": "add", "from": ["segment_lens1", "const_right_win_size"]},
            "max_length": {"class": "combine", "from": ["seq_lens", "segment_starts"], "kind": "sub"},
            "last_seg_idx": {"class": "combine", "from": ["segment_starts", "segment_lens2"], "kind": "add"},
            "greater_than_length": {
              "class": "compare", "from": ["last_seg_idx", "seq_lens"], "kind": "greater_equal"}, "segment_lens": {
              "class": "switch", "condition": "greater_than_length", "true_from": "max_length",
              "false_from": "segment_lens2"}, })
        elif att_seg_right_size == "full":
          net_dict["output"]["unit"].update({
            "segment_lens": {
              "class": "combine", "from": ["seq_lens", "segment_starts"], "kind": "sub"}, })

    def add_embedding():
      if att_seg_use_emb and att_seg_emb_size and (att_seg_left_size or att_seg_right_size):
        # in this case, we add an one-hot embedding to the encoder frames, indicating whether they belong to the
        # current segment or not
        if att_seg_left_size:
          # first in-segment-index of slice_nd output
          net_dict["output"]["unit"]["segment_left_index"] = {
            "class": "combine", "from": ["segment_starts0", "segment_starts"], "kind": "sub"}
          # the length of the segment (without additional window)
          net_dict["output"]["unit"]["real_seg_len"] = {
            "class": "combine", "from": [":i", "segment_starts0"], "kind": "sub"}
        else:
          # first in-segment-index of slice_nd output
          net_dict["output"]["unit"]["segment_left_index"] = {"class": "constant", "value": 0}
          # the length of the segment (without additional window)
          net_dict["output"]["unit"]["real_seg_len"] = {
            "class": "combine", "from": [":i", "segment_starts"], "kind": "sub"}
        # last in-segment-index of slice_nd output
        net_dict["output"]["unit"]["segment_right_index"] = {
          "class": "combine", "from": ["segment_left_index", "real_seg_len"], "kind": "add"}

        # 'segments' is redefined here, therefore we need to rename the existing layer
        net_dict["output"]["unit"]["segments1"] = net_dict["output"]["unit"]["segments"].copy()

        net_dict["output"]["unit"].update({
          "segment_indices": {"class": "range_in_axis", "from": "segments1", "axis": "stag:att_t"},
          "segments": {"class": "copy", "from": ["segments1", "embedding0"]}})


    """Segmental Attention: a segment is defined as current frame + all previous blank frames"""
    if att_area == "seg":
      att_time_tag = CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_t")')

      # add the base attention mechanism here. The variations below define the segment boundaries (segment_starts and
      # segment_lens)
      readout_level_dict.update({
        "segments0": {  # [B,t_sliced,D]
          "class": "slice_nd", "from": "base:encoder",
          "start": "segment_starts", "size": "segment_lens"},
        "segments": {
          "class": "reinterpret_data", "from": "segments0", "set_dim_tags": {"stag:sliced-time:segments": att_time_tag}},
        "att_ctx": {  # [B,D]
          "class": "linear", "from": "segments", "activation": None,
          "with_bias": True if ctx_with_bias else False, "n_out": EncKeyTotalDim,
          "L2": l2 if att_ctx_reg else None, "dropout": 0.2 if att_ctx_reg else 0.0},
        "att_val": {"class": "copy", "from": "segments"} if not (att_seg_use_emb and att_seg_emb_size) else {  # (B, T, D)
          "class": "linear", "from": ["segments"], "activation": None, "with_bias": False,
          "n_out": EncValueTotalDim // EncValueDecFactor, "L2": l2, "dropout": 0.2}, })

      add_clamping()
      add_left_window()
      add_right_window()
      add_embedding()

  def add_attention_query():
    if "am" in att_query:
      readout_level_dict.update({
        "am_position0": {
          "class": "combine", "kind": "add", "from": ["segment_lens", "segment_starts"]},
        "const1": {"class": "constant", "value": 1}, "const0": {"class": "constant", "value": 0},
        "am_position1": {
          "class": "combine", "kind": "sub", "from": ["am_position0", "const1"]},
        "am_smaller_0": {
          "class": "compare", "from": ["am_position1", "const0"], "kind": "less"},
        "am_position": {
          "class": "switch", "condition": "am_smaller_0", "true_from": "const0", "false_from": "am_position1"},
        "am": {
          "class": "gather", "from": "base:encoder", "position": "am_position", "axis": "t"},
      })
    readout_level_dict["att_query"] = {  # (B,D)
      "class": "linear", "from": att_query, "activation": None, "with_bias": False,
      "n_out": EncKeyTotalDim, "is_output_layer": False}

  def add_energies():
    if att_type == "dot":
      assert AttNumHeads == 1
      # use the dot-product to calculate the energies
      readout_level_dict.update({
        'att_energy0': {  # (B, t_att, 1)
          "class": "dot", "red1": "f", "red2": "f", "var1": "stag:att_t", "var2": None,
          "from": ['att_ctx', 'att_query'], "add_var2_if_empty": False}, "att_energy1": {
          "class": "expand_dims", "from": "att_energy0", "axis": "f"},
        "att_energy": {"class": "reinterpret_data", "from": "att_energy1", "set_dim_tags": {"f": att_heads_tag}}})
    elif att_type == "mlp":
      # use an MLP to calculate the energies
      readout_level_dict.update({
        'att_energy_in': {  # (B, t_att, D)
          "class": "combine", "kind": "add", "from": ["att_ctx", "att_query"], "n_out": EncKeyTotalDim},
        "energy_tanh": {
          "class": "activation", "activation": "tanh", "from": ["att_energy_in"]},  # (B, W, D)
        "att_energy0": {  # (B, t_att, 1)
          "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": AttNumHeads},
        "att_energy": {"class": "reinterpret_data", "from": "att_energy0", "set_dim_tags": {"f": att_heads_tag},
                       "is_output_layer": False}})
    elif att_type == "mlp_complex":
      # use more complex MLP to calculate the energies
      readout_level_dict.update({
        'att_energy_in': {  # (B, t_att, D)
          "class": "combine", "kind": "add", "from": ["att_ctx", "att_query"], "n_out": EncKeyTotalDim},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["att_energy_in"]},  # (B, W, D)
        "att_energy0": {  # (B, t_att, 1)
          "class": "linear", "activation": None, "with_bias": True, "from": ["energy_tanh"], "n_out": EncKeyTotalDim},
        "att_energy1": {  # (B, t_att, 1)
          "class": "linear", "activation": "tanh", "with_bias": True, "from": ["att_energy0"], "n_out": AttNumHeads},
        "att_energy": {"class": "reinterpret_data", "from": "att_energy1", "set_dim_tags": {"f": att_heads_tag},
                       "is_output_layer": False}})

  def add_weights():
    readout_level_dict.update({
      'att_weights0': {
        "class": "softmax_over_spatial", "from": 'att_energy', "axis": "stag:att_t",
        "energy_factor": EncKeyPerHeadDim ** -0.5},
      'att_weights': {
        "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0',
        "dropout": AttentionDropout, "is_output_layer": False}})

  def add_weight_feedback():
    if (att_seg_use_emb and att_seg_emb_size) or not (att_area == "win" and att_win_size == "full"):
      raise NotImplementedError

    net_dict["inv_fertility"] = {
      "class": "linear", "activation": "sigmoid", "with_bias": False, "from": "encoder_new", "n_out": AttNumHeads}  # [B,att_t,heads]

    net_dict["output"]["unit"].update({
      "accum_att_weights": {  # [B,att_t,1]
        "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
        "eval": "source(0) + source(1) * source(2) * 0.5", "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)}},
      "weight_feedback": {
        "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
        "n_out": EncKeyTotalDim}})

    net_dict["output"]["unit"]["att_energy_in"]["from"].append("weight_feedback")

  def add_attention_value_heads():
    readout_level_dict.update({
      "att_val_split0": {
        "class": "split_dims", "axis": "f",
        "dims": (AttNumHeads, -1), "from": "att_val"},
      "att_val_split": {
        "class": "reinterpret_data", "from": "att_val_split0",
        "set_dim_tags": {"dim:" + str(AttNumHeads): att_heads_tag}}})

  def add_attention_vector():
    # TODO: also, I employed a dirty fix here: during search, the attention vector goes through a redundant switch layer,
    # TODO: which is conditioned on the previous output. this solves the "buggy beam" of the masked attention layer during search
    # if task == "train":
    # calculate attention in all other cases
    readout_level_dict.update({
      'att0': {
        "class": "dot", "from": ["att_val_split", 'att_weights'], "reduce": "stag:att_t", "var1": "f", "var2": None,
        "add_var2_if_empty": False},  # (B, 1, V)
      "att": {"class": "merge_dims", "axes": "except_time", "from": "att0"}})
    # else:
    #   # calculate attention in all other cases
    #   readout_level_dict.update({
    #     'att0': {
    #       "class": "dot", "from": ["att_val_split", 'att_weights'], "reduce": "stag:att_t", "var1": "f",
    #       "var2": None, "add_var2_if_empty": False},  # (B, 1, V)
    #     "att1": {"class": "merge_dims", "axes": "except_time", "from": "att0"},
    #     "att": {
    #       "class": "switch", "condition": "prev:output_is_not_blank", "true_from": "att1", "false_from": "att1"}})

  def add_pooling():
    att_time_tag = CodeWrapper('Dim(kind=Dim.Types.Spatial, description="att_t")')
    readout_level_dict.update({
      "segments0": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},
      "segments": {
        "class": "reinterpret_data", "from": "segments0", "set_dim_tags": {"stag:sliced-time:segments": att_time_tag}},
      "att": {
        "class": "reduce", "mode": "mean", "axes": ["stag:att_t"], "from": "segments"}})


  # if (att_seg_use_emb and att_seg_emb_size) or att_area == "seg":
  add_segment_information()
  if att_type != "pooling":
    if att_seg_use_emb and att_seg_emb_size:
      add_segmental_embedding_vector()
    if att_area == "win":
      if type(att_win_size) == int:
        add_local_window()
      else:
        assert att_win_size == "full"
        add_global_window()
    else:
      assert att_area == "seg"
      add_segmental_window()
    add_attention_query()
    add_energies()
    add_weights()
    if att_weight_feedback:
      add_weight_feedback()
    add_attention_value_heads()
    add_attention_vector()
  else:
    add_pooling()

  return net_dict

