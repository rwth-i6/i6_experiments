net_dict = {
  "att_weights_range": {
    "axis": att_t_dim_tag,
    "class": "range_in_axis",
    "from": "att_weights",
  },
  "att_weights_scattered": {
    "class": "scatter_nd",
    "from": "att_weights",
    "out_spatial_dim": accum_att_weights_dim_tag,
    "position": "att_weights_range",
    "position_axis": att_t_dim_tag,
  },

  "inv_fertility": {
    "class": "slice_nd",
    "from": "base:inv_fertility",
    "out_spatial_dim": att_t_dim_tag,
    "size": "segment_lens",
    "start": "segment_starts",
  },
  "inv_fertility_scattered": {
    "class": "scatter_nd",
    "from": "inv_fertility",
    "out_spatial_dim": accum_att_weights_dim_tag,
    "position": "att_weights_range",
    "position_axis": att_t_dim_tag,
  },

  "accum_att_weights0": {
    "class": "eval",
    "eval": "source(0) + source(1) * source(2) * 0.5",
    "from": [
      "accum_att_weights_scattered",
      "att_weights_scattered",
      "inv_fertility_scattered",
    ],
    "initial_output": "base:initial_output_layer",
  },
  "accum_att_weights": {
    "class": "reinterpret_data",
    "enforce_batch_major": True,
    "from": "accum_att_weights0",
  },

  "overlap_accum_weights": {
    "axis": accum_att_weights_dim_tag,
    "class": "slice_nd",
    "from": "prev:accum_att_weights",
    "initial_output": 0.0,
    "out_spatial_dim": att_t_overlap_dim_tag,
    "size": "overlap_len",
    "start": "overlap_start",
  },
  "overlap_len": {
    "class": "switch",
    "condition": "overlap_mask",
    "false_from": "overlap_len0",
    "true_from": 0,
  },
  "overlap_len0": {
    "class": "eval",
    "eval": "source(0) + source(1) - source(2)",
    "from": ["prev:segment_starts", "prev:segment_lens", "segment_starts"],
  },
  "overlap_mask": {
    "class": "compare",
    "from": "overlap_len0",
    "kind": "less",
    "value": 0,
  },
  "overlap_range": {
    "axis": att_t_overlap_dim_tag,
    "class": "range_in_axis",
    "from": "overlap_accum_weights",
  },
  "overlap_start": {
    "class": "combine",
    "from": ["prev:segment_lens", "overlap_len"],
    "kind": "sub",
  },

  "accum_att_weights_scattered0": {
    "class": "scatter_nd",
    "from": "overlap_accum_weights",
    "out_spatial_dim": accum_att_weights_dim_tag,
    "position": "overlap_range",
    "position_axis": att_t_overlap_dim_tag,
  },
  "accum_att_weights_scattered": {
    "class": "reinterpret_data",
    "enforce_batch_major": True,
    "from": "accum_att_weights_scattered0",
  },
  "prev_accum_att_weights_sliced": {
    "axis": accum_att_weights_dim_tag,
    "class": "slice_nd",
    "from": "accum_att_weights_scattered",
    "out_spatial_dim": att_t_dim_tag,
    "size": "segment_lens",
    "start": 0,
  },
  "weight_feedback": {
    "activation": None,
    "class": "linear",
    "from": "prev_accum_att_weights_sliced",
    "n_out": 1024,
    "with_bias": False,
  },
}
