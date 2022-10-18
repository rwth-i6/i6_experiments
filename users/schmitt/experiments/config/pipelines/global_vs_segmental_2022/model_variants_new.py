import copy
import numpy as np


def build_alias(**kwargs):
  alias = ""
  alias += kwargs["model_type"]

  if kwargs["model_type"] == "seg":
    assert kwargs.keys() == default_variant_transducer.keys()
    kwargs = copy.deepcopy(kwargs)
    kwargs = {k.replace("_", "-"): v for k, v in kwargs.items()}

    alias += "." + kwargs["label-type"]
    if kwargs["concat-seqs"]:
      alias += ".concat"
    if kwargs["epoch-split"] != 6:
      alias += ".ep-split-%d" % kwargs["epoch-split"]
    if kwargs["ctx-size"] == "inf":
      alias += ".full-ctx"
    elif type(kwargs["ctx-size"]) == int:
      alias += ".ctx%d" % kwargs["ctx-size"]
    else:
      raise NotImplementedError
    alias += ".time-red%d" % int(np.prod(kwargs["time-red"]))
    if kwargs["enc-type"] != "lstm":
      alias += ".%s" % kwargs["enc-type"]
    if kwargs["hybrid-hmm-like-label-model"]:
      alias += ".hybrid-hmm"
    if kwargs["fast-rec"]:
      alias += ".fast-rec"
    if kwargs["fast-rec-full"]:
      alias += ".fast-rec-full"
    if kwargs["sep-sil-model"]:
      alias += ".sep-sil-model-%s" % kwargs["sep-sil-model"]
    if kwargs["exclude-sil-from-label-ctx"]:
      alias += ".no-sil-in-ctx"
    if not kwargs["use-attention"]:
      alias += ".no-att"
      alias += ".am{}".format(kwargs["lstm-dim"] * 2)
    else:
      if kwargs["att-area"] == "win":
        if type(kwargs["att-win-size"]) == int:
          alias += ".local-win{}".format(str(kwargs["att-win-size"]))
        else:
          alias += ".global"
        alias += ".{}-att".format(kwargs["att-type"])
        if kwargs["att-weight-feedback"] and kwargs["att-type"] == "mlp":
          alias += ".with-feedback"
      elif kwargs["att-area"] == "seg":
        if kwargs["att-seg-clamp-size"]:
          alias += ".clamped{}".format(kwargs["att-seg-clamp-size"])
        alias += ".seg"
        if kwargs["att-seg-right-size"] and kwargs["att-seg-left-size"] and kwargs["att-seg-right-size"] == kwargs[
          "att-seg-left-size"]:
          if type(kwargs["att-seg-right-size"]) == int:
            alias += ".plus-local-win{}".format(2 * kwargs["att-seg-left-size"])
          elif kwargs["att-seg-right-size"] == "full":
            alias += ".plus-global-win"
        elif kwargs["att-seg-right-size"]:
          if type(kwargs["att-seg-right-size"]) == int:
            alias += ".plus-right-win{}".format(kwargs["att-seg-right-size"])
        alias += ".{}-att".format(kwargs["att-type"])
      if kwargs["att-ctx-with-bias"]:
        alias += ".ctx-w-bias"
      if kwargs["att-query"] != "lm":
        alias += ".query-%s" % kwargs["att-query"]
      alias += ".am{}".format(kwargs["lstm-dim"] * 2)
      # alias += ".key{}".format(kwargs["lstm-dim"])
      # alias += ".query-{}".format(kwargs["att-query-in"]).replace("base:", "")
      if kwargs["att-num-heads"] != 1:
        alias += "." + str(kwargs["att-num-heads"]) + "heads"
      if kwargs["att-seg-use-emb"]:
        alias += ".with-seg-emb" + str(kwargs["att-seg-emb-size"])
        if kwargs["att-seg-emb-query"]:
          alias += "+query"

    if kwargs["label-smoothing"]:
      alias += ".label-smoothing%s" % str(kwargs["label-smoothing"]).replace(".", "")
    if kwargs["scheduled-sampling"]:
      alias += ".scheduled-sampling%s" % str(kwargs["scheduled-sampling"]).replace(".", "")

    if kwargs["emit-extra-loss"]:
      alias += ".emit-extra-loss%.1f" % kwargs["emit-extra-loss"]

    if "slow-rnn-inputs" in kwargs:
      alias += ".slow-rnn-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["slow-rnn-inputs"]]).replace("base:", "").replace("prev:", "prev_")

    if kwargs["prev-att-in-state"]:
      alias += ".prev-att-in-state"

    alias += ".%s-" % kwargs["length-model-type"]
    alias += "length-model-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["length-model-inputs"]])

    if "readout-inputs" in kwargs:
      alias += ".readout-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["readout-inputs"]]).replace("base:", "")

    if "emit-prob-inputs" in kwargs:
      alias += ".emit-prob-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["emit-prob-inputs"]])

    if kwargs["length-model-focal-loss"] != 0.0:
      alias += ".length-focal-%s" % kwargs["length-model-focal-loss"]
    if kwargs["label-model-focal-loss"] != 0.0:
      alias += ".label-focal-%s" % kwargs["label-model-focal-loss"]

    if kwargs["prev-target-in-readout"]:
      alias += ".prev-target-in-readout"
    if kwargs["weight-dropout"] != 0.1:
      alias += ".weight-drop%s" % kwargs["weight-dropout"]
    if kwargs["pretraining"] != "old":
      alias += ".%s-pre" % kwargs["pretraining"]
    if kwargs["pretrain-reps"] is not None:
      alias += ".%spretrain-reps" % kwargs["pretrain-reps"]
    if not kwargs["att-ctx-reg"]:
      alias += ".no-ctx-reg"
    if kwargs["chunk-size"] != 60:
      alias += ".chunk-size-%s" % kwargs["chunk-size"]

    if kwargs["direct-softmax"]:
      alias += ".dir-soft"

    if kwargs["efficient-loss"]:
      alias += ".eff-loss"

  else:
    assert kwargs["model_type"] == "glob"
    assert kwargs.keys() == default_variant_enc_dec.keys()
    kwargs = copy.deepcopy(kwargs)
    kwargs = {k.replace("_", "-"): v for k, v in kwargs.items()}

    alias += ".%s-model" % kwargs["glob-model-type"]
    alias += "." + kwargs["label-type"]
    if kwargs["concat-seqs"]:
      alias += ".concat"
    if kwargs["epoch-split"] != 6:
      alias += ".ep-split-%d" % kwargs["epoch-split"]
    alias += ".time-red%d" % int(np.prod(kwargs["time-red"]))
    alias += ".am%d" % (kwargs["lstm-dim"] * 2)
    # alias += ".att-num-heads%d" % kwargs["att-num-heads"] * 2
    if kwargs["pretrain-reps"] is not None:
      alias += ".%spretrain-reps" % kwargs["pretrain-reps"]

    if kwargs["glob-model-type"] == "best":
      if kwargs["weight-dropout"] != 0.0:
        alias += ".weight-drop%s" % kwargs["weight-dropout"]
      if not kwargs["with-state-vector"]:
        alias += ".no-state-vector"
      if not kwargs["with-weight-feedback"]:
        alias += ".no-weight-feedback"
      if not kwargs["prev-target-in-readout"]:
        alias += ".no-prev-target-in-readout"
      if not kwargs["use-l2"]:
        alias += ".no-l2"
      if kwargs["att-ctx-with-bias"]:
        alias += ".ctx-use-bias"
      if kwargs["focal-loss"] != 0.0:
        alias += ".focal-loss-%s" % kwargs["focal-loss"]
      if kwargs["pretrain-type"] != "best":
        alias += ".pretrain-%s" % kwargs["pretrain-type"]
      if kwargs["att-ctx-reg"]:
        alias += ".ctx-reg"

  alias += "." + kwargs["segment-selection"] + "-segs"

  return alias


default_variant_transducer = dict(
  model_type="seg", att_type="dot", att_area="win", att_win_size=5, att_seg_left_size=None, att_seg_right_size=None,
  att_seg_clamp_size=None, att_seg_use_emb=False, att_num_heads=1, att_seg_emb_size=2, att_weight_feedback=False,
  label_smoothing=None, scheduled_sampling=False, prev_att_in_state=True, pretraining="old", att_ctx_reg=True,
  length_model_inputs=["prev_non_blank_embed", "prev_out_is_non_blank", "am"], use_attention=True,
  label_type="phonemes", lstm_dim=1024, efficient_loss=False, emit_extra_loss=None, ctx_size="inf", time_red=[1],
  fast_rec=False, sep_sil_model=None, fast_rec_full=False, segment_selection="all", length_model_type="frame",
  hybrid_hmm_like_label_model=False, att_query="lm", length_model_focal_loss=2.0, label_model_focal_loss=2.0,
  prev_target_in_readout=False, weight_dropout=0.1, pretrain_reps=None, direct_softmax=False, att_ctx_with_bias=False,
  exclude_sil_from_label_ctx=False, max_seg_len=None, chunk_size=60, enc_type="lstm", concat_seqs=False, epoch_split=6)

default_variant_enc_dec = dict(
  model_type="glob", att_num_heads=1, lstm_dim=1024, label_type="bpe", time_red=[3, 2], segment_selection="all",
  glob_model_type="old", pretrain_reps=None, weight_dropout=0.0, with_state_vector=True, att_ctx_reg=False,
  with_weight_feedback=True, prev_target_in_readout=True, use_l2=True, att_ctx_with_bias=False, focal_loss=0.0,
  pretrain_type="best", concat_seqs=False, epoch_split=6)

model_variants = {
  # BPE GLOBAL

  # ---------------------------------------------------------------------------------------------------
  # BPE with silence without split

  # ---------------------------------------------------------------------------------------------------
  # BPE without separate silence

  # "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False),
  # },
  # "seg.bpe.concat.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False, concat_seqs=True),
  # },
  # "seg.bpe.concat.ep-split-12.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False, concat_seqs=True,
  #                  epoch_split=12),
  # },
  # "seg.bpe.full-ctx.time-red6.conf-tim.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False, enc_type="conf-tim"),
  # },
  # "seg.bpe.full-ctx.time-red6.conf-wei.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False, enc_type="conf-wei"),
  # },


  # Table 1 - lines 1+2; Table 6 - line 3
  "glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=1, with_weight_feedback=True,
      pretrain_type="like-seg", att_ctx_with_bias=True, use_l2=True, label_type="bpe-with-sil-split-sil"),
  },
  # Table 1 - line 3; Table 3 - lines 1-6; Table 5 - line 1; Table 6 - line 5; Table 7 - line 2; Table 9 - column 6
  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe-with-sil-split-sil",
                   lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False, att_ctx_with_bias=True),
  },

  # Table 6 - line 1; Table 9 - column 4; Table 10 - line 7
  "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=1, with_weight_feedback=False,
      pretrain_type="like-seg", att_ctx_with_bias=True, use_l2=True, label_type="bpe"),
  },
  # Table 6 - line 2
  "glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=1, with_weight_feedback=False,
      pretrain_type="like-seg", att_ctx_with_bias=True, use_l2=True, label_type="bpe-with-sil"),
  },
  # Table 6 - line 4; Table 7 - line 1; Table 9 - column 5; Table 10 - lines 8+9
  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=False),
  },
  # Table 8 - line 1
  "glob.best-model.bpe.time-red6.am2048.no-l2.ctx-use-bias.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best",
      att_ctx_with_bias=True, use_l2=False, label_type="bpe"),
  },
  # Table 8 - line 2
  "glob.best-model.bpe.time-red6.am2048.no-state-vector.no-l2.ctx-use-bias.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best",
      att_ctx_with_bias=True, use_l2=False, label_type="bpe", with_state_vector=False),
  },
  # Table 8 - line 4
  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=True),
  },
  # Table 8 - line 3
  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, att_ctx_reg=True),
  },


}
