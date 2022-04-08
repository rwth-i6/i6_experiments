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
    if kwargs["ctx-size"] == "inf":
      alias += ".full-ctx"
    elif type(kwargs["ctx-size"]) == int:
      alias += ".ctx%d" % kwargs["ctx-size"]
    else:
      raise NotImplementedError
    alias += ".time-red%d" % int(np.prod(kwargs["time-red"]))
    if kwargs["hybrid-hmm-like-label-model"]:
      alias += ".hybrid-hmm"
    if kwargs["fast-rec"]:
      alias += ".fast-rec"
    if kwargs["fast-rec-full"]:
      alias += ".fast-rec-full"
    if kwargs["sep-sil-model"]:
      alias += ".sep-sil-model-%s" % kwargs["sep-sil-model"]
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

    if "length-model-inputs" in kwargs:
      alias += ".length-model-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["length-model-inputs"]])

    if "readout-inputs" in kwargs:
      alias += ".readout-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["readout-inputs"]]).replace("base:", "")

    if "emit-prob-inputs" in kwargs:
      alias += ".emit-prob-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["emit-prob-inputs"]])

    if kwargs["length-model-focal-loss"] != 2.0:
      alias += ".length-focal-%s" % kwargs["length-model-focal-loss"]
    if kwargs["label-model-focal-loss"] != 2.0:
      alias += ".label-focal-%s" % kwargs["label-model-focal-loss"]

    if kwargs["prev-target-in-readout"]:
      alias += ".prev-target-in-readout"
    if kwargs["weight-dropout"] != 0.1:
      alias += ".weight-drop%s" % kwargs["weight-dropout"]
    if kwargs["pretraining"] != "old":
      alias += ".%s-pre" % kwargs["pretraining"]
    if kwargs["pretrain-reps"] is not None:
      alias += ".%spretrain-reps" % kwargs["pretrain-reps"]

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

  alias += "." + kwargs["segment-selection"] + "-segs"

  return alias


default_variant_transducer = dict(
  model_type="seg", att_type="dot", att_area="win", att_win_size=5, att_seg_left_size=None, att_seg_right_size=None,
  att_seg_clamp_size=None, att_seg_use_emb=False, att_num_heads=1, att_seg_emb_size=2, att_weight_feedback=False,
  label_smoothing=None, scheduled_sampling=False, prev_att_in_state=True, pretraining="old",
  length_model_inputs=["prev_non_blank_embed", "prev_out_is_non_blank", "am"], use_attention=True,
  label_type="phonemes", lstm_dim=1024, efficient_loss=False, emit_extra_loss=None, ctx_size="inf", time_red=[1],
  fast_rec=False, sep_sil_model=None, fast_rec_full=False, segment_selection="all",
  hybrid_hmm_like_label_model=False, att_query="lm", length_model_focal_loss=2.0, label_model_focal_loss=2.0,
  prev_target_in_readout=False, weight_dropout=0.1, pretrain_reps=None, direct_softmax=False)

default_variant_enc_dec = dict(
  model_type="glob", att_num_heads=1, lstm_dim=1024, label_type="bpe", time_red=[3, 2], segment_selection="all",
  glob_model_type="old", pretrain_reps=None, weight_dropout=0.0, with_state_vector=True,
  with_weight_feedback=True, prev_target_in_readout=True)

model_variants = {
  # BPE GLOBAL
  # "glob.new-model.bpe.time-red6.am2048.6pretrain-reps.all-segs": {
  #   "config": dict(
  #     default_variant_enc_dec, segment_selection="all", glob_model_type="new", pretrain_reps=6),
  # },
  # "glob.like-seg-model.bpe.time-red6.am2048.all-segs": {
  #   "config": dict(
  #     default_variant_enc_dec, segment_selection="all", glob_model_type="like-seg"),
  # },
  "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=6),
  },
  "glob.best-model.bpe-with-sil.time-red6.am2048.6pretrain-reps.bpe-sil-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="bpe-sil", glob_model_type="best", pretrain_reps=6,
      label_type="bpe-with-sil"),
  },
  "glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.6pretrain-reps.bpe-sil-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="bpe-sil", glob_model_type="best", pretrain_reps=6,
      label_type="bpe-with-sil-split-sil"),
  },
  "glob.best-model.phonemes.time-red1.am2048.6pretrain-reps.phonemes-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="phonemes", glob_model_type="best", pretrain_reps=6,
      label_type="phonemes", time_red=[1], lstm_dim=1024),
  },
  "glob.best-model.phonemes-split-sil.time-red1.am2048.6pretrain-reps.phonemes-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="phonemes", glob_model_type="best", pretrain_reps=6,
      label_type="phonemes-split-sil", time_red=[1], lstm_dim=1024),
  },
  "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.weight-drop0.1.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=6, weight_dropout=0.1),
  },
  "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-state-vector.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=6, with_state_vector=False),
  },
  "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-weight-feedback.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=6, with_weight_feedback=False),
  },
  "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-prev-target-in-readout.all-segs": {
    "config": dict(
      default_variant_enc_dec, segment_selection="all", glob_model_type="best", pretrain_reps=6, prev_target_in_readout=False),
  },
  # ---------------------------------------------------------------------------------------------------
  # BPE with silence without split

  # ---------------------------------------------------------------------------------------------------
  # BPE without separate silence

  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all"),
  },

  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0),
  },

  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop1e-07.new-pre.6pretrain-reps.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0000001),
  },
  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop1e-07.new-pre.6pretrain-reps.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0000001),
  },
  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop1e-07.new-pre.6pretrain-reps.dir-soft.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True, direct_softmax=True,
                   segment_selection="all", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0000001),
  },
  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0),
  },

  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.new-pre.6pretrain-reps.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True),
  },

  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.new-pre.6pretrain-reps.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new"),
  },

  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.global.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.all-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="win", att_win_size="full", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="all", label_model_focal_loss=0.0),
  },
  # ---------------------------------------------------------------------------------------------------
  # BPE with split silence


  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.bpe-sil-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe-with-sil-split-sil", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="bpe-sil"),
  },

  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop1e-07.new-pre.6pretrain-reps.bpe-sil-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe-with-sil-split-sil", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="bpe-sil", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0000001),
  },

  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop1e-07.new-pre.6pretrain-reps.bpe-sil-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe-with-sil-split-sil", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="bpe-sil", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0000001),
  },

  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.bpe-sil-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe-with-sil-split-sil", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="bpe-sil", label_model_focal_loss=0.0),
  },

  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.global.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.bpe-sil-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="bpe-with-sil-split-sil", lstm_dim=1024, ctx_size="inf",
                   att_type="mlp", att_area="win", att_win_size="full", time_red=[3, 2], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="bpe-sil", label_model_focal_loss=0.0),
  },

  # PHONEMES

  # "seg.phonemes-split-sil.full-ctx.time-red1.fast-rec.fast-rec-full.seg.mlp-att.am512.length-model-in_am+prev-out-embed.phonemes-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="phonemes-split-sil", lstm_dim=256,
  #                  ctx_size="inf", att_type="mlp", att_area="seg", time_red=[1], prev_att_in_state=False,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="phonemes"), },

  "seg.phonemes-split-sil.full-ctx.time-red1.fast-rec.fast-rec-full.seg.mlp-att.am1024.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.prev-target-in-readout.weight-drop1e-07.new-pre.6pretrain-reps.phonemes-segs": {
    "config": dict(default_variant_transducer, use_attention=True, label_type="phonemes-split-sil", lstm_dim=512, ctx_size="inf",
                   att_type="mlp", att_area="seg", time_red=[1], prev_att_in_state=False, length_model_focal_loss=0.0,
                   length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                   segment_selection="phonemes", label_model_focal_loss=0.0, pretrain_reps=6, pretraining="new",
                   prev_target_in_readout=True, weight_dropout=0.0000001),
  },

  # "seg.phonemes-split-sil.full-ctx.time-red1.fast-rec.fast-rec-full.seg.mlp-att.am512.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.phonemes-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="phonemes-split-sil", lstm_dim=256, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[1], prev_att_in_state=False, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="phonemes", label_model_focal_loss=0.0),
  # },
  #
  # "seg.phonemes-split-sil.ctx1.time-red1.fast-rec.fast-rec-full.seg.mlp-att.am512.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.phonemes-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="phonemes-split-sil", lstm_dim=256, ctx_size=1,
  #                  att_type="mlp", att_area="seg", time_red=[1], prev_att_in_state=False, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="phonemes", label_model_focal_loss=0.0),
  # },
  #
  # "seg.phonemes-split-sil.full-ctx.time-red1.fast-rec.fast-rec-full.seg.mlp-att.am512.prev-att-in-state.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.phonemes-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="phonemes-split-sil", lstm_dim=256, ctx_size="inf",
  #                  att_type="mlp", att_area="seg", time_red=[1], prev_att_in_state=True, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="phonemes", label_model_focal_loss=0.0),
  # },
  #
  # "seg.phonemes-split-sil.full-ctx.time-red1.fast-rec.fast-rec-full.seg.pooling-att.am512.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.phonemes-segs": {
  #   "config": dict(default_variant_transducer, use_attention=True, label_type="phonemes-split-sil", lstm_dim=256, ctx_size="inf",
  #                  att_type="pooling", att_area="seg", time_red=[1], prev_att_in_state=False, length_model_focal_loss=0.0,
  #                  length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
  #                  segment_selection="phonemes", label_model_focal_loss=0.0),
  # },


}
