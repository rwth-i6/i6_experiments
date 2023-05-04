import copy
import numpy as np

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT, RETURNN_CURRENT_ROOT


def build_alias(**kwargs):
  alias = ""
  alias += kwargs["model_type"]

  if kwargs["model_type"] == "seg":
    assert kwargs.keys() == default_variant_transducer.keys()
    kwargs = copy.deepcopy(kwargs)
    kwargs = {k.replace("_", "-"): v for k, v in kwargs.items()}

    if not kwargs["do-chunk-fix"]:
      alias += ".no-chunk-fix"

    if kwargs["segment-center-window-size"] is not None:
      alias += ".win-size%s" % (int(kwargs["segment-center-window-size"] * 2) if not kwargs["use-glob-win"] else "-glob")

    if not kwargs["use-time-sync-loop"]:
      alias += ".no-time-loop"
    if kwargs["use-eos"]:
      alias += ".use-eos"

    if kwargs["features"] != "gammatone":
      alias += ".%s-features" % kwargs["features"]
    alias += "." + kwargs["label-type"]
    if kwargs["epoch-split"] != 6:
      alias += ".ep-split-%d" % kwargs["epoch-split"]
    if kwargs["lr-measure"] != "dev_error_output/label_prob":
      if kwargs["lr-measure"] == "dev_error_label_model/label_prob":
        alias += ".lr-meas-lab"

    # if kwargs["enc-type"] == "lstm":
    if kwargs["min-learning-rate"] != (0.001 / 50.):
      alias += ".mlr-%s" % kwargs["min-learning-rate"]
    if kwargs["gradient-clip"] != 0:
      alias += ".gc-%s" % kwargs["gradient-clip"]
    if kwargs["gradient-noise"] != 0.0:
      alias += ".gn-%s" % kwargs["gradient-noise"]
    if kwargs["nadam"]:
      alias += ".nadam"
    if kwargs["batch-size"] != 10e3:
      alias += ".bs-%s" % kwargs["batch-size"]
    if kwargs["specaugment"] != "albert":
      alias += ".sa-%s" % kwargs["specaugment"]
    if kwargs["dynamic-lr"]:
      alias += ".d-lr"
    if kwargs["newbob-learning-rate-decay"] != 0.7:
      alias += ".nb-lrd-%s" % kwargs["newbob-learning-rate-decay"]
    # else:
    #   if kwargs["min-learning-rate"] != 1e-06:
    #     alias += ".mlr-%s" % kwargs["min-learning-rate"]
    #   if kwargs["gradient-clip"] != 20.:
    #     alias += ".gc-%s" % kwargs["gradient-clip"]
    #   if kwargs["gradient-noise"] != 0.1:
    #     alias += ".gn-%s" % kwargs["gradient-noise"]
    #   if not kwargs["nadam"]:
    #     alias += ".adam"
    #   if kwargs["batch-size"] != 15000:
    #     alias += ".bs-%s" % kwargs["batch-size"]
    #   if kwargs["specaugment"] != "wei":
    #     alias += ".sa-%s" % kwargs["specaugment"]
    #   if not kwargs["dynamic-lr"]:
    #     alias += ".static-lr"
    #   if kwargs["newbob-learning-rate-decay"] != 0.9:
    #     alias += ".nb-lrd-%s" % kwargs["newbob-learning-rate-decay"]
    #   if kwargs["conf-dropout"] != 0.03:
    #     alias += ".drop-%s" % kwargs["conf-dropout"]

    if kwargs["learning-rates"] is not None:
      if type(kwargs["learning-rates"]) == list:
        raise NotImplementedError
      elif type(kwargs["learning-rates"]) == str:
        if kwargs["learning-rates"] == "repeat_per_pretrain":
          alias += ".lr-warm1"
    if kwargs["use-zoneout"]:
      alias += ".zoneout"
    if kwargs["max-seqs"] != 200:
      alias += ".max-seqs-%d" % kwargs["max-seqs"]
    if kwargs["newbob-multi-num-epochs"] != 6:
      alias += ".nb-mne-%s" % kwargs["newbob-multi-num-epochs"]
    if not kwargs["ctc-aux-loss"]:
      alias += ".no-ctc"
    if kwargs["ctx-size"] == "inf":
      alias += ".full-ctx"
    elif type(kwargs["ctx-size"]) == int:
      alias += ".ctx%d" % kwargs["ctx-size"]
    else:
      raise NotImplementedError
    alias += ".time-red%d" % int(np.prod(kwargs["time-red"]))
    if kwargs["enc-type"] != "lstm":
      alias += ".%s" % kwargs["enc-type"]
      if kwargs["conf-num-blocks"] != 12:
        alias += "-%d-blocks" % kwargs["conf-num-blocks"]
    if kwargs["conf-use-blstm"]:
      alias += ".bi-lstm"
    if not kwargs["conf-batch-norm"]:
      alias += ".ln"
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
      if kwargs["att-weights-kl-div-scale"] != 0.:
        alias += ".kl-div-%0.2f" % kwargs["att-weights-kl-div-scale"]
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
    if kwargs["length-model-loss-scale"] != 1.:
      alias += ".length-loss-scale-%s" % kwargs["length-model-loss-scale"]

    if "readout-inputs" in kwargs:
      alias += ".readout-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["readout-inputs"]]).replace("base:", "")

    if "emit-prob-inputs" in kwargs:
      alias += ".emit-prob-in_" + "+".join([inp.replace("_", "-") for inp in kwargs["emit-prob-inputs"]])

    if kwargs["length-model-focal-loss"] != 0.0:
      alias += ".length-focal-%s" % kwargs["length-model-focal-loss"]
    if kwargs["label-model-focal-loss"] != 0.0:
      alias += ".label-focal-%s" % kwargs["label-model-focal-loss"]

    if kwargs["prev-target-in-readout"]:
      alias += ".targ-in-read"
    if kwargs["weight-dropout"] != 0.1:
      alias += ".weight-drop%s" % kwargs["weight-dropout"]
    if kwargs["pretrain"]:
      if kwargs["pretraining"] != "old":
        alias += ".%s-pre" % kwargs["pretraining"]
      if kwargs["pretrain-reps"] is not None:
        alias += ".%spretrain-reps" % kwargs["pretrain-reps"]
    else:
      alias += ".no-pt"
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
    if kwargs["config-builder"] != "old":
      alias += ".%s-config-builder" % kwargs["config-builder"]
    alias += ".%s-model" % kwargs["glob-model-type"]
    if kwargs["features"] != "gammatone":
      alias += ".%s-features" % kwargs["features"]
    alias += "." + kwargs["label-type"]
    if kwargs["epoch-split"] != 6:
      alias += ".ep-split-%d" % kwargs["epoch-split"]
    alias += ".time-red%d" % int(np.prod(kwargs["time-red"]))
    alias += ".am%d" % (kwargs["lstm-dim"] * 2)
    # alias += ".att-num-heads%d" % kwargs["att-num-heads"] * 2
    if kwargs["pretrain-reps"] is not None:
      alias += ".%spretrain-reps" % kwargs["pretrain-reps"]
    if kwargs["enc-type"] != "lstm":
      alias += "." + kwargs["enc-type"]
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

  return alias


default_variant_transducer = dict(
  model_type="seg", att_type="dot", att_area="win", att_win_size=5, att_seg_left_size=None, att_seg_right_size=None,
  att_seg_clamp_size=None, att_seg_use_emb=False, att_num_heads=1, att_seg_emb_size=2, att_weight_feedback=False,
  label_smoothing=None, scheduled_sampling=False, prev_att_in_state=True, pretraining="old", att_ctx_reg=True,
  length_model_inputs=["prev_non_blank_embed", "prev_out_is_non_blank", "am"], use_attention=True,
  label_type="phonemes", lstm_dim=1024, efficient_loss=False, emit_extra_loss=None, ctx_size="inf", time_red=[1],
  fast_rec=False, sep_sil_model=None, fast_rec_full=False, length_model_type="frame",
  hybrid_hmm_like_label_model=False, att_query="lm", length_model_focal_loss=2.0, label_model_focal_loss=2.0,
  prev_target_in_readout=False, weight_dropout=0.1, pretrain_reps=None, direct_softmax=False, att_ctx_with_bias=False,
  exclude_sil_from_label_ctx=False, max_seg_len=None, chunk_size=60, enc_type="lstm", epoch_split=6,
  segment_center_window_size=None, pretrain=True,
  min_learning_rate=0.001/50., gradient_clip=0, newbob_learning_rate_decay=0.7, newbob_multi_num_epochs=6,
  gradient_noise=0.0, nadam=False, batch_size=10000, specaugment="albert", dynamic_lr=False, ctc_aux_loss=True,
  length_model_loss_scale=1., use_time_sync_loop=True, use_eos=False, use_glob_win=False, use_zoneout=False,
  lr_measure="dev_error_output/label_prob", conf_use_blstm=False, conf_batch_norm=True, max_seqs=200, conf_num_blocks=12,
  learning_rates=None, conf_l2=None, conf_dropout=0.03, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT,
  att_weights_kl_div_scale=0., do_chunk_fix=True, set_dim_tag_correctly=True, features="gammatone"
)

default_variant_enc_dec = dict(
  model_type="glob", att_num_heads=1, lstm_dim=1024, label_type="bpe", time_red=[3, 2],
  glob_model_type="old", pretrain_reps=None, weight_dropout=0.0, with_state_vector=True, att_ctx_reg=False,
  with_weight_feedback=True, prev_target_in_readout=True, use_l2=True, att_ctx_with_bias=False, focal_loss=0.0,
  pretrain_type="best", epoch_split=6, enc_type="lstm", pretrain=True, config_builder="old",
  returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT, set_dim_tag_correctly=True, features="gammatone")

models = {
  "segmental": {
    "seg.rna-bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
      "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe", lstm_dim=1024, ctx_size="inf",
                     att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                     length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                     label_model_focal_loss=0.0, att_ctx_reg=False, lr_measure="dev_error_label_model/label_prob",
                     set_dim_tag_correctly=False),
    },
  },
  "segmental_conformer": {
    "seg.raw-features.rna-bpe.lr-meas-lab.full-ctx.time-red6.conf-mohammad-11-7.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
      "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe", lstm_dim=1024, ctx_size="inf",
                     att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                     length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                     label_model_focal_loss=0.0, att_ctx_reg=False, lr_measure="dev_error_label_model/label_prob",
                     enc_type="conf-mohammad-11-7", set_dim_tag_correctly=True, returnn_root=RETURNN_CURRENT_ROOT,
                     features="raw"),
    },
  },
  "segmental_wo_chunk_fix": {
    "seg.no-chunk-fix.rna-bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
      "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe", lstm_dim=1024, ctx_size="inf",
                     att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                     length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True, do_chunk_fix=False,
                     label_model_focal_loss=0.0, att_ctx_reg=False, lr_measure="dev_error_label_model/label_prob",
                     set_dim_tag_correctly=False),
    },
  },
  "segmental_w_kl_div": {
    "seg.rna-bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.kl-div-1.00.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
      "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe", lstm_dim=1024, ctx_size="inf",
                     att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                     length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                     label_model_focal_loss=0.0, att_ctx_reg=False, lr_measure="dev_error_label_model/label_prob",
                     att_weights_kl_div_scale=1., set_dim_tag_correctly=False),
    },
  },
  "segmental_w_silence": {
    "seg.rna-bpe-sil-time-red-6.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
      "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe-sil-time-red-6", lstm_dim=1024, ctx_size="inf",
                     att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                     length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                     label_model_focal_loss=0.0, att_ctx_reg=False, lr_measure="dev_error_label_model/label_prob",
                     set_dim_tag_correctly=False),
    },
  },
  "segmental_w_split_silence": {
    "seg.rna-bpe-split-sil-time-red-6.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
      "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe-split-sil-time-red-6", lstm_dim=1024, ctx_size="inf",
                     att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                     length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                     label_model_focal_loss=0.0, att_ctx_reg=False, lr_measure="dev_error_label_model/label_prob",
                     set_dim_tag_correctly=False),
    },
  },
  "center-window": {
    "seg.win-size8.rna-bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
        "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe", lstm_dim=1024, ctx_size="inf",
                       att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                       length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True,
                       label_model_focal_loss=0.0, att_ctx_reg=False, segment_center_window_size=4,
                       lr_measure="dev_error_label_model/label_prob", set_dim_tag_correctly=False),
      },
  },
  "center-window_wo_chunk_fix": {
    "seg.no-chunk-fix.win-size8.rna-bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg": {
        "config": dict(default_variant_transducer, use_attention=True, label_type="rna-bpe", lstm_dim=1024, ctx_size="inf",
                       att_type="mlp", att_area="seg", time_red=[3, 2], prev_att_in_state=True, length_model_focal_loss=0.0,
                       length_model_inputs=["am", "prev_out_embed"], fast_rec=True, fast_rec_full=True, do_chunk_fix=False,
                       label_model_focal_loss=0.0, att_ctx_reg=False, segment_center_window_size=4,
                       lr_measure="dev_error_label_model/label_prob", set_dim_tag_correctly=False),
      },
  },
  "global": {
    "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg": {
      "config": dict(
        default_variant_enc_dec, glob_model_type="best", pretrain_reps=1, set_dim_tag_correctly=False,
        with_weight_feedback=False, pretrain_type="like-seg", att_ctx_with_bias=True, use_l2=True),
      },
    "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.no-prev-target-in-readout.pretrain-like-seg": {
      "config": dict(
        default_variant_enc_dec, glob_model_type="best", pretrain_reps=1, set_dim_tag_correctly=False,
        with_weight_feedback=False, pretrain_type="like-seg", att_ctx_with_bias=False, use_l2=True,
        prev_target_in_readout=False),
    },
  },
  "global_conformer": {
    "glob.best-model.raw-features.bpe.time-red6.am2048.1pretrain-reps.conf-mohammad-11-7.no-weight-feedback.no-prev-target-in-readout.pretrain-like-seg": {
      "config": dict(
        default_variant_enc_dec, glob_model_type="best", pretrain_reps=1,
        with_weight_feedback=False, pretrain_type="like-seg", att_ctx_with_bias=False, use_l2=True,
        prev_target_in_readout=False, enc_type="conf-mohammad-11-7", set_dim_tag_correctly=True,
        returnn_root=RETURNN_CURRENT_ROOT, features="raw"),
    },
  }
}

global_model_variants = {}
for model_type in models:
  if model_type in ["global"]:
    global_model_variants[model_type] = models[model_type]

segmental_model_variants = {}
for model_type in models:
  if model_type in [
    "segmental", "segmental_w_silence", "segmental_w_split_silence", "center-window"]:
    segmental_model_variants[model_type] = models[model_type]

segmental_model_variants_w_length_model = {}
for model_type in models:
  if model_type in ["segmental", "segmental_w_silence", "segmental_w_split_silence", "center-window"]:
    segmental_model_variants_w_length_model[model_type] = models[model_type]
