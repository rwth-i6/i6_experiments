from i6_experiments.users.schmitt.experiments.swb.transducer.network import custom_construction_algo, new_custom_construction_algo
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.legacy_v1.segmental.network import get_global_import_net_dict, get_extended_net_dict
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.legacy_v1.segmental.attention import add_attention
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.dataset.legacy_v1 import *
from i6_experiments.users.schmitt.recombination import *
from i6_experiments.users.schmitt.conformer_pretrain import *
from i6_experiments.users.schmitt.rna import *
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.vocab import *
from i6_experiments.users.schmitt.dynamic_lr import *

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

import numpy as np


class SegmentalSWBExtendedConfig:
  def __init__(
    self, vocab, att_seg_emb_size, att_seg_use_emb, att_win_size, lstm_dim, direct_softmax, enc_type,
    att_weight_feedback, att_type, att_seg_clamp_size, att_seg_left_size, att_seg_right_size, att_area,
    att_num_heads, length_model_inputs, label_smoothing, prev_att_in_state, fast_rec_full, pretrain_reps,
    length_model_type, att_ctx_with_bias, att_ctx_reg, exclude_sil_from_label_ctx, att_weights_kl_div_scale,
    scheduled_sampling, use_attention, emit_extra_loss, efficient_loss, time_red, ctx_size="full",
    hybrid_hmm_like_label_model=False, att_query="lm", prev_target_in_readout=False, weight_dropout=0.1,
    fast_rec=False, pretrain=True, sep_sil_model=None, sil_idx=None, sos_idx=0, pretraining="old",
    network_type="default", global_length_var=None, chunk_size=60, segment_center_window_size=None,
    train_data_opts=None, cv_data_opts=None, devtrain_data_opts=None, search_data_opts=None,
    search_use_recomb=False, feature_stddev=None, recomb_bpe_merging=True, dump_output=False,
    label_dep_length_model=False, label_dep_means=None, max_seg_len=None, length_model_focal_loss=2.0,
    label_model_focal_loss=2.0, import_model=None, learning_rates=None, length_scale=1., batch_size=10000,
    specaugment="albert", dynamic_lr=False, ctc_aux_loss=True, length_model_loss_scale=1., use_time_sync_loop=True,
    use_eos=False, use_glob_win=False, conf_use_blstm=False, conf_batch_norm=True, conf_num_blocks=12,
    use_zoneout=False, conf_dropout=0.03, conf_l2=None, behavior_version=None, nadam=False, force_eos=False,
    import_model_train_epoch1=None, set_dim_tag_correctly=True, features="gammatone",

    target="orth_classes", target_num_labels=1030, targetb_blank_idx=0, data_dim=40, beam_size=12,
    epoch_split=6, rasr_config="/u/schmitt/experiments/transducer/config/rasr-configs/merged.config",
    _attention_type=0, post_config={}, task="train", num_epochs=150, min_learning_rate=0.001/50.,
    search_output_layer="decision", max_seqs=200, gradient_clip=0, gradient_noise=0.0,
    newbob_learning_rate_decay=.7, newbob_multi_num_epochs=6, lr_measure="dev_error_output/label_prob"):

    self.post_config = post_config

    # data
    self.target = target
    self.target_num_labels = target_num_labels
    self.targetb_blank_idx = targetb_blank_idx
    self.task = task
    self.vocab = vocab
    self.rasr_config = rasr_config
    self.epoch_split = epoch_split
    self.targetb_num_labels = target_num_labels + 1
    self._cf_cache = {}
    self._alignment = None

    self.extern_data = {
      "data": {
        "dim": 40 if features == "gammatone" else 1,
        "same_dim_tags_as": {"t": CodeWrapper("Dim(kind=Dim.Types.Spatial, description='time')")}},
      "alignment": {
        "dim": self.targetb_num_labels, "sparse": True,
        "same_dim_tags_as": {
          "t": CodeWrapper("Dim(kind=Dim.Types.Spatial, description='output-len')")}}}
    if set_dim_tag_correctly:
      self.extern_data["data"]["same_dim_tags_as"]["t"] = CodeWrapper(
        "Dim(kind=Dim.Types.Spatial, description='time', dimension=None)")
      self.extern_data["alignment"]["same_dim_tags_as"]["t"] = CodeWrapper(
        "Dim(kind=Dim.Types.Spatial, description='time', dimension=None)")
    if task != "train":
      self.extern_data["targetb"] = {"dim": self.targetb_num_labels, "sparse": True,
                                     "available_for_inference": False}

    # other options
    self.network = {}
    self.use_tensorflow = True
    if self.task == "train":
      self.beam_size = 4
    else:
      self.num_epochs = num_epochs
      self.beam_size = beam_size
    self.learning_rate = 0.001
    self.min_learning_rate = min_learning_rate
    self.search_output_layer = search_output_layer
    self.debug_print_layer_output_template = True
    self.batching = "random"
    self.log_batch_size = True
    self.batch_size = 4000
    self.max_seqs = max_seqs
    self.max_seq_length = {target: 75}
    self.truncation = -1
    self.gradient_clip = gradient_clip
    if nadam:
      self.nadam = True
    else:
      self.adam = True
    self.optimizer_epsilon = 1e-8
    self.accum_grad_multiple_step = 3
    self.stop_on_nonfinite_train_score = False
    self.tf_log_memory_usage = True
    self.gradient_noise = gradient_noise
    self.learning_rate_control = "newbob_multi_epoch"
    self.learning_rate_control_error_measure = lr_measure
    self.learning_rate_control_relative_error_relative_lr = True
    self.learning_rate_control_min_num_epochs_per_new_lr = 3
    self.use_learning_rate_control_always = True
    self.newbob_multi_num_epochs = newbob_multi_num_epochs
    self.newbob_multi_update_interval = 1
    self.newbob_learning_rate_decay = newbob_learning_rate_decay

    # prolog
    self.import_prolog = ["from returnn.tf.util.data import Dim", "import os", "import numpy as np",
                          "from subprocess import check_output, CalledProcessError"]


    ##########################################################################

    self.batch_size = batch_size if self.task == "train" else 4000
    if features == "raw":
      self.batch_size *= 80
    # chunk_size = 60
    if chunk_size is not None:
      self.chunking = ({
        "data": chunk_size * int(np.prod(time_red)), "alignment": chunk_size}, {
        "data": chunk_size * int(np.prod(time_red)) // 2, "alignment": chunk_size // 2})

    if behavior_version is not None:
      self.behavior_version = behavior_version
      self.optimizer = {"class": "nadam" if nadam else "adam"}

    self.accum_grad_multiple_step = 2

    self.function_prolog = [_mask, random_mask, transform if specaugment == "albert" else transform_wei]
    if dynamic_lr:
      self.function_prolog += [dynamic_learning_rate]

    if import_model is not None:
      self.load = import_model
    if import_model_train_epoch1 is not None:
      self.import_model_train_epoch1 = import_model_train_epoch1
    if type(learning_rates) == list:
      self.learning_rates = learning_rates
    elif type(learning_rates) == str:
      if learning_rates == "repeat_per_pretrain":
        self.learning_rates = list(np.linspace(self.learning_rate / pretrain_reps, self.learning_rate, num=pretrain_reps)) * 4 + list(np.linspace(self.learning_rate / 20, self.learning_rate, num=20))

    if self.task == "search":
      if recomb_bpe_merging:
        self.function_prolog += [get_vocab_tf]

      else:
        self.function_prolog += [
          get_vocab_tf_no_bpe_merging,
          "get_vocab_tf = get_vocab_tf_no_bpe_merging"]

      self.function_prolog += [
        get_vocab_sym,
        out_str,
        targetb_recomb_recog,
        get_filtered_score_op,
        get_filtered_score_cpp,
      ]

    if pretraining == "old":
      custom_construction_algo_func = custom_construction_algo
      custom_construction_algo_str = "custom_construction_algo"
    elif pretraining == "mohammad-conf":
      custom_construction_algo_func = custom_construction_algo_mohammad_conf
      custom_construction_algo_str = "custom_construction_algo_mohammad_conf"
    else:
      assert pretraining == "new"
      custom_construction_algo_func = new_custom_construction_algo
      custom_construction_algo_str = "new_custom_construction_algo"

    self.function_prolog += [custom_construction_algo_func]

    if self.task == "train" and enc_type == "conf-mohammad-11-7":
      self.function_prolog += [speed_pert]
    # if self.task == "train":
    #   self.function_prolog += [
    #     switchout_target,
    #   ]
    if network_type == "default":
      self.network = get_extended_net_dict(
        pretrain_idx=None, learning_rate=self.learning_rate, num_epochs=150, length_model_type=length_model_type,
        enc_val_dec_factor=1, target_num_labels=self.target_num_labels, target=self.target, task=self.task,
        targetb_num_labels=self.targetb_num_labels, scheduled_sampling=scheduled_sampling, lstm_dim=lstm_dim, l2=0.0001,
        beam_size=self.beam_size, length_model_inputs=length_model_inputs, prev_att_in_state=prev_att_in_state,
        targetb_blank_idx=self.targetb_blank_idx, use_att=use_attention, fast_rec_full=fast_rec_full,
        label_smoothing=label_smoothing, emit_extra_loss=emit_extra_loss, emit_loss_scale=1.0, enc_type=enc_type,
        global_length_var=global_length_var, exclude_sil_from_label_ctx=exclude_sil_from_label_ctx,
        efficient_loss=efficient_loss, time_reduction=time_red, ctx_size=ctx_size, fast_rec=fast_rec,
        sep_sil_model=sep_sil_model, sil_idx=sil_idx, sos_idx=sos_idx, prev_target_in_readout=prev_target_in_readout,
        feature_stddev=feature_stddev, search_use_recomb=search_use_recomb, dump_output=dump_output,
        label_dep_length_model=label_dep_length_model, label_dep_means=label_dep_means, direct_softmax=direct_softmax,
        max_seg_len=max_seg_len, hybrid_hmm_like_label_model=hybrid_hmm_like_label_model, length_scale=length_scale,
        length_model_focal_loss=length_model_focal_loss, label_model_focal_loss=label_model_focal_loss,
        specaugment=specaugment, ctc_aux_loss=ctc_aux_loss, length_model_loss_scale=length_model_loss_scale,
        use_time_sync_loop=use_time_sync_loop, use_eos=use_eos, conf_use_blstm=conf_use_blstm,
        conf_batch_norm=conf_batch_norm, conf_num_blocks=conf_num_blocks, use_zoneout=use_zoneout,
        conf_dropout=conf_dropout, conf_l2=conf_l2, behavior_version=behavior_version, force_eos=force_eos,
        att_weights_kl_div_scale=att_weights_kl_div_scale, set_dim_tag_correctly=set_dim_tag_correctly)
      if use_attention:
        self.network = add_attention(
          self.network, att_seg_emb_size=att_seg_emb_size, att_seg_use_emb=att_seg_use_emb,
          att_win_size=att_win_size, task=self.task, EncValueTotalDim=lstm_dim * 2, EncValueDecFactor=1,
          EncKeyTotalDim=lstm_dim, att_weight_feedback=att_weight_feedback, att_type=att_type,
          att_seg_clamp_size=att_seg_clamp_size, att_seg_left_size=att_seg_left_size, att_ctx_reg=att_ctx_reg,
          att_seg_right_size=att_seg_right_size, att_area=att_area, AttNumHeads=att_num_heads,
          EncValuePerHeadDim=int(lstm_dim * 2 // att_num_heads), l2=0.0001, AttentionDropout=weight_dropout,
          EncKeyPerHeadDim=int(lstm_dim // att_num_heads), att_query=att_query, ctx_with_bias=att_ctx_with_bias,
          segment_center_window_size=segment_center_window_size, use_time_sync_loop=use_time_sync_loop,
          use_glob_win=use_glob_win, behavior_version=behavior_version, set_dim_tag_correctly=set_dim_tag_correctly)
    else:
      assert network_type in ["global_import", "global_import_w_feedback", "global_import_wo_feedback_wo_state_vector"]
      if network_type in [
        "global_import", "global_import_wo_feedback_wo_state_vector"
      ]:
        weight_feedback = False
      else:
        weight_feedback = True
      self.network = get_global_import_net_dict(
        task=self.task, weight_feedback=weight_feedback, feature_stddev=feature_stddev,
        prev_att_in_state=False if network_type == "global_import_wo_feedback_wo_state_vector" else True,
        targetb_num_labels=self.targetb_num_labels, target_num_labels=self.target_num_labels,
        targetb_blank_index=self.targetb_blank_idx, sil_idx=sil_idx, sos_idx=sos_idx)

    if self.task == "train":
      assert train_data_opts and cv_data_opts and devtrain_data_opts
      self.train = get_dataset_dict_w_alignment(**train_data_opts)
      self.dev = get_dataset_dict_w_alignment(**cv_data_opts)
      self.eval_datasets = {'devtrain': get_dataset_dict_w_alignment(**devtrain_data_opts)}
    elif self.task == "search":
      assert search_data_opts
      self.search_data = get_dataset_dict_wo_alignment(**search_data_opts)

    if pretrain and import_model_train_epoch1 is None:
      self.pretrain = {'copy_param_mode': 'subset', 'construction_algo': CodeWrapper(custom_construction_algo_str)}

      if pretrain_reps is not None:
        self.pretrain["repetitions"] = pretrain_reps

    if self.task == "search":
      self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}

    if enc_type == "conf-wei" or enc_type == "conf-mohammad-11-7":
      self.import_prolog += ["import sys", "sys.setrecursionlimit(4000)"]

  def get_config(self):
    config_dict = {k: v for k, v in self.__dict__.items() if
                   not (k.endswith("_prolog") or k.endswith("_epilog") or k == "post_config")}
    prolog = [prolog_item for k, prolog_list in self.__dict__.items() if k.endswith("_prolog") for prolog_item in
              prolog_list]
    epilog = [epilog_item for k, epilog_list in self.__dict__.items() if k.endswith("_epilog") for epilog_item in
              epilog_list]
    # print(epilog)
    post_config = self.post_config

    return ReturnnConfig(config=config_dict, post_config=post_config, python_prolog=prolog, python_epilog=epilog)
