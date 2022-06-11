from .network import *
from .attention import *
from i6_experiments.users.schmitt.experiments.swb.dataset import *
from i6_experiments.users.schmitt.recombination import *
from i6_experiments.users.schmitt.rna import *
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.vocab import *
from i6_experiments.users.schmitt.switchout import *
from i6_experiments.users.schmitt.targetb import *

from recipe.i6_core.returnn.config import ReturnnConfig, CodeWrapper

import numpy as np

class TransducerSWBBaseConfig:
  def __init__(self, vocab,
               target="orth_classes", target_num_labels=1030, targetb_blank_idx=0, data_dim=40, beam_size=12,
               epoch_split=6, rasr_config="/u/schmitt/experiments/transducer/config/rasr-configs/merged.config",
               _attention_type=0, post_config={}, task="train", num_epochs=150,
               search_output_layer="decision", max_seqs=200):

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
        "dim": data_dim,
        "same_dim_tags_as": {"t": CodeWrapper("Dim(kind=Dim.Types.Spatial, description='time')")}},
      "alignment": {
        "dim": self.targetb_num_labels, "sparse": True,
        "same_dim_tags_as": {
          "t": CodeWrapper("Dim(kind=Dim.Types.Spatial, description='output-len')")}}}
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
    self.min_learning_rate = self.learning_rate / 50.
    self.search_output_layer = search_output_layer
    self.debug_print_layer_output_template = True
    self.batching = "random"
    self.log_batch_size = True
    self.batch_size = 4000
    self.max_seqs = max_seqs
    self.max_seq_length = {target: 75}
    self.truncation = -1
    self.gradient_clip = 0
    self.adam = True
    self.optimizer_epsilon = 1e-8
    self.accum_grad_multiple_step = 3
    self.stop_on_nonfinite_train_score = False
    self.tf_log_memory_usage = True
    self.gradient_noise = 0.0
    self.learning_rate_control = "newbob_multi_epoch"
    self.learning_rate_control_error_measure = "dev_error_output/label_prob"
    self.learning_rate_control_relative_error_relative_lr = True
    self.learning_rate_control_min_num_epochs_per_new_lr = 3
    self.use_learning_rate_control_always = True
    self.newbob_multi_num_epochs = 6
    self.newbob_multi_update_interval = 1
    self.newbob_learning_rate_decay = 0.7

    # prolog
    self.import_prolog = ["from returnn.tf.util.data import Dim", "import os", "import numpy as np",
                          "from subprocess import check_output, CalledProcessError"]

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

  def set_for_search(self, dataset_key):
    self.extern_data["targetb"] = {"dim": self.targetb_num_labels, "sparse": True, "available_for_inference": False}
    self.dataset_epilog += ["search_data = get_dataset_dict('%s')" % dataset_key]
    self.batch_size = 4000
    self.beam_size = 12

  def set_config_for_search(self, config: ReturnnConfig, dataset_key):
    config.config["extern_data"]["targetb"] = {"dim": self.targetb_num_labels, "sparse": True,
                                              "available_for_inference": False}
    # index = config.python_epilog.index("eval_datasets = {'devtrain': get_dataset_dict('devtrain')}")
    config.python_epilog += ["search_data = get_dataset_dict('%s')" % dataset_key]
    # config.python_epilog.insert(index+1, "search_data = get_dataset_dict('%s')" % dataset_key)
    config.config.update({
      "batch_size": 4000,
      "beam_size": 12
    })

  def update(self, **kwargs):
    self.__dict__.update(kwargs)

    if "EncKeyTotalDim" in kwargs:
      self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
    if "AttNumHeads" in kwargs:
      self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
      self.EncValuePerHeadDim = self.EncValueTotalDim // self.AttNumHeads


class TransducerSWBAlignmentConfig(TransducerSWBBaseConfig):
  def __init__(self, *args, **kwargs):

    super().__init__(*args, **kwargs)

    self.extern_data["align_score"] = {"shape": (1,), "dtype": "float32"}
    self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}

    self.function_prolog += [
      rna_loss,
      rna_alignment,
      rna_alignment_out,
      rna_loss_out,
      get_alignment_net_dict,
      custom_construction_algo_alignment
    ]

    self.network_prolog = [
      "get_net_dict = get_alignment_net_dict",
      "custom_construction_algo = custom_construction_algo_alignment"]

    self.network_epilog = [
      "network = get_net_dict(pretrain_idx=None)",
      "pretrain = {'copy_param_mode': 'subset', 'construction_algo': custom_construction_algo}"]


class TransducerSWBExtendedConfig(TransducerSWBBaseConfig):
  def __init__(
    self, *args, att_seg_emb_size, att_seg_use_emb, att_win_size, lstm_dim, direct_softmax,
    att_weight_feedback, att_type, att_seg_clamp_size, att_seg_left_size, att_seg_right_size, att_area,
    att_num_heads, length_model_inputs, label_smoothing, prev_att_in_state, fast_rec_full, pretrain_reps,
    length_model_type, att_ctx_with_bias, att_ctx_reg, exclude_sil_from_label_ctx,
    scheduled_sampling, use_attention, emit_extra_loss, efficient_loss, time_red, ctx_size="full",
    hybrid_hmm_like_label_model=False, att_query="lm", prev_target_in_readout, weight_dropout,
    fast_rec=False, pretrain=True, sep_sil_model=None, sil_idx=None, sos_idx=0, pretraining="old",
    network_type="default", global_length_var=None, chunk_size=60,
    train_data_opts=None, cv_data_opts=None, devtrain_data_opts=None, search_data_opts=None,
    search_use_recomb=False, feature_stddev=None, recomb_bpe_merging=True, dump_output=False,
    label_dep_length_model=False, label_dep_means=None, max_seg_len=None, length_model_focal_loss=2.0,
    label_model_focal_loss=2.0, import_model=None, learning_rates=None, length_scale=1., **kwargs):

    super().__init__(*args, **kwargs)

    self.batch_size = 10000 if self.task == "train" else 4000
    # chunk_size = 60
    self.chunking = ({
      "data": chunk_size * int(np.prod(time_red)), "alignment": chunk_size}, {
      "data": chunk_size * int(np.prod(time_red)) // 2, "alignment": chunk_size // 2})
    self.accum_grad_multiple_step = 2

    self.function_prolog = [_mask, random_mask, transform]

    if import_model is not None:
      self.load = import_model
    if learning_rates is not None:
      self.learning_rates = learning_rates

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
    else:
      assert pretraining == "new"
      custom_construction_algo_func = new_custom_construction_algo
      custom_construction_algo_str = "new_custom_construction_algo"

    self.function_prolog += [custom_construction_algo_func]
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
        label_smoothing=label_smoothing, emit_extra_loss=emit_extra_loss, emit_loss_scale=1.0,
        global_length_var=global_length_var, exclude_sil_from_label_ctx=exclude_sil_from_label_ctx,
        efficient_loss=efficient_loss, time_reduction=time_red, ctx_size=ctx_size, fast_rec=fast_rec,
        sep_sil_model=sep_sil_model, sil_idx=sil_idx, sos_idx=sos_idx, prev_target_in_readout=prev_target_in_readout,
        feature_stddev=feature_stddev, search_use_recomb=search_use_recomb, dump_output=dump_output,
        label_dep_length_model=label_dep_length_model, label_dep_means=label_dep_means, direct_softmax=direct_softmax,
        max_seg_len=max_seg_len, hybrid_hmm_like_label_model=hybrid_hmm_like_label_model, length_scale=length_scale,
        length_model_focal_loss=length_model_focal_loss, label_model_focal_loss=label_model_focal_loss)
      if use_attention:
        self.network = add_attention(
          self.network, att_seg_emb_size=att_seg_emb_size, att_seg_use_emb=att_seg_use_emb,
          att_win_size=att_win_size, task=self.task, EncValueTotalDim=lstm_dim * 2, EncValueDecFactor=1,
          EncKeyTotalDim=lstm_dim, att_weight_feedback=att_weight_feedback, att_type=att_type,
          att_seg_clamp_size=att_seg_clamp_size, att_seg_left_size=att_seg_left_size, att_ctx_reg=att_ctx_reg,
          att_seg_right_size=att_seg_right_size, att_area=att_area, AttNumHeads=att_num_heads,
          EncValuePerHeadDim=int(lstm_dim * 2 // att_num_heads), l2=0.0001, AttentionDropout=weight_dropout,
          EncKeyPerHeadDim=int(lstm_dim // att_num_heads), att_query=att_query, ctx_with_bias=att_ctx_with_bias)
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

    if pretrain:
      self.pretrain = {'copy_param_mode': 'subset', 'construction_algo': CodeWrapper(custom_construction_algo_str)}

      if pretrain_reps is not None:
        self.pretrain["repetitions"] = pretrain_reps

    if self.task == "search":
      self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}
