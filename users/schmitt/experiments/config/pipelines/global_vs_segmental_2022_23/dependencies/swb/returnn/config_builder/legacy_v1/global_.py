# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.swb.global_enc_dec import network
from i6_experiments.users.schmitt.experiments.swb.global_enc_dec.network import custom_construction_algo, new_custom_construction_algo, best_custom_construction_algo
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.legacy_v1.global_ import get_best_net_dict, get_new_net_dict, get_net_dict, get_net_dict_like_seg_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.dataset.legacy_v1 import *
# from i6_experiments.users.schmitt.experiments.swb.dataset import *
from i6_experiments.users.schmitt.conformer_pretrain import *
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask

from i6_core.returnn.config import ReturnnConfig, CodeWrapper


class GlobalEncoderDecoderConfig:
  def __init__(
          self, vocab, glob_model_type, target_num_labels=1030, import_model=None,
          epoch_split=6, beam_size=12, feature_stddev=None, dump_output=False, enc_type="lstm",
          rasr_config="/u/schmitt/experiments/transducer/config/rasr-configs/merged.config",
          task="train", num_epochs=150, lstm_dim=1024, att_num_heads=1, sos_idx=0, sil_idx=None,
          train_data_opts=None, cv_data_opts=None, devtrain_data_opts=None, search_data_opts=None,
          time_red=(3, 2), pretrain=True, pretrain_reps=None, label_name="bpe", post_config={},
          weight_dropout=0.0, with_state_vector=True, with_weight_feedback=True, prev_target_in_readout=True,
          use_l2=True, att_ctx_with_bias=False, focal_loss=0.0, pretrain_type="best", att_ctx_reg=False,
          import_model_train_epoch1=None, set_dim_tag_correctly=True, features="gammatone"):

    self.post_config = post_config

    self.target_num_labels = target_num_labels
    self.task = task
    self.use_tensorflow = True
    self.beam_size = beam_size
    self.vocab = vocab
    self.rasr_config = rasr_config
    self.epoch_split = epoch_split
    if glob_model_type != "like-seg":
      self.cache_size = "0"
    self.search_output_layer = "decision"
    self.debug_print_layer_output_template = True
    self.batching = "random"
    self.log_batch_size = True
    self.max_seqs = 200
    self.max_seq_length = {label_name: 75}
    self.truncation = -1

    if import_model is not None:
      self.load = import_model
    if import_model_train_epoch1 is not None:
      self.import_model_train_epoch1 = import_model_train_epoch1

    self.gradient_clip = 0
    self.adam = True
    self.optimizer_epsilon = 1e-8
    self.accum_grad_multiple_step = 2
    self.stop_on_nonfinite_train_score = False
    self.tf_log_memory_usage = True
    self.gradient_noise = 0.0
    self.learning_rate = 0.001
    self.min_learning_rate = self.learning_rate / 50. if glob_model_type != "like-seg" else 2e-05
    self.learning_rate_control = "newbob_multi_epoch"
    self.learning_rate_control_error_measure = "dev_error_output/output_prob"
    self.learning_rate_control_relative_error_relative_lr = True
    self.learning_rate_control_min_num_epochs_per_new_lr = 3
    self.use_learning_rate_control_always = True
    self.newbob_multi_num_epochs = 6
    self.newbob_multi_update_interval = 1
    self.newbob_learning_rate_decay = 0.7

    if task == "search":
      self.num_epochs = num_epochs

    self.extern_data = {
      "data": {
        "dim": 40 if features == "gammatone" else 1,
        "same_dim_tags_as": {"t": CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time')")}},
      label_name: {
        "dim": self.target_num_labels, "sparse": True}}
    if set_dim_tag_correctly:
      self.extern_data["data"]["same_dim_tags_as"]["t"] = CodeWrapper(
        "DimensionTag(kind=DimensionTag.Types.Spatial, description='time', dimension=None)")

    self.batch_size = 10000 if self.task == "train" else 4000
    if features == "raw":
      self.batch_size *= 80

    if glob_model_type == "new":
      get_net_dict_func = get_new_net_dict
    elif glob_model_type == "like-seg":
      get_net_dict_func = get_net_dict_like_seg_model
    elif glob_model_type == "best":
      get_net_dict_func = get_best_net_dict
    else:
      get_net_dict_func = get_net_dict

    if pretrain_type == "new":
      custom_construction_algo_str = "new_custom_construction_algo"
      custom_construction_algo_func = new_custom_construction_algo
    elif pretrain_type == "like-seg":
      custom_construction_algo_str = "custom_construction_algo"
      custom_construction_algo_func = custom_construction_algo
    elif pretrain_type == "mohammad-conf":
      custom_construction_algo_func = custom_construction_algo_mohammad_conf
      custom_construction_algo_str = "custom_construction_algo_mohammad_conf"
    else:
      assert pretrain_type == "best"
      custom_construction_algo_func = best_custom_construction_algo
      custom_construction_algo_str = "best_custom_construction_algo"

    self.import_prolog = ["from returnn.tf.util.data import DimensionTag", "import os", "import numpy as np",
                          "from subprocess import check_output, CalledProcessError"]
    self.function_prolog = [custom_construction_algo_func, _mask, random_mask, transform]

    if self.task == "train" and enc_type == "conf-mohammad-11-7":
      self.function_prolog += [speed_pert]

    if glob_model_type == "best":
      self.network = get_net_dict_func(
        lstm_dim=lstm_dim, att_num_heads=att_num_heads, att_key_dim=lstm_dim, beam_size=beam_size, sos_idx=sos_idx,
        feature_stddev=feature_stddev, weight_dropout=weight_dropout, with_state_vector=with_state_vector,
        with_weight_feedback=with_weight_feedback, prev_target_in_readout=prev_target_in_readout, sil_idx=sil_idx,
        target=label_name, task=task, targetb_num_labels=target_num_labels+1, dump_output=dump_output,
        use_l2=use_l2, att_ctx_with_bias=att_ctx_with_bias, focal_loss=focal_loss, att_ctx_reg=att_ctx_reg,
        enc_type=enc_type)
    else:
      self.network = get_net_dict_func(lstm_dim=lstm_dim, att_num_heads=att_num_heads, att_key_dim=lstm_dim,
        beam_size=beam_size, sos_idx=sos_idx, time_red=time_red, l2=0.0001, learning_rate=self.learning_rate,
        feature_stddev=feature_stddev, target=label_name, task=task)

    if self.task == "train":
      assert train_data_opts and cv_data_opts and devtrain_data_opts
      if "label_hdf" not in train_data_opts:
        # in this case, the labels are added via the ExternSprintDataset vocab option
        get_dataset_dict_func = get_dataset_dict_wo_alignment
      else:
        # in this case, the labels are added via an external hdf file
        get_dataset_dict_func = get_dataset_dict_w_labels
      self.train = get_dataset_dict_func(**train_data_opts)
      self.dev = get_dataset_dict_func(**cv_data_opts)
      self.eval_datasets = {'devtrain': get_dataset_dict_func(**devtrain_data_opts)}
    elif self.task == "search":
      assert search_data_opts and label_name == "bpe"
      self.search_data = get_dataset_dict_wo_alignment(**search_data_opts)

    if pretrain and import_model_train_epoch1 is None:
      self.pretrain = {
        'copy_param_mode': 'subset',
        'construction_algo': CodeWrapper(custom_construction_algo_str)}
      if pretrain_reps is not None:
        self.pretrain["repetitions"] = pretrain_reps

    if enc_type == "conf-wei" or enc_type == "conf-mohammad-11-7":
      self.import_prolog += ["import sys", "sys.setrecursionlimit(4000)"]

  def get_config(self):
    config_dict = {k: v for k, v in self.__dict__.items() if
                   not (k.endswith("_prolog") or k.endswith("_epilog") or k == "post_config")}
    prolog = [prolog_item for k, prolog_list in self.__dict__.items() if k.endswith("_prolog") for prolog_item in
              prolog_list]
    epilog = [epilog_item for k, epilog_list in self.__dict__.items() if k.endswith("_epilog") for epilog_item in
              epilog_list]

    return ReturnnConfig(config=config_dict, post_config=self.post_config, python_prolog=prolog, python_epilog=epilog)