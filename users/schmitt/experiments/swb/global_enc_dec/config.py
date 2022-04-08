from i6_experiments.users.schmitt.experiments.swb.global_enc_dec import network
from i6_experiments.users.schmitt.experiments.swb.dataset import *
from i6_experiments.users.schmitt.recombination import *
from i6_experiments.users.schmitt.rna import *
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.vocab import *
from i6_experiments.users.schmitt.switchout import *

from recipe.i6_core.returnn.config import ReturnnConfig, CodeWrapper


class GlobalEncoderDecoderConfig:
  def __init__(
          self, vocab, glob_model_type, target_num_labels=1030,
          epoch_split=6, beam_size=12, feature_stddev=None, dump_output=False,
          rasr_config="/u/schmitt/experiments/transducer/config/rasr-configs/merged.config",
          task="train", num_epochs=150, lstm_dim=1024, att_num_heads=1, sos_idx=0,
          train_data_opts=None, cv_data_opts=None, devtrain_data_opts=None, search_data_opts=None,
          time_red=(3, 2), pretrain=True, pretrain_reps=None, label_name="bpe", post_config={},
          weight_dropout=0.0, with_state_vector=True, with_weight_feedback=True, prev_target_in_readout=True):

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
        "dim": 40,
        "same_dim_tags_as": {"t": CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time')")}},
      label_name: {
        "dim": self.target_num_labels, "sparse": True}}

    self.batch_size = 10000 if self.task == "train" else 4000

    if glob_model_type == "new":
      custom_construction_algo = network.new_custom_construction_algo
      custom_construction_algo_str = "new_custom_construction_algo"
      get_net_dict = network.get_new_net_dict
    elif glob_model_type == "like-seg":
      custom_construction_algo = network.custom_construction_algo
      custom_construction_algo_str = "custom_construction_algo"
      get_net_dict = network.get_net_dict_like_seg_model
    elif glob_model_type == "best":
      custom_construction_algo = network.best_custom_construction_algo
      custom_construction_algo_str = "best_custom_construction_algo"
      get_net_dict = network.get_best_net_dict
    else:
      custom_construction_algo = network.custom_construction_algo
      custom_construction_algo_str = "custom_construction_algo"
      get_net_dict = network.get_net_dict

    self.import_prolog = ["from returnn.tf.util.data import DimensionTag", "import os", "import numpy as np",
                          "from subprocess import check_output, CalledProcessError"]
    self.function_prolog = [custom_construction_algo, _mask, random_mask, transform]

    if glob_model_type == "best":
      self.network = get_net_dict(
        lstm_dim=lstm_dim, att_num_heads=att_num_heads, att_key_dim=lstm_dim, beam_size=beam_size, sos_idx=sos_idx,
        feature_stddev=feature_stddev, weight_dropout=weight_dropout, with_state_vector=with_state_vector,
        with_weight_feedback=with_weight_feedback, prev_target_in_readout=prev_target_in_readout,
        target=label_name, task=task, targetb_num_labels=target_num_labels+1, dump_output=dump_output)
    else:
      self.network = get_net_dict(lstm_dim=lstm_dim, att_num_heads=att_num_heads, att_key_dim=lstm_dim,
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

    if pretrain:
      self.pretrain = {
        'copy_param_mode': 'subset',
        'construction_algo': CodeWrapper(custom_construction_algo_str)}
      if pretrain_reps is not None:
        self.pretrain["repetitions"] = pretrain_reps

  def get_config(self):
    config_dict = {k: v for k, v in self.__dict__.items() if
                   not (k.endswith("_prolog") or k.endswith("_epilog") or k == "post_config")}
    prolog = [prolog_item for k, prolog_list in self.__dict__.items() if k.endswith("_prolog") for prolog_item in
              prolog_list]
    epilog = [epilog_item for k, epilog_list in self.__dict__.items() if k.endswith("_epilog") for epilog_item in
              epilog_list]

    return ReturnnConfig(config=config_dict, post_config=self.post_config, python_prolog=prolog, python_epilog=epilog)