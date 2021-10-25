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

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import *
from i6_core.returnn.search import *
from i6_core.recognition.scoring import Hub5ScoreJob
from sisyphus import *


class TransducerSWBBaseConfig:
  def __init__(self, vocab, target="orth_classes", target_num_labels=1030, targetb_blank_idx=0, data_dim=40,
               alignment_same_len=True,
               epoch_split=6, rasr_config="/u/schmitt/experiments/transducer/config/rasr-configs/merged.config",
               _attention_type=0):

    # data
    self.target = target
    self.target_num_labels = target_num_labels
    self.targetb_blank_idx = targetb_blank_idx
    self.task = "train"
    self.vocab = vocab
    self.rasr_config = rasr_config
    self.epoch_split = epoch_split
    self.targetb_num_labels = target_num_labels + 1
    self._cf_cache = {}
    self._alignment = None

    # network
    self._attention_type = _attention_type
    self.EncKeyTotalDim = 200
    self.AttNumHeads = 1  # must be 1 for hard-att
    self.AttentionDropout = 0.1
    self.l2 = 0.0001
    self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
    self.EncValueTotalDim = 2048
    self.EncValuePerHeadDim = self.EncValueTotalDim // self.AttNumHeads
    self.LstmDim = self.EncValueTotalDim // 2

    self.extern_data = {
      "data": {"dim": data_dim},  # Gammatone 40-dim
      "alignment": {"dim": self.targetb_num_labels, "sparse": True}}

    time_tag_str = "DimensionTag(kind=DimensionTag.Types.Spatial, description='time')"
    output_len_tag_str = "DimensionTag(kind=DimensionTag.Types.Spatial, description='output-len')"  # downsampled time
    self.extern_data_epilog = [
      "extern_data['data']['same_dim_tags_as'] = {'t': %s}" % time_tag_str,
      "extern_data['alignment']['same_dim_tags_as'] = {'t': %s}" % output_len_tag_str if alignment_same_len else "None"]

    # other options
    self.network = {}
    self.use_tensorflow = True
    if self.task == "train":
        self.beam_size = 4
    else:
        self.beam_size = 12
    self.learning_rate = 0.001
    self.min_learning_rate = self.learning_rate / 50.
    self.search_output_layer = "decision"
    self.debug_print_layer_output_template = True
    self.batching = "random"
    self.log_batch_size = True
    self.batch_size = 4000
    self.max_seqs = 200
    self.max_seq_length = {target: 75}
    self.truncation = -1
    self.cleanup_old_models = True
    self.gradient_clip = 0
    self.adam = True
    self.optimizer_epsilon = 1e-8
    self.accum_grad_multiple_step = 3
    self.stop_on_nonfinite_train_score = False
    self.tf_log_memory_usage = True
    self.gradient_noise = 0.0
    self.learning_rate_control = "newbob_multi_epoch"
    self.learning_rate_control_error_measure = "dev_error_output/output_prob"
    self.learning_rate_control_relative_error_relative_lr = True
    self.learning_rate_control_min_num_epochs_per_new_lr = 3
    self.use_learning_rate_control_always = True
    self.newbob_multi_num_epochs = 6
    self.newbob_multi_update_interval = 1
    self.newbob_learning_rate_decay = 0.7

    # prolog
    self.import_prolog = ["from returnn.tf.util.data import DimensionTag", "import os", "import numpy",
                          "from subprocess import check_output, CalledProcessError"]
    self.function_prolog = [
      summary,
      _mask,
      random_mask,
      transform,
      get_vocab_tf,
      get_vocab_sym,
      out_str,
      targetb_recomb_train,
      targetb_recomb_recog,
      get_filtered_score_op,
      get_filtered_score_cpp,
      custom_construction_algo,
      cf,
      get_dataset_dict,
      add_attention
    ]

    # epilog
    self.network_epilog = [
      "network = get_net_dict(pretrain_idx=None)",
      "add_attention(network, _attention_type)",
      "pretrain = {'copy_param_mode': 'subset', 'construction_algo': custom_construction_algo}"
    ]

    self.dataset_epilog = [
      "train = get_dataset_dict('train')",
      "dev = get_dataset_dict('cv')",
      "eval_datasets = {'devtrain': get_dataset_dict('devtrain')}"
    ]

  def get_config(self):
    config_dict = {k: v for k, v in self.__dict__.items() if not (k.endswith("_prolog") or k.endswith("_epilog"))}
    prolog = [prolog_item for k, prolog_list in self.__dict__.items() if k.endswith("_prolog") for prolog_item in
              prolog_list]
    epilog = [epilog_item for k, epilog_list in self.__dict__.items() if k.endswith("_epilog") for epilog_item in
              epilog_list]

    return ReturnnConfig(config=config_dict, python_prolog=prolog, python_epilog=epilog)

  def set_for_search(self, dataset_key):
    self.extern_data["targetb"] = {"dim": self.targetb_num_labels, "sparse": True, "available_for_inference": False}
    self.dataset_epilog += ["search_data = get_dataset_dict('%s')" % dataset_key]
    self.batch_size = 4000
    self.beam_size = 12

  def update(self, **kwargs):
    self.__dict__.update(kwargs)


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
      get_alignment_net_dict
    ]

    self.network_prolog = ["get_net_dict = get_alignment_net_dict"]


class TransducerSWBExtendedConfig(TransducerSWBBaseConfig):
  def __init__(self,
               *args,
               **kwargs):

    super().__init__(*args, **kwargs)

    self._alignment = "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap"

    self.batch_size = 10000
    self._chunk_size = 60
    self._time_red = 6
    self.chunking_epilog = ["""chunking = ({
  "data": _chunk_size * _time_red,
  "alignment": _chunk_size},{
  "data": _chunk_size * _time_red // 2,
  "alignment": _chunk_size // 2})"""]
    self.accum_grad_multiple_step = 2

    self.function_prolog += [
      switchout_target,
      targetb_linear,
      targetb_linear_out,
      targetb_search_or_fallback,
      get_extended_net_dict
    ]

    self.network_prolog = ["get_net_dict = get_extended_net_dict"]

  def set_for_search(self, dataset_key):
    super().set_for_search(dataset_key)
    self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}


def run_experiment(config: TransducerSWBBaseConfig, name, num_epochs, is_pretrain, mem_rqmt, time_rqmt):
  train_job = ReturnnTrainingJob(
    config.get_config(),
    num_epochs=num_epochs,
    log_verbosity=5,
    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    returnn_root="/u/schmitt/src/returnn",
    mem_rqmt=mem_rqmt,
    time_rqmt=time_rqmt)
  train_job.add_alias(name + "/train")
  alias = train_job.get_one_alias()
  tk.register_output(alias + "/config", train_job.out_returnn_config_file)
  tk.register_output(alias + "/models", train_job.out_model_dir)
  tk.register_output(alias + "/learning_rates", train_job.out_learning_rates)
  tk.register_output(alias + "/plot_se", train_job.out_plot_se)
  tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)

  checkpoint = train_job.out_checkpoints[num_epochs]

  if is_pretrain:
    index_path = tk.uncached_path(checkpoint.index_path)
    suffix = index_path[-len(".000.index"):]
    model_path = index_path[: -len(suffix)]
    model_path += ".pretrain" + suffix[:len(".000")]

  # remove the pretraining and load previously trained model
  config.network_epilog.pop(2)
  config.network_epilog += ["import_model_train_epoch1 = %r" % model_path]

  retrain_job = ReturnnTrainingJob(
    config.get_config(),
    num_epochs=num_epochs,
    log_verbosity=5,
    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    returnn_root="/u/schmitt/src/returnn",
    mem_rqmt=mem_rqmt,
    time_rqmt=time_rqmt)
  retrain_job.add_input(train_job.out_checkpoints[num_epochs].index_path)
  retrain_job.add_alias(name + "/retrain")
  alias = retrain_job.get_one_alias()
  tk.register_output(alias + "/config", retrain_job.out_returnn_config_file)
  tk.register_output(alias + "/models", retrain_job.out_model_dir)
  tk.register_output(alias + "/learning_rates", retrain_job.out_learning_rates)
  tk.register_output(alias + "/plot_se", retrain_job.out_plot_se)
  tk.register_output(alias + "/plot_lr", retrain_job.out_plot_lr)

  return retrain_job.out_checkpoints, config


def eval_model(config, checkpoint, stm_job, reference, name, dataset_key, mem_rqmt, time_rqmt):
  config.set_for_search(dataset_key)
  search_job = ReturnnSearchJob(
    search_data={},
    model_checkpoint=checkpoint,
    returnn_config=config.get_config(),
    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    returnn_root="/u/schmitt/src/returnn",
    mem_rqmt=mem_rqmt,
    time_rqmt=time_rqmt)

  search_job.add_alias(name + "/search_%s" % dataset_key)
  alias = search_job.get_one_alias()
  tk.register_output(alias + "/bpe_search_results", search_job.out_search_file)
  tk.register_output(alias + "/config", search_job.out_returnn_config_file)

  bpe_to_words_job = SearchBPEtoWordsJob(search_job.out_search_file)
  bpe_to_words_job.add_alias(name + "/words_%s" % dataset_key)
  alias = bpe_to_words_job.get_one_alias()
  tk.register_output(alias + "/word_search_results", bpe_to_words_job.out_word_search_results)

  ctm_job = SearchWordsToCTMJob(
    bpe_to_words_job.out_word_search_results,
    stm_job.bliss_corpus)
  ctm_job.add_alias(name + "/ctm_%s" % dataset_key)
  alias = ctm_job.get_one_alias()
  tk.register_output(alias + "/ctm_search_results", ctm_job.out_ctm_file)

  score_job = Hub5ScoreJob(
    reference, "/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm", ctm_job.out_ctm_file
  )
  score_job.add_alias(name + "/scores_%s" % dataset_key)
  alias = score_job.get_one_alias()
  tk.register_output(alias + "/score_reports", score_job.report_dir)

