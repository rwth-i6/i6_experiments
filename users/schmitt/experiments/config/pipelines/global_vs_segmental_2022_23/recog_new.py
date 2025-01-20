from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from i6_core.returnn.search import ReturnnSearchJobV2, SearchWordsToCTMJob, SearchBPEtoWordsJob, SearchTakeBestJob, SearchOutputRawReplaceJob
from i6_core.recognition.scoring import Hub5ScoreJob, ScliteJob
from i6_core.returnn.training import Checkpoint, PtCheckpoint
from i6_core.returnn.compile import CompileTFGraphJob
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.rasr.config import RasrConfig
from i6_core.corpus.segments import SplitSegmentFileJob
from i6_core.features.common import samples_flow
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_core.corpus.convert import CorpusToStmJob

from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2
from i6_experiments.users.schmitt.flow import get_raw_wav_feature_flow
from i6_experiments.users.schmitt.scoring import ProcessConcatStmAndCtm
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF, GlobalAttConfigBuilderRF, SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import GlobalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBCorpus
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.search_errors import calc_search_errors
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.att_weights import dump_att_weights, dump_ctc_probs
import i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.analysis as analysis_rf
from i6_experiments.users.schmitt.corpus.concat.convert import WordsToCTMJobV2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_CURRENT_ROOT, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables, RasrExecutablesNew
from i6_experiments.users.schmitt.rasr.recognition import RASRDecodingJobParallel, RASRDecodingStatisticsJob
from i6_experiments.users.schmitt.rasr.convert import RASRLatticeToCTMJob, ConvertCTMBPEToWordsJob
from i6_experiments.users.schmitt.alignment.alignment import AlignmentRemoveAllBlankSeqsJob, AlignmentStatisticsJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment, SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.realignment_new import RasrRealignmentExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora import tedlium2, librispeech
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import _returnn_v2_train_step, from_scratch_training


class DecodingExperiment(ABC):
  def __init__(
          self,
          alias: str,
          config_builder: Union[ConfigBuilder, ConfigBuilderRF],
          checkpoint: Union[Checkpoint, PtCheckpoint, Dict],
          checkpoint_alias: str,
          recog_opts: Optional[Dict] = None,
  ):
    self.config_builder = config_builder
    self.checkpoint_alias = checkpoint_alias
    if isinstance(checkpoint, Checkpoint) or isinstance(checkpoint, PtCheckpoint):
      self.checkpoint = checkpoint
    else:
      assert isinstance(checkpoint, dict)
      self.checkpoint = config_builder.get_recog_checkpoints(**checkpoint)[checkpoint_alias]

    self.recog_opts = copy.deepcopy(self.default_recog_opts)
    if recog_opts is not None:
      self.recog_opts.update(recog_opts)

    self.corpus_key = self.recog_opts["dataset_opts"]["corpus_key"]
    self.stm_corpus_key = self.corpus_key

    self.alias = alias

    self.returnn_python_exe = self.config_builder.variant_params["returnn_python_exe"]
    self.returnn_root = self.config_builder.variant_params["returnn_root"]

    self.ilm_correction_opts = self.recog_opts.get("ilm_correction_opts")
    if self.ilm_correction_opts is not None and self.ilm_correction_opts["type"] == "mini_att":
      if "mini_att_checkpoint" not in self.ilm_correction_opts:
        assert "mini_att_checkpoint" not in self.ilm_correction_opts, (
          "mini_att_checkpoint is expected to be set by get_mini_att_checkpoint"
        )
        train_mini_lstm_opts = {
          "use_se_loss": self.ilm_correction_opts["use_se_loss"],
          "get_global_att_config_builder_rf_func": self.ilm_correction_opts.get("get_global_att_config_builder_rf_func"),
        }
        self.ilm_correction_opts["mini_att_checkpoint"] = self.get_mini_att_checkpoint(
          train_mini_lstm_opts=train_mini_lstm_opts
        )

    self.score_job = None
    self.analyze_gradients_job = None

  def get_ilm_correction_alias(self, alias: str):
    if self.ilm_correction_opts is not None:
      alias += "/ilm_correction_scale-%f" % self.ilm_correction_opts["scale"]
      if self.ilm_correction_opts.get("type", "mini_att") == "mini_att":
        alias += "/mini_att"
        if self.ilm_correction_opts.get("use_se_loss", False):
          alias += "/w_se_loss"
        else:
          alias += "/wo_se_loss"
        if self.ilm_correction_opts.get("mini_att_train_num_epochs", None):
          alias += "/mini_att_train_num_epochs-%d" % self.ilm_correction_opts["mini_att_train_num_epochs"]
      elif self.ilm_correction_opts["type"] == "zero_att":
        alias += "/zero_att"

      alias += f"/{'wo_' if not self.ilm_correction_opts.get('correct_eos', True) else ''}correct-eos"
      # if "correct_eos" in self.ilm_correction_opts:
      #   if self.ilm_correction_opts["correct_eos"]:
      #     alias += "/correct_eos"
      #   else:
      #     alias += "/wo_correct_eos"
    else:
      alias += "/wo_ilm_correction"

    return alias

  @property
  @abstractmethod
  def default_recog_opts(self) -> Dict:
    pass

  @property
  @abstractmethod
  def default_analysis_opts(self) -> Dict:
    pass

  @abstractmethod
  def get_mini_att_checkpoint(self, train_mini_lstm_opts: Dict) -> Checkpoint:
    pass

  @abstractmethod
  def run_analysis(self, analysis_opts: Optional[Dict] = None):
    pass

  @abstractmethod
  def get_ctm_path(self) -> Path:
    pass

  def _get_stm_path(self) -> Path:
    return self.config_builder.variant_params["dataset"]["corpus"].stm_paths[self.stm_corpus_key]

  def run_eval(self):
    if type(self.config_builder.variant_params["dataset"]["corpus"]) == SWBCorpus or (
      type(self.config_builder.variant_params["dataset"]["corpus"]) == librispeech.LibrispeechCorpora
    ):
      if type(self.config_builder.variant_params["dataset"]["corpus"]) == SWBCorpus:
        self.score_job = Hub5ScoreJob(
          ref=self.config_builder.variant_params["dataset"]["corpus"].stm_paths[self.stm_corpus_key],
          glm=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
          hyp=self.get_ctm_path()
        )
      else:
        stm_path = self._get_stm_path()
        ctm_path = self.get_ctm_path()

        concat_num = self.recog_opts["dataset_opts"].get("concat_num")
        if concat_num is not None and concat_num > 10:
          corpus_key = self.corpus_key
          process_concat_stm_and_ctm_job = ProcessConcatStmAndCtm(
            ctm_path=ctm_path,
            stm_path=stm_path,
            seq_tag_prefix=corpus_key
          )
          stm_path = process_concat_stm_and_ctm_job.out_stm
          ctm_path = process_concat_stm_and_ctm_job.out_ctm

        self.score_job = ScliteJob(
          ref=stm_path,
          hyp=ctm_path
        )

      self.score_job.add_alias("%s/scores" % (self.alias,))
      tk.register_output(self.score_job.get_one_alias(), self.score_job.out_report_dir)
    else:
      corpus = self.config_builder.variant_params["dataset"]["corpus"]
      corpus.run_eval(
        corpus_key=self.corpus_key,
        ctm_path=self.get_ctm_path(),
        alias=self.alias
      )


class GlobalAttDecodingExperiment(DecodingExperiment, ABC):
  def __init__(self, config_builder: GlobalConfigBuilder, **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)
    self.config_builder = config_builder

  def get_mini_att_checkpoint(self, train_mini_lstm_opts: Dict) -> Checkpoint:
    num_epochs = 10

    if isinstance(self.config_builder, GlobalConfigBuilder):
      train_mini_lstm_exp = GlobalTrainExperiment(
        config_builder=self.config_builder,
        alias=self.alias,
        num_epochs=num_epochs,
        train_opts={
          "import_model_train_epoch1": self.checkpoint,
          "lr_opts": {
            "type": "newbob",
            "learning_rate": 1e-4,
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "learning_rate_control_error_measure": "dev_error_label_model/output_prob"
          },
          "train_mini_lstm_opts": train_mini_lstm_opts,
          "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
          "max_seq_length": {"targets": 75}
        }
      )
      mini_att_checkpoints, model_dir, learning_rates = train_mini_lstm_exp.run_train()
    else:
      assert isinstance(self.config_builder, GlobalAttConfigBuilderRF)
      train_mini_lstm_exp = GlobalTrainExperiment(
        config_builder=self.config_builder,
        alias=self.alias,
        num_epochs=num_epochs,
        train_rqmt={"time": 10},
        train_opts={
          "preload_from_files": {
            "pretrained_global_att_params": {
              "filename": self.checkpoint,
              "init_for_train": True,
              "ignore_missing": True,
            }
          },
          "train_def": from_scratch_training,
          "train_step_func": _returnn_v2_train_step,
          "batching": "random",
          "aux_loss_layers": None,
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": 1e-4,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": num_epochs
          },
          "train_mini_lstm_opts": train_mini_lstm_opts,
        }
      )
      mini_att_checkpoints, model_dir, learning_rates = train_mini_lstm_exp.run_train()

    return mini_att_checkpoints[num_epochs]


class SegmentalAttDecodingExperiment(DecodingExperiment, ABC):
  def __init__(self, config_builder: SegmentalConfigBuilder, **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)
    self.config_builder = config_builder

    ctc_shallow_fusion_opts = self.recog_opts.get("ctc_shallow_fusion_opts")
    if ctc_shallow_fusion_opts:
      self.alias += "/time-sync-recog-w-ctc_shallow_fusion-%f" % ctc_shallow_fusion_opts["ctc_scale"]
    else:
      self.alias += "/time-sync-recog"

  def get_mini_att_checkpoint(self, train_mini_lstm_opts: Dict) -> Checkpoint:
    train_opts = {}

    num_epochs = 10

    if isinstance(self.config_builder, SegmentalConfigBuilder):
      train_mini_lstm_exp = SegmentalTrainExperiment(
        config_builder=self.config_builder,
        alias=self.alias,
        num_epochs=num_epochs,
        train_opts={
          "import_model_train_epoch1": self.checkpoint,
          "lr_opts": {
            "type": "newbob",
            "learning_rate": 1e-4,
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "learning_rate_control_error_measure": "dev_error_label_model/output_prob"
          },
          "train_mini_lstm_opts": train_mini_lstm_opts,
          **train_opts
        }
      )
    else:
      assert isinstance(self.config_builder, SegmentalAttConfigBuilderRF)
      _, config_builder_ = train_mini_lstm_opts["get_global_att_config_builder_rf_func"](
        use_weight_feedback=self.config_builder.use_weight_feedback,
        use_att_ctx_in_state=self.config_builder.use_att_ctx_in_state,
        decoder_state=self.config_builder.label_decoder_state,
        label_type=f"bpe{self.config_builder.variant_params['dependencies'].model_hyperparameters.target_num_labels_wo_blank}",
      )

      train_mini_lstm_exp = GlobalTrainExperiment(
        config_builder=config_builder_,
        alias=self.alias,
        num_epochs=num_epochs,
        train_rqmt={"time": 10},
        train_opts={
          "preload_from_files": {
            "pretrained_global_att_params": {
              "filename": self.checkpoint,
              "init_for_train": True,
              "ignore_missing": True,
            }
          },
          "train_def": from_scratch_training,
          "train_step_func": _returnn_v2_train_step,
          "batching": "random",
          "aux_loss_layers": None,
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": 1e-4,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": num_epochs
          },
          "train_mini_lstm_opts": train_mini_lstm_opts,
        }
      )

    mini_att_checkpoints, model_dir, learning_rates = train_mini_lstm_exp.run_train()
    return mini_att_checkpoints[num_epochs]


class ReturnnDecodingExperiment(DecodingExperiment, ABC):
  def __init__(
          self,
          search_rqmt: Optional[Dict] = None,
          search_alias: Optional[str] = None,
          **kwargs):
    super().__init__(**kwargs)

    self.concat_num = self.recog_opts.get("dataset_opts", {}).get("concat_num")
    if self.concat_num is not None:
      self.stm_corpus_key += "_concat-%d" % self.concat_num

    self.search_rqmt = search_rqmt if search_rqmt is not None else {}
    self.search_hyps_file = None
    self.best_search_hyps_hdf = None

    self.alias += "/returnn_decoding" if search_alias is None else f"/{search_alias}"

    self.use_recombination = self.recog_opts.get("use_recombination")
    if self.use_recombination is not None:
      assert self.use_recombination in {"sum", "max"}
      self.alias += f"_w-{self.use_recombination}-recomb"

    if isinstance(self, ReturnnSegmentalAttDecodingExperiment):
      length_scale = self.config_builder.variant_params["network"]["length_scale"]
      if length_scale != 1.0:
        self.alias += f"_length-scale-{length_scale:.2f}"
    self.alias += "/%s-checkpoint" % self.checkpoint_alias

    att_readout_scale = self.recog_opts.get("att_readout_scale")
    if att_readout_scale is not None:
      self.alias = f"{self.alias}/att-scale-{att_readout_scale:.2f}"
    h_t_readout_scale = self.recog_opts.get("h_t_readout_scale")
    if h_t_readout_scale is not None:
      self.alias = f"{self.alias}_h_t-scale-{h_t_readout_scale:.2f}"

    base_model_scale = self.recog_opts.get("base_model_scale", 1.0)
    self.alias += f"/scale-{base_model_scale:.2f}"

    blank_penalty = self.recog_opts.get("blank_penalty")
    if blank_penalty:
      self.alias += f"_b-pen-{blank_penalty:.1f}"

    blank_scale = self.recog_opts.get("blank_scale")
    if blank_scale is not None:
      self.alias += f"_b-scale-{blank_scale:.1f}"
    emit_scale = self.recog_opts.get("emit_scale")
    if emit_scale is not None:
      self.alias += f"_e-scale-{emit_scale:.1f}"

    length_normalization_exponent = self.recog_opts.get("length_normalization_exponent")
    if length_normalization_exponent is not None:
      self.alias += f"_len-norm-exp-{length_normalization_exponent:.1f}"

    external_aed_opts = self.recog_opts.get("external_aed_opts")
    if external_aed_opts is not None:
      self.alias = f"{self.alias}_w-ext-aed-scale-{external_aed_opts['scale']}"
    external_transducer_opts = self.recog_opts.get("external_transducer_opts")
    if external_transducer_opts is not None:
      self.alias = f"{self.alias}_w-ext-transducer-scale-{external_transducer_opts['scale']}"

    lm_opts = self.recog_opts.get("lm_opts")
    if lm_opts is not None:
      self.alias += f"/bpe-{lm_opts['type']}-{lm_opts['alias']}-lm-scale-{lm_opts['scale']}-lm-eos-scale-{lm_opts.get('eos_scale', 1.0)}"
      if "add_lm_eos_last_frame" in lm_opts:
        self.alias += f"_add-lm-eos-to-b-hyps-{lm_opts['add_lm_eos_last_frame']}"
      self.alias += f"_add-lm-eos-to-nb-hyps-{lm_opts.get('add_lm_eos_to_non_blank_end_hyps', False)}"
    else:
      self.alias += "/no-lm"
    self.alias = self.get_ilm_correction_alias(self.alias)

    self.alias = f"{self.alias}/beam-size-{self.recog_opts['beam_size']}/{self.stm_corpus_key}"

  def get_ctm_path(self) -> Path:
    recog_config = self.config_builder.get_recog_config(opts=self.recog_opts)

    device = "gpu"
    if self.search_rqmt.get("gpu", 1) == 0:# and self.search_rqmt["gpu"] == 0:
      device = "cpu"

    if isinstance(self.config_builder, ConfigBuilder):
      search_job = ReturnnSearchJobV2(
        search_data={},
        model_checkpoint=self.checkpoint,
        returnn_config=recog_config,
        returnn_python_exe=self.returnn_python_exe,
        returnn_root=self.returnn_root,
        device=device,
        mem_rqmt=self.search_rqmt.get("mem", 4),
        time_rqmt=self.search_rqmt.get("time", 1),
      )
      search_job.add_alias("%s/search" % (self.alias,))

      if recog_config.config["network"]["decision"]["class"] == "decide":
        out_search_file = search_job.out_search_file
      else:
        assert recog_config.config["network"]["decision"]["class"] == "copy"
        search_take_best_job = SearchTakeBestJob(search_py_output=search_job.out_search_file)
        out_search_file = search_take_best_job.out_best_search_results
    else:
      search_job = ReturnnForwardJobV2(
        model_checkpoint=self.checkpoint,
        returnn_config=recog_config,
        returnn_root=self.returnn_root,
        returnn_python_exe=self.returnn_python_exe,
        output_files=["output.py.gz", "best_hyp.hdf"],
        mem_rqmt=self.search_rqmt.get("mem", 6),
        time_rqmt=self.search_rqmt.get("time", 1),
      )
      search_job.rqmt["sbatch_args"] = self.search_rqmt.get("sbatch_args", [])
      search_job.add_alias(f"{self.alias}/search")
      self.search_hyps_file = search_job.out_files["output.py.gz"]
      self.best_search_hyps_hdf = search_job.out_files["best_hyp.hdf"]
      search_take_best_job = SearchTakeBestJob(search_py_output=search_job.out_files["output.py.gz"])
      out_search_file = search_take_best_job.out_best_search_results

    if self.config_builder.variant_params["dependencies"].bpe_codes_path is None:
      out_search_results = SearchOutputRawReplaceJob(
        out_search_file,
        replacement_list=[(" ", ""), ("▁", " ")],
        output_gzip=True
      ).out_search_results
    else:
      out_search_results = SearchBPEtoWordsJob(out_search_file).out_word_search_results

    if self.concat_num is not None:
      return WordsToCTMJobV2(
        words_path=out_search_results
      ).out_ctm_file
    else:
      search_words_to_ctm_job = SearchWordsToCTMJob(
        out_search_results,
        self.config_builder.variant_params["dataset"]["corpus"].corpus_paths[self.corpus_key])

      return search_words_to_ctm_job.out_ctm_file

  def run_analysis(self, analysis_opts: Optional[Dict] = None):
    _analysis_opts = copy.deepcopy(self.default_analysis_opts)
    if analysis_opts is not None:
      _analysis_opts.update(analysis_opts)

    if isinstance(self.config_builder, ConfigBuilderRF):
      att_weight_seq_tags = _analysis_opts["att_weight_seq_tags"]

      if _analysis_opts.get("analyze_gradients", False):
        self.analyze_gradients_job = analysis_rf.analyze_gradients(
          config_builder=self.config_builder,
          seq_tags=att_weight_seq_tags,
          corpus_key=self.stm_corpus_key,
          checkpoint=self.checkpoint,
          returnn_root=self.returnn_root,
          returnn_python_exe=self.returnn_python_exe,
          alias=self.alias,
          hdf_targets=analysis_opts.get("ground_truth_hdf"),
          ref_alignment_hdf=_analysis_opts.get("ref_alignment_hdf"),
          ref_alignment_blank_idx=_analysis_opts.get("ref_alignment_blank_idx"),
          ref_alignment_vocab_path=_analysis_opts.get("ref_alignment_vocab_path"),
          seq_alias="ground-truth",
          do_forced_align_on_gradients=_analysis_opts.get("do_forced_align_on_gradients", False),
          plot_encoder_gradient_graph=_analysis_opts.get("plot_encoder_gradient_graph", False),
          plot_encoder_layers=_analysis_opts.get("analyze_gradients_plot_encoder_layers", False),
          plot_log_gradients=_analysis_opts.get("analyze_gradients_plot_log_gradients", False),
        )
      if _analysis_opts.get("analyze_gradients_search", False):
        analysis_rf.analyze_gradients(
          config_builder=self.config_builder,
          seq_tags=att_weight_seq_tags,
          corpus_key=self.stm_corpus_key,
          checkpoint=self.checkpoint,
          returnn_root=self.returnn_root,
          returnn_python_exe=self.returnn_python_exe,
          alias=self.alias,
          hdf_targets=self.best_search_hyps_hdf,
          ref_alignment_hdf=_analysis_opts["ref_alignment_hdf"],
          ref_alignment_blank_idx=_analysis_opts["ref_alignment_blank_idx"],
          ref_alignment_vocab_path=_analysis_opts["ref_alignment_vocab_path"],
          seq_alias="search"
        )

      if _analysis_opts.get("dump_gradients", False):
        analysis_rf.dump_gradients(
          config_builder=self.config_builder,
          seq_tags=att_weight_seq_tags,
          corpus_key=self.stm_corpus_key,
          checkpoint=self.checkpoint,
          returnn_root=self.returnn_root,
          returnn_python_exe=self.returnn_python_exe,
          alias=self.alias,
          hdf_targets=analysis_opts.get("ground_truth_hdf"),
          seq_alias="ground-truth",
          input_layer_name=_analysis_opts.get("dump_gradients_input_layer_name", "encoder_input"),
        )

      if _analysis_opts.get("dump_self_att", False):
        analysis_rf.dump_self_att(
          config_builder=self.config_builder,
          seq_tags=att_weight_seq_tags,
          corpus_key=self.stm_corpus_key,
          checkpoint=self.checkpoint,
          returnn_root=self.returnn_root,
          returnn_python_exe=self.returnn_python_exe,
          alias=self.alias,
          hdf_targets=analysis_opts.get("ground_truth_hdf"),
          seq_alias="ground-truth",
        )

      if _analysis_opts.get("calc_search_errors", False):
        analysis_rf.calculate_search_errors(
          config_builder=self.config_builder,
          checkpoint=self.checkpoint,
          returnn_root=self.returnn_root,
          returnn_python_exe=self.returnn_python_exe,
          search_hyps_file=self.search_hyps_file,
          best_search_hyp_hdf=self.best_search_hyps_hdf,
          alias=self.alias,
          corpus_key=self.stm_corpus_key,
          realignment_use_recombination=self.use_recombination,
        )
    else:
      forward_recog_config = self.config_builder.get_recog_config_for_forward_job(opts=self.recog_opts)
      forward_search_job = ReturnnForwardJob(
        model_checkpoint=self.checkpoint,
        returnn_config=forward_recog_config,
        returnn_root=self.config_builder.variant_params["returnn_root"],
        returnn_python_exe=self.config_builder.variant_params["returnn_python_exe"],
        eval_mode=False
      )
      forward_search_job.add_alias("%s/analysis/forward_recog_dump_seq" % self.alias)
      search_hdf = forward_search_job.out_default_hdf
      search_not_all_blank_segments = None

      if "baseline_v2/baseline/train_from_global_att_checkpoint/standard-training/win-size-5_200-epochs" in self.alias:
        statistics_job = AlignmentStatisticsJob(
          alignment=search_hdf,
          json_vocab=self.config_builder.dependencies.vocab_path,
          blank_idx=10025,
          silence_idx=20000,  # dummy idx which is larger than the vocab size
          returnn_root=RETURNN_ROOT,
          returnn_python_exe=self.returnn_python_exe
        )
        statistics_job.add_alias("%s/analysis/statistics/%s" % (self.alias, self.corpus_key))
        tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)

      # remove the alignments, which only consist of blank labels because this leads to errors in the following Forward jobs
      # temporarily, only do this for selected models to avoid unnecessarily restarting completed jobs
      for variant in [
        "att_weight_interpolation_no_length_model",
      ]:
        if variant in self.alias:
          remove_all_blank_seqs_job = AlignmentRemoveAllBlankSeqsJob(
            hdf_align_path=forward_search_job.out_default_hdf,
            blank_idx=self.config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx,
            returnn_root=RETURNN_ROOT,
            returnn_python_exe=self.config_builder.variant_params["returnn_python_exe"],
          )
          search_hdf = remove_all_blank_seqs_job.out_align
          search_not_all_blank_segments = remove_all_blank_seqs_job.out_segment_file
          break

      for hdf_alias, hdf_targets in zip(
              ["ground_truth", "search"],
              [_analysis_opts["ground_truth_hdf"], search_hdf]
      ):
        dump_att_weights(
          self.config_builder,
          variant_params=self.config_builder.variant_params,
          checkpoint=self.checkpoint,
          hdf_targets=hdf_targets,
          ref_alignment=_analysis_opts["att_weight_ref_alignment_hdf"],
          corpus_key=self.corpus_key,
          hdf_alias=hdf_alias,
          alias=self.alias,
          ref_alignment_blank_idx=_analysis_opts["att_weight_ref_alignment_blank_idx"],
          seq_tags_to_analyse=_analysis_opts["att_weight_seq_tags"],
          plot_energies=_analysis_opts["plot_energies"],
          dump_ctc=_analysis_opts["dump_ctc"],
          calc_att_weight_stats=_analysis_opts["calc_att_weight_stats"],
          sclite_report_dir=self.score_job.out_report_dir,
        )

        if _analysis_opts.get("dump_ctc_probs"):
          assert isinstance(self.config_builder, GlobalConfigBuilder)
          assert _analysis_opts["att_weight_seq_tags"] is not None, "att_weight_seq_tags must be set for dump_ctc_probs"
          dump_ctc_probs(
            self.config_builder,
            variant_params=self.config_builder.variant_params,
            checkpoint=self.checkpoint,
            hdf_targets=hdf_targets,
            corpus_key=self.corpus_key,
            hdf_alias=hdf_alias,
            alias=self.alias,
            seq_tags_to_analyse=_analysis_opts["att_weight_seq_tags"],
          )

      calc_search_errors(
        self.config_builder,
        variant_params=self.config_builder.variant_params,
        checkpoint=self.checkpoint,
        ground_truth_hdf_targets=_analysis_opts["ground_truth_hdf"],
        search_hdf_targets=search_hdf,
        corpus_key=self.corpus_key,
        alias=self.alias,
        segment_file=search_not_all_blank_segments,
      )


class ReturnnGlobalAttDecodingExperiment(GlobalAttDecodingExperiment, ReturnnDecodingExperiment):
  @property
  def default_recog_opts(self) -> Dict:
    return {
      "concat_num": None,
      "lm_opts": None,
      "ilm_correction_opts": None,
      "beam_size": 12,
      "search_corpus_key": "dev-other"
    }

  @property
  def default_analysis_opts(self) -> Dict:
    return {
      "ground_truth_hdf": None,
      "att_weight_seq_tags": None,
      "plot_energies": False,
      "dump_ctc": True,
      "calc_att_weight_stats": False,
    }


class ReturnnSegmentalAttDecodingExperiment(SegmentalAttDecodingExperiment, ReturnnDecodingExperiment):
  @property
  def default_recog_opts(self) -> Dict:
    return {
      "concat_num": None,
      "lm_opts": None,
      "ilm_correction_opts": None,
      "load_ignore_missing_vars": False,
      "beam_size": 12,
      "search_corpus_key": "dev-other"
    }

  @property
  def default_analysis_opts(self) -> Dict:
    return {
      "att_weight_seq_tags": None,
      "plot_energies": False,
      "dump_ctc": True,
      "calc_att_weight_stats": False,
    }


class RasrDecodingExperiment(DecodingExperiment, ABC):
  def __init__(
          self,
          max_segment_len: int,
          reduction_factor: int,
          reduction_subtrahend: int,
          concurrent: int,
          native_lstm2_so_path: Path = Path("/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
          pruning_opts: Optional[Dict] = None,
          pruning_preset: Optional[str] = "simple-beam-search",
          search_rqmt: Optional[Dict] = None,
          length_norm: bool = False,
          lm_opts: Optional[Dict] = None,
          lm_lookahead_opts: Optional[Dict] = None,
          open_vocab: bool = True,
          segment_list: Optional[List[str]] = None,
          **kwargs):
    super().__init__(**kwargs)

    if self.ilm_correction_opts is not None and self.ilm_correction_opts["type"] == "mini_att":
      raise NotImplementedError("ILM correction is not yet implemented for RASR decoding")

    assert pruning_opts is None and pruning_preset is not None, (
      "For now, we only allow the use of pruning_preset"
    )
    self.pruning_opts = {}
    if pruning_preset is not None:
      self.pruning_opts.update(self.get_pruning_preset(pruning_preset))
    if pruning_opts is not None:
      self.pruning_opts.update(pruning_opts)
    assert set(self.pruning_opts.keys()) == {
      "label_pruning", "label_pruning_limit", "word_end_pruning", "word_end_pruning_limit", "simple_beam_search",
      "full_sum_decoding", "allow_label_recombination", "allow_word_end_recombination"
    }, "pruning_opts is not as expected"

    self.reduction_subtrahend = reduction_subtrahend
    self.reduction_factor = reduction_factor
    self.concurrent = concurrent
    self.max_segment_len = max_segment_len
    self.length_norm = length_norm
    self.search_rqmt = search_rqmt if search_rqmt is not None else {}
    self.lm_opts = lm_opts
    self.lm_lookahead_opts = lm_lookahead_opts
    self.open_vocab = open_vocab
    self.native_lstm2_so_path = native_lstm2_so_path
    self.segment_list = segment_list

    self.alias += "/rasr_decoding/%s-checkpoint/max-seg-len-%d" % (self.checkpoint_alias, self.max_segment_len)

    if self.pruning_opts["simple_beam_search"]:
      self.alias += "/simple_beam_search"
    else:
      self.alias += "/score_based_pruning"
    if open_vocab:
      self.alias += "/open_vocab"
    else:
      self.alias += "/closed_vocab"
    if lm_opts is not None:
      self.lm_opts = copy.deepcopy(lm_opts)
      self.alias += "/lm-%s_scale-%f" % (lm_opts["type"], lm_opts["scale"])
      self.alias = self.get_ilm_correction_alias(self.alias)
    else:
      self.lm_opts = copy.deepcopy(self._get_default_lm_opts())
      self.alias += "/no_lm"

    if self.lm_lookahead_opts is not None:
      self.lm_lookahead_opts = copy.deepcopy(lm_lookahead_opts)
      self.alias += "/lm-lookahead-scale-%f" % lm_lookahead_opts["scale"]
    else:
      self.lm_lookahead_opts = copy.deepcopy(self._get_default_lm_lookahead_opts())
      self.alias += "/wo-lm-lookahead"

  @staticmethod
  def get_pruning_preset(pruning_preset: str) -> Dict:
    if pruning_preset == "simple-beam-search":
      return {
        "label_pruning": 12.0,
        "label_pruning_limit": 12,
        "word_end_pruning": 12.0,
        "word_end_pruning_limit": 12,
        "simple_beam_search": True,
        "full_sum_decoding": False,
        "allow_label_recombination": False,
        "allow_word_end_recombination": False,
      }
    elif pruning_preset == "score-based":
      return {
        "allow_label_recombination": True,
        "allow_word_end_recombination": True,
        "full_sum_decoding": True,
        "label_pruning": 8.0,
        "label_pruning_limit": 128,
        "word_end_pruning": 8.0,
        "word_end_pruning_limit": 128,
        "simple_beam_search": False,
      }
    else:
      raise ValueError("Unknown pruning_preset: %s" % pruning_preset)

  def _get_returnn_graph(self) -> Path:
    recog_config = self.config_builder.get_compile_tf_graph_config(opts=self.recog_opts)
    recog_config.config["network"]["output"]["unit"]["target_embed_masked"]["unit"]["subnetwork"]["target_embed0"]["safe_embedding"] = True
    if "target_embed_length_model_masked" in recog_config.config["network"]["output"]["unit"]:
      recog_config.config["network"]["output"]["unit"]["target_embed_length_model_masked"]["unit"]["subnetwork"]["target_embed_length_model"]["safe_embedding"] = True

    compile_job = CompileTFGraphJob(
      returnn_config=recog_config,
      returnn_python_exe=self.returnn_python_exe,
      returnn_root=self.returnn_root,
      rec_step_by_step="output",
    )
    compile_job.add_alias("%s/compile" % self.alias)
    # tk.register_output(compile_job.get_one_alias(), compile_job.out_graph)

    return compile_job.out_graph

  @staticmethod
  def _get_default_lm_opts() -> Optional[Dict]:
    return None

  @staticmethod
  def _get_default_lm_lookahead_opts() -> Optional[Dict]:
    return None

  def _get_segment_path(self) -> Path:
    if self.segment_list is None:
      return self.config_builder.variant_params["dataset"]["corpus"].segment_corpus_jobs[self.corpus_key].out_single_segment_files[1]
    else:
      return WriteToTextFileJob(content=self.segment_list).out_file

  def _get_stm_path(self) -> Path:
    if self.segment_list is None:
      return super()._get_stm_path()
    else:
      filter_corpus_job = FilterCorpusBySegmentsJob(
        bliss_corpus=self.config_builder.variant_params["dataset"]["corpus"].corpus_paths[self.corpus_key],
        segment_file=self._get_segment_path(),
      )
      return CorpusToStmJob(bliss_corpus=filter_corpus_job.out_corpus).out_stm_path

  def _get_lexicon_path(self) -> Path:
    if self.open_vocab:
      return self.config_builder.variant_params["dependencies"].rasr_format_paths.bpe_no_phoneme_lexicon_path
    else:
      if self.lm_opts is None or self.lm_opts["type"] == "tfrnn":
        return self.config_builder.variant_params["dependencies"].rasr_format_paths.tfrnn_lm_bpe_phoneme_lexicon_path
      else:
        return self.config_builder.variant_params["dependencies"].rasr_format_paths.tfrnn_lm_bpe_phoneme_lexicon_path

  def _get_decoding_config(self) -> Tuple[CommonRasrParameters, RasrConfig]:
    return RasrConfigBuilder.get_decoding_config(
      corpus_path=self.config_builder.variant_params["dataset"]["corpus"].corpus_paths_wav[self.corpus_key],
      segment_path=self._get_segment_path(),
      lexicon_path=self._get_lexicon_path(),
      feature_cache_path=self.config_builder.variant_params["dataset"]["corpus"].oggzip_paths[self.corpus_key],
      feature_extraction_file="feature.flow",
      length_norm=self.length_norm,
      max_seg_len=self.max_segment_len,
      loop_update_history=True,
      blank_update_history=True,
      label_file_path=self.config_builder.variant_params["dependencies"].rasr_format_paths.label_file_path,
      start_label_index=self.config_builder.variant_params["dependencies"].model_hyperparameters.sos_idx,
      blank_label_index=self.config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx,
      skip_silence=False,
      label_recombination_limit=-1,
      reduction_factors=self.reduction_factor,
      reduction_subtrahend=self.reduction_subtrahend,
      debug=False,
      meta_graph_path=self._get_returnn_graph(),
      max_batch_size=256,
      label_scorer_type="tf-rnn-transducer",
      lm_opts=self.lm_opts,
      lm_lookahead_opts=self.lm_lookahead_opts,
      open_vocab=self.open_vocab,
      native_lstm2_so_path=self.native_lstm2_so_path,
      **self.pruning_opts
    )

  def get_ctm_path(self) -> Path:
    decoding_crp, decoding_config = self._get_decoding_config()

    if self.concurrent > 1:
      split_segments_job = SplitSegmentFileJob(
        segment_file=self._get_segment_path(),
        concurrent=self.concurrent
      )

      decoding_crp.corpus_config.segments.file = None
      decoding_crp.segment_path = split_segments_job.out_segment_path
      decoding_crp.concurrent = self.concurrent

    rasr_decoding_job = RASRDecodingJobParallel(
      rasr_exe_path=RasrExecutablesNew.flf_tool_path,
      flf_lattice_tool_config=decoding_config,
      crp=decoding_crp,
      feature_flow=get_raw_wav_feature_flow(dc_detection=False, scale_input=3.0517578125e-05, input_options={"block-size": "1"}),
      model_checkpoint=self.checkpoint,
      dump_best_trace=False,
      mem_rqmt=self.search_rqmt.get("mem", 4),
      time_rqmt=self.search_rqmt.get("time", 1),
      use_gpu=self.search_rqmt.get("gpu", 1) > 0
    )
    rasr_decoding_job.add_alias("%s/search_%s" % (self.alias, self.corpus_key))

    if "closed_vocab/lm-tfrnn_scale-0.300000/lm-lookahead-scale-0.500000/wo_ilm_correction" in self.alias:
      stats_job = RASRDecodingStatisticsJob(search_logs=rasr_decoding_job.out_log_file, corpus_duration_hours=10)
      stats_job.add_alias("%s/stats_%s" % (self.alias, self.corpus_key))
      tk.register_output(stats_job.get_one_alias(), stats_job.elapsed_time)

    # self._best_traces = DumpAlignmentFromTxtJob(
    #   alignment_txt=rasr_decoding_job.out_best_traces,
    #   segment_file=self.dependencies.segment_paths[self.corpus_key],
    #   num_classes=self.dependencies.model_hyperparameters.target_num_labels).out_hdf_align

    lattice_to_ctm_crp, lattice_to_ctm_config = RasrConfigBuilder.get_lattice_to_ctm_config(
      corpus_path=self.config_builder.variant_params["dataset"]["corpus"].corpus_paths[self.corpus_key],
      segment_path=self._get_segment_path(),
      lexicon_path=self._get_lexicon_path(),
      lattice_path=rasr_decoding_job.out_lattice_bundle
    )

    lattice_to_ctm_job = RASRLatticeToCTMJob(
      rasr_exe_path=RasrExecutablesNew.flf_tool_path,
      lattice_path=rasr_decoding_job.out_lattice_bundle,
      crp=lattice_to_ctm_crp,
      flf_lattice_tool_config=lattice_to_ctm_config)
    lattice_to_ctm_job.add_alias("%s/ctm_%s" % (self.alias, self.corpus_key))

    return ConvertCTMBPEToWordsJob(bpe_ctm_file=lattice_to_ctm_job.out_ctm).out_ctm_file

  def run_analysis(self, analysis_opts: Optional[Dict] = None):
    raise NotImplementedError("Analysis is not yet implemented for RASR decoding")


class RasrGlobalAttDecodingExperiment(GlobalAttDecodingExperiment, RasrDecodingExperiment):
  @property
  def default_recog_opts(self) -> Dict:
    return {"search_corpus_key": "dev-other"}

  @property
  def default_analysis_opts(self) -> Dict:
    return {}


class RasrSegmentalAttDecodingExperiment(SegmentalAttDecodingExperiment, RasrDecodingExperiment):
  @property
  def default_recog_opts(self) -> Dict:
    return {"search_corpus_key": "dev-other"}

  @property
  def default_analysis_opts(self) -> Dict:
    return {}


class DecodingPipeline(ABC):
  def __init__(
          self,
          alias: str,
          config_builder: ConfigBuilder,
          checkpoint: Union[Checkpoint, Dict],
          checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
          recog_opts: Optional[Dict] = None,
          analysis_opts: Optional[Dict] = None,
          beam_sizes: Tuple[int, ...] = (12,),
          lm_scales: Tuple[float, ...] = (0.0,),
          lm_opts: Optional[Dict] = None,
          ilm_scales: Tuple[float, ...] = (0.0,),
          ilm_opts: Optional[Dict] = None,
          run_analysis: bool = False,
          search_rqmt: Optional[Dict] = None,
          search_alias: Optional[str] = None,
          corpus_keys: Tuple[str, ...] = ("dev-other",),
          only_do_analysis: bool = False
  ):
    self.recog_opts = recog_opts if recog_opts is not None else {}
    for key in ("lm_opts", "ilm_correction_opts", "beam_size", "search_corpus_key"):
      assert key not in self.recog_opts, f"{key} is set by the pipeline"

    if "dataset_opts" not in self.recog_opts:
      self.recog_opts["dataset_opts"] = {}

    self.alias = alias
    self.config_builder = config_builder
    self.checkpoint = checkpoint
    self.checkpoint_aliases = checkpoint_aliases
    self.analysis_opts = analysis_opts
    self.beam_sizes = beam_sizes
    self.lm_scales = lm_scales
    self.lm_opts = lm_opts if lm_opts is not None else {}
    self.ilm_scales = ilm_scales
    self.ilm_opts = ilm_opts if ilm_opts is not None else {}
    self.run_analysis = run_analysis
    self.search_rqmt = search_rqmt if search_rqmt is not None else {}
    self.search_alias = search_alias
    self.corpus_keys = corpus_keys
    self.only_do_analysis = only_do_analysis

    self.decoding_exps = []

  @abstractmethod
  def run_experiment(
          self, beam_size: int, lm_scale: float, ilm_scale: float, checkpoint_alias: str):
    pass

  def run(self):
    for checkpoint_alias in self.checkpoint_aliases:
      for beam_size in self.beam_sizes:
        for lm_scale in self.lm_scales:
          for ilm_scale in self.ilm_scales:
            for corpus_key in self.corpus_keys:
              self.recog_opts.update({
                "lm_opts": {"scale": lm_scale, **self.lm_opts} if lm_scale > 0 else None,
                "ilm_correction_opts": {
                  "scale": ilm_scale, **self.ilm_opts} if ilm_scale > 0 else None,
                "beam_size": beam_size,
                # "search_corpus_key": corpus_key
              })
              self.recog_opts["dataset_opts"]["corpus_key"] = corpus_key

              exp = self.run_experiment(
                beam_size=beam_size,
                lm_scale=lm_scale,
                ilm_scale=ilm_scale,
                checkpoint_alias=checkpoint_alias
              )
              self.decoding_exps.append(exp)


class ReturnnGlobalAttDecodingPipeline(DecodingPipeline):
  def __init__(self, config_builder: GlobalConfigBuilder, **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)
    self.config_builder = config_builder

  def run_experiment(self, beam_size: int, lm_scale: float, ilm_scale: float, checkpoint_alias: str):
    recog_opts = copy.deepcopy(self.recog_opts)
    search_rqmt = copy.deepcopy(self.search_rqmt)

    if lm_scale > 0:
      if beam_size in (64, 84):
        if "batch_size" not in recog_opts:
          recog_opts["batch_size"] = 1250
        if "time" not in search_rqmt:
          search_rqmt["time"] = 6

    exp = ReturnnGlobalAttDecodingExperiment(
      alias=self.alias,
      config_builder=self.config_builder,
      recog_opts=recog_opts,
      checkpoint=self.checkpoint,
      checkpoint_alias=checkpoint_alias,
      search_alias=self.search_alias,
      search_rqmt=search_rqmt
    )
    if not self.only_do_analysis:
      exp.run_eval()
    if self.run_analysis:
      exp.run_analysis(self.analysis_opts)

    return exp


class ReturnnSegmentalAttDecodingPipeline(DecodingPipeline):
  def __init__(self, config_builder: SegmentalConfigBuilder, **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)
    self.config_builder = config_builder

    self.realignment = None
    if self.run_analysis and not isinstance(self.config_builder, ConfigBuilderRF):
      self.realignment = RasrRealignmentExperiment(
        alias=self.alias,
        reduction_factor=960,
        reduction_subtrahend=399,
        job_rqmt={
          "mem": 4,
          "time": 1,
          "gpu": 0
        },
        concurrent=100,
        checkpoint=self.checkpoint,
        checkpoint_alias=self.checkpoint_aliases[0],
        config_builder=config_builder,
      ).get_realignment()
      if "ground_truth_hdf" not in self.analysis_opts:
        self.analysis_opts["ground_truth_hdf"] = self.realignment

  def run_experiment(self, beam_size: int, lm_scale: float, ilm_scale: float, checkpoint_alias: str):
    recog_opts = copy.deepcopy(self.recog_opts)
    search_rqmt = copy.deepcopy(self.search_rqmt)

    if lm_scale > 0:
      if beam_size == 12:
        if "batch_size" not in recog_opts:
          recog_opts["batch_size"] = 12_000
        if "time" not in search_rqmt:
          search_rqmt["time"] = 2
      elif beam_size in (24, 32, 64, 84):
        if "batch_size" not in recog_opts:
          recog_opts["batch_size"] = 1250
        if "time" not in search_rqmt:
          search_rqmt["time"] = 12
    else:
      if "batch_size" not in recog_opts:
        recog_opts["batch_size"] = 15_000
      if "time" not in search_rqmt:
        search_rqmt["time"] = 2

    exp = ReturnnSegmentalAttDecodingExperiment(
      alias=self.alias,
      config_builder=self.config_builder,
      recog_opts=recog_opts,
      checkpoint=self.checkpoint,
      checkpoint_alias=checkpoint_alias,
      search_rqmt=search_rqmt,
      search_alias=self.search_alias
    )
    if not self.only_do_analysis:
      exp.run_eval()
    if self.run_analysis:
      exp.run_analysis(self.analysis_opts)

    return exp
