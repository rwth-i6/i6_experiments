from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.search import ReturnnSearchJobV2, SearchWordsToCTMJob, SearchBPEtoWordsJob, SearchTakeBestJob
from i6_core.recognition.scoring import Hub5ScoreJob, ScliteJob
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.compile import CompileTFGraphJob
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.rasr.config import RasrConfig
from i6_core.corpus.segments import SplitSegmentFileJob
from i6_core.features.common import samples_flow
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_core.corpus.convert import CorpusToStmJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBCorpus
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.search_errors import calc_search_errors
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.att_weights import dump_att_weights, dump_length_model_probs
from i6_experiments.users.schmitt.corpus.concat.convert import WordsToCTMJobV2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_CURRENT_ROOT, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables, RasrExecutablesNew
from i6_experiments.users.schmitt.rasr.recognition import RASRDecodingJobParallel, RASRDecodingStatisticsJob
from i6_experiments.users.schmitt.rasr.convert import RASRLatticeToCTMJob, ConvertCTMBPEToWordsJob
from i6_experiments.users.schmitt.alignment.alignment import AlignmentRemoveAllBlankSeqsJob


class DecodingExperiment(ABC):
  def __init__(
          self,
          alias: str,
          config_builder: ConfigBuilder,
          checkpoint: Checkpoint,
          corpus_key: str,
          ilm_correction_opts: Optional[Dict] = None,
  ):
    self.config_builder = config_builder
    self.checkpoint = checkpoint
    self.corpus_key = corpus_key
    self.stm_corpus_key = corpus_key
    self.ilm_correction_opts = ilm_correction_opts

    self.alias = alias

    self.returnn_python_exe = self.config_builder.variant_params["returnn_python_exe"]
    self.returnn_root = self.config_builder.variant_params["returnn_root"]

  def get_ilm_correction_alias(self, alias: str):
    if self.ilm_correction_opts is not None:
      alias += "/ilm_correction_scale-%f" % self.ilm_correction_opts["scale"]
      if "correct_eos" in self.ilm_correction_opts:
        if self.ilm_correction_opts["correct_eos"]:
          alias += "/correct_eos"
        else:
          alias += "/wo_correct_eos"
      if self.ilm_correction_opts.get("mini_att_in_s_for_train", False):
        alias += "/w_mini_att_in_s_for_train"
      else:
        alias += "/wo_mini_att_in_s_for_train"
      if self.ilm_correction_opts.get("use_se_loss", False):
        alias += "/w_se_loss"
      else:
        alias += "/wo_se_loss"
      if self.ilm_correction_opts.get("mini_att_train_num_epochs", None):
        alias += "/mini_att_train_num_epochs-%d" % self.ilm_correction_opts["mini_att_train_num_epochs"]
    else:
      alias += "/wo_ilm_correction"

    return alias

  def get_config_recog_opts(self):
    return {
      "search_corpus_key": self.corpus_key,
      "ilm_correction_opts": self.ilm_correction_opts,
    }

  @abstractmethod
  def get_ctm_path(self) -> Path:
    pass

  def _get_stm_path(self) -> Path:
    return self.config_builder.variant_params["dataset"]["corpus"].stm_paths[self.stm_corpus_key]

  def run_eval(self):
    if type(self.config_builder.variant_params["dataset"]["corpus"]) == SWBCorpus:
      score_job = Hub5ScoreJob(
        ref=self.config_builder.variant_params["dataset"]["corpus"].stm_paths[self.stm_corpus_key],
        glm=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
        hyp=self.get_ctm_path()
      )
    else:
        score_job = ScliteJob(
          ref=self._get_stm_path(),
          hyp=self.get_ctm_path()
        )

    score_job.add_alias("%s/scores_%s" % (self.alias, self.stm_corpus_key))
    tk.register_output(score_job.get_one_alias(), score_job.out_report_dir)


class ReturnnDecodingExperimentV2(DecodingExperiment):
  def __init__(
          self,
          concat_num: Optional[int],
          search_rqmt: Optional[Dict],
          batch_size: Optional[int],
          load_ignore_missing_vars: bool = False,
          lm_opts: Optional[Dict] = None,
          **kwargs):
    super().__init__(**kwargs)

    if concat_num is not None:
      self.stm_corpus_key += "_concat-%d" % concat_num

    self.batch_size = batch_size
    self.concat_num = concat_num
    self.search_rqmt = search_rqmt
    self.load_ignore_missing_vars = load_ignore_missing_vars
    self.lm_opts = lm_opts

    self.alias += "/returnn_decoding"

    if lm_opts is not None:
      self.alias += "/bpe-lm-scale-%f" % (lm_opts["scale"],)
      if "add_lm_eos_last_frame" in lm_opts:
        self.alias += "_add-lm-eos-%s" % lm_opts["add_lm_eos_last_frame"]
      self.alias = self.get_ilm_correction_alias(self.alias)
    else:
      self.alias += "/no-lm"
      if self.ilm_correction_opts is not None:
        self.alias = self.get_ilm_correction_alias(self.alias)

  def get_config_recog_opts(self):
    recog_opts = super().get_config_recog_opts()
    recog_opts.update({
      "batch_size": self.batch_size,
      "dataset_opts": {"concat_num": self.concat_num},
      "load_ignore_missing_vars": self.load_ignore_missing_vars,
    }
      "lm_opts": self.lm_opts,
    })
    return recog_opts

  def get_ctm_path(self) -> Path:
    recog_config = self.config_builder.get_recog_config(opts=self.get_config_recog_opts())

    device = "gpu"
    if self.search_rqmt and self.search_rqmt["gpu"] == 0:
      device = "cpu"

    search_job = ReturnnSearchJobV2(
      search_data={},
      model_checkpoint=self.checkpoint,
      returnn_config=recog_config,
      returnn_python_exe=self.returnn_python_exe,
      returnn_root=self.returnn_root,
      device=device,
      mem_rqmt=4,
      time_rqmt=1)

    if self.search_rqmt:
      search_job.rqmt = self.search_rqmt

    search_job.add_alias("%s/search_%s" % (self.alias, self.stm_corpus_key))

    if recog_config.config["network"]["decision"]["class"] == "decide":
      out_search_file = search_job.out_search_file
    else:
      assert recog_config.config["network"]["decision"]["class"] == "copy"
      search_take_best_job = SearchTakeBestJob(search_py_output=search_job.out_search_file)
      out_search_file = search_take_best_job.out_best_search_results

    bpe_to_words_job = SearchBPEtoWordsJob(out_search_file)

    if self.concat_num is not None:
      return WordsToCTMJobV2(
        words_path=bpe_to_words_job.out_word_search_results
      ).out_ctm_file
    else:
      search_words_to_ctm_job = SearchWordsToCTMJob(
        bpe_to_words_job.out_word_search_results,
        self.config_builder.variant_params["dataset"]["corpus"].corpus_paths[self.corpus_key])

      return search_words_to_ctm_job.out_ctm_file

  def run_analysis(
          self,
          ground_truth_hdf: Optional[Path],
          att_weight_ref_alignment_hdf: Path,
          att_weight_ref_alignment_blank_idx: int,
          att_weight_seq_tags: Optional[List[str]] = None,
  ):
    forward_recog_config = self.config_builder.get_recog_config_for_forward_job(opts=self.get_recog_opts())
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

    # remove the alignments, which only consist of blank labels because this leads to errors in the following Forward jobs
    # temporarily, only do this for selected models to avoid unnecessarily restarting completed jobs
    for variant in [
      "no_label_feedback",
      "non_blank_ctx",
      "linear_layer",
      "couple_length_and_label_model",
      "use_label_model_state",
      "chunking",
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
            [ground_truth_hdf, search_hdf]
    ):
      dump_att_weights(
        self.config_builder,
        variant_params=self.config_builder.variant_params,
        checkpoint=self.checkpoint,
        hdf_targets=hdf_targets,
        ref_alignment=att_weight_ref_alignment_hdf,
        corpus_key=self.corpus_key,
        hdf_alias=hdf_alias,
        alias=self.alias,
        ref_alignment_blank_idx=att_weight_ref_alignment_blank_idx,
        seq_tags_to_analyse=att_weight_seq_tags,
      )

      if "couple_length_and_label_model" in self.alias:
        # also dump the length model scores
        dump_length_model_probs(
          self.config_builder,
          variant_params=self.config_builder.variant_params,
          checkpoint=self.checkpoint,
          hdf_targets=hdf_targets,
          ref_alignment=att_weight_ref_alignment_hdf,
          corpus_key=self.corpus_key,
          hdf_alias=hdf_alias,
          alias=self.alias,
          ref_alignment_blank_idx=att_weight_ref_alignment_blank_idx,
          seq_tags_to_analyse=att_weight_seq_tags,
        )

    calc_search_errors(
      self.config_builder,
      variant_params=self.config_builder.variant_params,
      checkpoint=self.checkpoint,
      ground_truth_hdf_targets=ground_truth_hdf,
      search_hdf_targets=search_hdf,
      corpus_key=self.corpus_key,
      alias=self.alias,
      segment_file=search_not_all_blank_segments,
    )


class RasrDecodingExperiment(DecodingExperiment):
  def __init__(
          self,
          search_rqmt: Optional[Dict],
          length_norm: bool,
          label_pruning: float,
          label_pruning_limit: int,
          word_end_pruning: float,
          word_end_pruning_limit: int,
          simple_beam_search: bool,
          full_sum_decoding: bool,
          allow_recombination: bool,
          max_segment_len: int,
          reduction_factor: int,
          reduction_subtrahend: int,
          concurrent: int,
          native_lstm2_so_path: Path,
          lm_opts: Optional[Dict] = None,
          lm_lookahead_opts: Optional[Dict] = None,
          open_vocab: bool = True,
          segment_list: Optional[List[str]] = None,
          **kwargs):
    super().__init__(**kwargs)

    self.reduction_subtrahend = reduction_subtrahend
    self.reduction_factor = reduction_factor
    self.concurrent = concurrent
    self.max_segment_len = max_segment_len
    self.allow_recombination = allow_recombination
    self.full_sum_decoding = full_sum_decoding
    self.simple_beam_search = simple_beam_search
    self.word_end_pruning_limit = word_end_pruning_limit
    self.word_end_pruning = word_end_pruning
    self.label_pruning_limit = label_pruning_limit
    self.label_pruning = label_pruning
    self.length_norm = length_norm
    self.search_rqmt = search_rqmt
    self.lm_opts = lm_opts
    self.lm_lookahead_opts = lm_lookahead_opts
    self.open_vocab = open_vocab
    self.native_lstm2_so_path = native_lstm2_so_path
    self.segment_list = segment_list

    self.alias += "/rasr_decoding/max-seg-len-%d" % self.max_segment_len

    if simple_beam_search:
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
      if self.ilm_correction_opts is not None:
        self.alias = self.get_ilm_correction_alias(self.alias)

    if self.lm_lookahead_opts is not None:
      self.lm_lookahead_opts = copy.deepcopy(lm_lookahead_opts)
      self.alias += "/lm-lookahead-scale-%f" % lm_lookahead_opts["scale"]
    else:
      self.lm_lookahead_opts = copy.deepcopy(self._get_default_lm_lookahead_opts())
      self.alias += "/wo-lm-lookahead"

  def _get_returnn_graph(self) -> Path:
    recog_config = self.config_builder.get_compile_tf_graph_config(opts=self.get_config_recog_opts())
    recog_config.config["network"]["output"]["unit"]["target_embed_masked"]["unit"]["subnetwork"]["target_embed0"]["safe_embedding"] = True
    compile_job = CompileTFGraphJob(
      returnn_config=recog_config,
      returnn_python_exe=self.returnn_python_exe,
      returnn_root=self.returnn_root,
      rec_step_by_step="output",
    )
    compile_job.add_alias("%s/compile" % self.alias)
    tk.register_output(compile_job.get_one_alias(), compile_job.out_graph)

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
      label_pruning=self.label_pruning,
      label_pruning_limit=self.label_pruning_limit,
      word_end_pruning=self.word_end_pruning,
      word_end_pruning_limit=self.word_end_pruning_limit,
      length_norm=self.length_norm,
      full_sum_decoding=self.full_sum_decoding,
      allow_word_end_recombination=self.allow_recombination,
      allow_label_recombination=self.allow_recombination,
      max_seg_len=self.max_segment_len,
      simple_beam_search=self.simple_beam_search,
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
      feature_flow=samples_flow(dc_detection=False, scale_input=3.0517578125e-05, input_options={"block-size": "1"}),
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

    return ConvertCTMBPEToWordsJob(bpe_ctm_file=lattice_to_ctm_job.out_ctm).out_ctm_file
