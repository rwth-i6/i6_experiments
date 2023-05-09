from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import LabelDefinition, SegmentalLabelDefinition, GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.graph import ReturnnGraph
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.corpora.corpora import SWBCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_recog_config as get_segmental_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_compile_config as get_segmental_compile_config

from i6_experiments.users.schmitt.returnn.search import ReturnnDumpSearchJob
from i6_experiments.users.schmitt.rasr.convert import RASRLatticeToCTMJob, ConvertCTMBPEToWordsJob
from i6_experiments.users.schmitt.rasr.recognition import RASRDecodingJobParallel
from i6_experiments.users.schmitt.alignment.alignment import DumpAlignmentFromTxtJob

from i6_core.returnn.search import ReturnnSearchJobV2, SearchWordsToCTMJob, SearchBPEtoWordsJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.recognition.scoring import Hub5ScoreJob
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.corpus.segments import SplitSegmentFileJob

from sisyphus import *

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional


class DecodingExperiment(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          returnn_config: ReturnnConfig,
          variant_params: Dict,
          checkpoint: Checkpoint,
          corpus_key: str,
          dump_best_traces: bool,
          base_alias: str
  ):
    self.returnn_config = returnn_config
    self.checkpoint = checkpoint
    self.corpus_key = corpus_key
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.dump_best_traces = dump_best_traces

    self.base_alias = base_alias

    self._best_traces = None

  @property
  def best_traces(self) -> Path:
    return self._best_traces

  @abstractmethod
  def get_ctm_path(self) -> Path:
    pass

  def run_eval(self):
    score_job = Hub5ScoreJob(
      ref=self.dependencies.stm_paths[self.corpus_key],
      glm=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
      hyp=self.get_ctm_path()
    )
    score_job.add_alias("%s/scores_%s" % (self.base_alias, self.corpus_key))
    tk.register_output(score_job.get_one_alias(), score_job.out_report_dir)


class ReturnnDecodingExperiment(DecodingExperiment):
  def get_ctm_path(self) -> Path:
    if self.dump_best_traces:
      search_job = ReturnnDumpSearchJob(
        search_data={},
        model_checkpoint=self.checkpoint,
        returnn_config=self.returnn_config,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        mem_rqmt=6,
        time_rqmt=1)
      self._best_traces = search_job.out_search_seqs_file
    else:
      search_job = ReturnnSearchJobV2(
        search_data={},
        model_checkpoint=self.checkpoint,
        returnn_config=self.returnn_config,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        device="gpu",
        mem_rqmt=4,
        time_rqmt=1)

    search_job.add_alias("%s/search_%s" % (self.base_alias, self.corpus_key))

    bpe_to_words_job = SearchBPEtoWordsJob(search_job.out_search_file)

    return SearchWordsToCTMJob(
      bpe_to_words_job.out_word_search_results,
      self.dependencies.stm_jobs[self.corpus_key].bliss_corpus).out_ctm_file


class SegmentalReturnnDecodingExperiment(ReturnnDecodingExperiment):
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies


class GlobalReturnnDecodingExperiment(ReturnnDecodingExperiment):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies


class RasrDecodingExperiment(DecodingExperiment):
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          length_norm: bool,
          label_pruning: Optional[float],
          label_pruning_limit: int,
          word_end_pruning: Optional[float],
          word_end_pruning_limit: int,
          full_sum_decoding: bool,
          allow_recombination: bool,
          max_segment_len: int,
          time_rqmt: int,
          mem_rqmt: int,
          concurrent: int = 1,
          **kwargs
  ):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

    self.concurrent = concurrent

    # if the two pruning thresholds are not set, we use simple beam search
    # otherwise, we use threshold based pruning + histogram pruning
    self.simple_beam_search = label_pruning is None and word_end_pruning is None
    self.label_pruning = label_pruning if label_pruning is not None else 12.0  # add dummy value (not used)
    self.word_end_pruning = word_end_pruning if word_end_pruning is not None else 12.0  # add dummy value (not used)

    self.label_pruning_limit = label_pruning_limit
    self.word_end_pruning_limit = word_end_pruning_limit

    self.length_norm = length_norm

    self.full_sum_decoding = full_sum_decoding
    self.allow_recombination = allow_recombination

    self.max_segment_len = max_segment_len

    self.time_rqmt = time_rqmt
    self.mem_rqmt = mem_rqmt

  def _get_decoding_config(self) -> Tuple[CommonRasrParameters, RasrConfig]:
    return RasrConfigBuilder.get_decoding_config(
      corpus_path=self.dependencies.corpus_paths[self.corpus_key],
      length_norm=self.length_norm,
      segment_path=self.dependencies.segment_paths[self.corpus_key],
      lexicon_path=self.dependencies.rasr_format_paths.decoding_lexicon_path,
      feature_cache_path=self.dependencies.feature_cache_paths[self.corpus_key],
      label_pruning=self.label_pruning,
      label_unit="word",
      label_pruning_limit=self.label_pruning_limit,
      word_end_pruning_limit=self.word_end_pruning_limit,
      loop_update_history=True,
      simple_beam_search=self.simple_beam_search,
      full_sum_decoding=self.full_sum_decoding,
      blank_update_history=True,
      word_end_pruning=self.word_end_pruning,
      allow_word_end_recombination=self.allow_recombination,
      allow_label_recombination=self.allow_recombination,
      lm_type="simple-history",
      lm_image=None,
      lm_scale=None,
      use_lm_score=False,
      lm_file=None,
      label_file_path=self.dependencies.rasr_format_paths.label_file_path,
      start_label_index=self.dependencies.model_hyperparameters.sos_idx,
      skip_silence=False,
      blank_label_index=self.dependencies.model_hyperparameters.blank_idx,
      label_recombination_limit=-1,
      reduction_factors=int(np.prod(self.variant_params["config"]["time_red"])),
      debug=False,
      meta_graph_path=ReturnnGraph(self.returnn_config).meta_graph_path,
      lm_lookahead=False,
      max_seg_len=self.max_segment_len,
      lm_lookahead_cache_size_high=None,
      lm_lookahead_cache_size_low=None,
      lm_lookahead_scale=None,
      lm_lookahead_history_limit=None,
      max_batch_size=256,
      label_scorer_type="tf-rnn-transducer",
    )

  def get_ctm_path(self) -> Path:
    decoding_crp, decoding_config = self._get_decoding_config()

    split_segments_job = SplitSegmentFileJob(
      segment_file=self.dependencies.segment_paths[self.corpus_key], concurrent=self.concurrent)

    decoding_crp.corpus_config.segments.file = None
    decoding_crp.segment_path = split_segments_job.out_segment_path
    decoding_crp.concurrent = self.concurrent

    rasr_decoding_job = RASRDecodingJobParallel(
      rasr_exe_path=RasrExecutables.flf_tool_path,
      flf_lattice_tool_config=decoding_config,
      crp=decoding_crp,
      model_checkpoint=self.checkpoint,
      dump_best_trace=self.dump_best_traces,
      mem_rqmt=self.mem_rqmt,
      time_rqmt=self.time_rqmt,
      use_gpu=True)
    rasr_decoding_job.add_alias("%s/rasr-decoding_%s" % (self.base_alias, self.corpus_key))

    if self.dump_best_traces:
      self._best_traces = DumpAlignmentFromTxtJob(
        alignment_txt=rasr_decoding_job.out_best_traces,
        segment_file=self.dependencies.segment_paths[self.corpus_key],
        num_classes=self.dependencies.model_hyperparameters.target_num_labels).out_hdf_align

    lattice_to_ctm_crp, lattice_to_ctm_config = RasrConfigBuilder.get_lattice_to_ctm_config(
      corpus_path=self.dependencies.corpus_paths[self.corpus_key],
      segment_path=self.dependencies.segment_paths[self.corpus_key],
      lexicon_path=self.dependencies.rasr_format_paths.decoding_lexicon_path,
      lattice_path=rasr_decoding_job.out_lattice_bundle
    )

    lattice_to_ctm_job = RASRLatticeToCTMJob(
      rasr_exe_path=RasrExecutables.flf_tool_path,
      lattice_path=rasr_decoding_job.out_lattice_bundle,
      crp=lattice_to_ctm_crp,
      flf_lattice_tool_config=lattice_to_ctm_config)

    return ConvertCTMBPEToWordsJob(bpe_ctm_file=lattice_to_ctm_job.out_ctm).out_ctm_file
