from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple, Union

from i6_core.returnn.compile import CompileTFGraphJob
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.rasr.config import RasrConfig
from i6_core.corpus.segments import SplitSegmentFileJob
from i6_core.features.common import samples_flow
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.returnn.training import Checkpoint

from i6_experiments.users.schmitt.flow import get_raw_wav_feature_flow_w_alignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutablesNew
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT, RETURNN_ROOT
from i6_experiments.users.schmitt.rasr.realignment import RASRRealignmentParallelJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder
from i6_experiments.users.schmitt.hdf import build_hdf_from_alignment


class RasrRealignmentExperiment:
  def __init__(
          self,
          alias: str,
          config_builder: SegmentalConfigBuilder,
          checkpoint: Union[Checkpoint, Dict],
          checkpoint_alias: str,
          reduction_factor: int,
          reduction_subtrahend: int,
          concurrent: int,
          native_lstm2_so_path: Path = Path(
            "/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
          pruning_opts: Optional[Dict] = None,
          recog_config_opts: Optional[Dict] = None,
          job_rqmt: Optional[Dict] = None,
  ):
    self.alias = alias
    self.alias += "/rasr_realignment"

    self.config_builder = config_builder
    self.checkpoint_alias = checkpoint_alias
    if isinstance(checkpoint, Checkpoint):
      self.checkpoint = checkpoint
    else:
      assert isinstance(checkpoint, dict)
      self.checkpoint = config_builder.get_recog_checkpoints(**checkpoint)[checkpoint_alias]

    self.pruning_opts = copy.deepcopy(self.get_default_pruning_opts())
    if pruning_opts is not None:
      self.pruning_opts.update(pruning_opts)

    self.recog_config_opts = copy.deepcopy(self.get_default_recog_config_opts())
    if recog_config_opts is not None:
      self.recog_config_opts.update(recog_config_opts)

    self.reduction_subtrahend = reduction_subtrahend
    self.reduction_factor = reduction_factor
    self.concurrent = concurrent
    self.job_rqmt = job_rqmt if job_rqmt is not None else {}
    self.native_lstm2_so_path = native_lstm2_so_path
    self.corpus_key = self.recog_config_opts["search_corpus_key"]

    self.returnn_python_exe = self.config_builder.variant_params["returnn_python_exe"]
    self.returnn_root = self.config_builder.variant_params["returnn_root"]

  @staticmethod
  def get_default_pruning_opts() -> Dict:
    return {
      "label_pruning": 50.0,
      "label_pruning_limit": 100000,
      "length_norm": False,
      "max_segment_len": -1,
    }

  @staticmethod
  def get_default_recog_config_opts() -> Dict:
    return {
      "search_corpus_key": "dev-other",
    }

  def _get_returnn_graph(self) -> Path:
    recog_config = self.config_builder.get_compile_tf_graph_config(opts=self.recog_config_opts)
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

    return compile_job.out_graph

  def _get_lexicon_path(self) -> Path:
    return self.config_builder.variant_params["dependencies"].rasr_format_paths.tfrnn_lm_bpe_phoneme_lexicon_path

  def _get_realignment_config(self) -> Tuple[CommonRasrParameters, RasrConfig]:
    return RasrConfigBuilder.get_realignment_config(
      corpus_path=self.config_builder.variant_params["dataset"]["corpus"].corpus_paths_wav[self.corpus_key],
      segment_path=self.config_builder.variant_params["dataset"]["corpus"].segment_corpus_jobs[self.corpus_key].out_single_segment_files[1],
      lexicon_path=self._get_lexicon_path(),
      feature_cache_path=None,
      feature_extraction_file="feature.flow",
      loop_update_history=True,
      blank_update_history=True,
      label_file=self.config_builder.variant_params["dependencies"].rasr_format_paths.label_file_path,
      start_label_index=self.config_builder.variant_params["dependencies"].model_hyperparameters.sos_idx,
      blank_label_index=self.config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx,
      label_recombination_limit=-1,
      reduction_factors=self.reduction_factor,
      reduction_subtrahend=self.reduction_subtrahend,
      meta_graph_path=self._get_returnn_graph(),
      native_lstm2_so_path=self.native_lstm2_so_path,
      allophone_path=self.config_builder.variant_params["dependencies"].rasr_format_paths.allophone_path,
      state_tying_path=self.config_builder.variant_params["dependencies"].rasr_format_paths.state_tying_path,
      context_size=-1,
      **self.pruning_opts
    )

  def get_realignment(self):
    realignment_crp, realignment_config = self._get_realignment_config()

    if self.concurrent > 1:
      split_segments_job = SplitSegmentFileJob(
        segment_file=self.config_builder.variant_params["dataset"]["corpus"].segment_corpus_jobs[self.corpus_key].out_single_segment_files[1],
        concurrent=self.concurrent
      )

      realignment_crp.corpus_config.segments.file = None
      realignment_crp.segment_path = split_segments_job.out_segment_path
      realignment_crp.concurrent = self.concurrent

    rasr_realignment_job = RASRRealignmentParallelJob(
      rasr_exe_path=RasrExecutablesNew.am_trainer_path,
      am_model_trainer_config=realignment_config,
      crp=realignment_crp,
      feature_flow=get_raw_wav_feature_flow_w_alignment(dc_detection=False, scale_input=3.0517578125e-05, input_options={"block-size": "1"}),
      model_checkpoint=self.checkpoint,
      mem_rqmt=self.job_rqmt.get("mem", 4),
      time_rqmt=self.job_rqmt.get("time", 1),
      use_gpu=self.job_rqmt.get("gpu", 1) > 0,
      blank_allophone_state_idx=self.config_builder.variant_params["dependencies"].rasr_format_paths.blank_allophone_state_idx,
    )
    rasr_realignment_job.add_alias("%s/realign_%s" % (self.alias, self.corpus_key))
    # tk.register_output(rasr_realignment_job.get_one_alias(), rasr_realignment_job.out_alignment_bundle)

    hdf_align = build_hdf_from_alignment(
      alignment_cache=rasr_realignment_job.out_alignment_bundle,
      allophone_file=self.config_builder.variant_params["dependencies"].rasr_format_paths.allophone_path,
      state_tying_file=self.config_builder.variant_params["dependencies"].rasr_format_paths.state_tying_path,
      returnn_python_exe=RETURNN_EXE_NEW,
      returnn_root=RETURNN_CURRENT_ROOT,
      silence_phone="[blank]"
    )

    return hdf_align

  @property
  def default_realign_opts(self) -> Dict:
    return {"search_corpus_key": "dev-other"}
