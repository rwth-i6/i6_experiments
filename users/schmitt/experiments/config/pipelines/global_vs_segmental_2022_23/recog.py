from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBCorpus
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.search_errors import calc_search_errors
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.att_weights import dump_att_weights
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_core.returnn.training import Checkpoint, ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.config import CodeWrapper

from i6_core.returnn.search import ReturnnSearchJobV2, SearchWordsToCTMJob, SearchBPEtoWordsJob, SearchTakeBestJob
from i6_core.recognition.scoring import Hub5ScoreJob, ScliteJob
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig

from i6_experiments.users.schmitt.corpus.concat.convert import WordsToCTMJobV2

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_CURRENT_ROOT, RETURNN_ROOT


class DecodingExperiment(ABC):
  def __init__(
          self,
          alias: str,
          config_builder: ConfigBuilder,
          checkpoint: Checkpoint,
          corpus_key: str,
  ):
    self.config_builder = config_builder
    self.checkpoint = checkpoint
    self.corpus_key = corpus_key
    self.stm_corpus_key = corpus_key

    self.alias = alias

  @abstractmethod
  def get_ctm_path(self) -> Path:
    pass

  def run_eval(self):
    if type(self.config_builder.variant_params["dataset"]["corpus"]) == SWBCorpus:
      score_job = Hub5ScoreJob(
        ref=self.config_builder.variant_params["dataset"]["corpus"].stm_paths[self.stm_corpus_key],
        glm=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
        hyp=self.get_ctm_path()
      )
    else:
        score_job = ScliteJob(
          ref=self.config_builder.variant_params["dataset"]["corpus"].stm_paths[self.stm_corpus_key],
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
          **kwargs):
    super().__init__(**kwargs)

    if concat_num is not None:
      self.stm_corpus_key += "_concat-%d" % concat_num

    self.batch_size = batch_size
    self.concat_num = concat_num
    self.search_rqmt = search_rqmt
    self.load_ignore_missing_vars = load_ignore_missing_vars

    self.returnn_python_exe = self.config_builder.variant_params["returnn_python_exe"]
    self.returnn_root = self.config_builder.variant_params["returnn_root"]

  def get_recog_opts(self):
    return {
      "search_corpus_key": self.corpus_key,
      "batch_size": self.batch_size,
      "dataset_opts": {"concat_num": self.concat_num},
      "load_ignore_missing_vars": self.load_ignore_missing_vars,
    }

  def get_ctm_path(self) -> Path:
    recog_config = self.config_builder.get_recog_config(opts=self.get_recog_opts())

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
          plot_center_positions: bool = False,
          dump_att_weight_penalty: bool = False,
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

    for hdf_alias, hdf_targets in zip(
            ["ground_truth", "search"],
            [ground_truth_hdf, forward_search_job.out_default_hdf]
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
        plot_center_positions=plot_center_positions,
        dump_att_weight_penalty=dump_att_weight_penalty,
      )

    calc_search_errors(
      self.config_builder,
      variant_params=self.config_builder.variant_params,
      checkpoint=self.checkpoint,
      ground_truth_hdf_targets=ground_truth_hdf,
      search_hdf_targets=forward_search_job.out_default_hdf,
      corpus_key=self.corpus_key,
      alias=self.alias,
    )


# def run_recog(
#         config_builder: ConfigBuilder,
#         variant_params: Dict,
#         recog_opts_list: List[Dict],
#         checkpoint: Checkpoint,
#         alias: str,
# ):
#   recog_config_builder = copy.deepcopy(config_builder)
#
#   for recog_opts in recog_opts_list:
#     search_rqmt = recog_opts.pop("search_rqmt", None)
#     recog_config = recog_config_builder.get_recog_config(opts=recog_opts)
#     ReturnnDecodingExperiment(
#       returnn_config=recog_config,
#       variant_params=variant_params,
#       checkpoint=checkpoint,
#       corpus_key=recog_opts["search_corpus_key"],
#       search_rqmt=search_rqmt,
#       base_alias=alias,
#       concat_num=recog_opts.get("dataset_opts", {}).get("concat_num", None)
#     ).run_eval(use_hub5_score_job=(type(variant_params["dataset"]["corpus"]) == SWBCorpus))

    # if "center-window_att_global_ctc_align_diff_win_sizes_diff_epochs_diff_lrs" in alias:
    #   recog_config.config["network"]["decision"] = {
    #     "class": "copy",
    #     "from": "output_wo_b",
    #     "target": "targets",
    #     "is_output_layer": True
    #   }
    #   ReturnnDecodingExperiment(
    #     returnn_config=recog_config,
    #     variant_params=variant_params,
    #     checkpoint=train_job.out_checkpoints[n_epochs],
    #     corpus_key=recog_opts["search_corpus_key"],
    #     base_alias=alias + "n-best-test"
    #   ).run_eval(use_hub5_score_job=(type(variant_params["dataset"]["corpus"]) == SWBCorpus))

    # if "center-window_att_global_ctc_align_diff_win_sizes_diff_epochs_diff_lrs" in alias or "weight_feedback" in alias or "win-size-32" in alias:


# def run_analysis(
#         config_builder: ConfigBuilder,
#         variant_params: Dict,
#         ground_truth_hdf: Optional[Path],
#         att_weight_ref_alignment_hdf: Path,
#         att_weight_ref_alignment_blank_idx: int,
#         forward_recog_opts: Dict,
#         checkpoint: Checkpoint,
#         corpus_key: str,
#         alias: str,
# ):
#     forward_recog_opts = copy.deepcopy(forward_recog_opts)
#     forward_recog_opts["search_corpus_key"] = corpus_key
#     forward_recog_config = config_builder.get_recog_config_for_forward_job(opts=forward_recog_opts)
#     forward_search_job = ReturnnForwardJob(
#       model_checkpoint=checkpoint,
#       returnn_config=forward_recog_config,
#       returnn_root=variant_params["returnn_root"],
#       returnn_python_exe=variant_params["returnn_python_exe"],
#       eval_mode=False
#     )
#     forward_search_job.add_alias("%s/analysis/forward_recog_dump_seq" % alias)
#
#     for hdf_alias, hdf_targets in zip(
#             ["ground_truth", "search"],
#             [ground_truth_hdf, forward_search_job.out_default_hdf]
#     ):
#
#       dump_att_weights(
#         config_builder,
#         variant_params=variant_params,
#         checkpoint=checkpoint,
#         hdf_targets=hdf_targets,
#         ref_alignment=att_weight_ref_alignment_hdf,
#         corpus_key=corpus_key,
#         hdf_alias=hdf_alias,
#         alias=alias,
#         ref_alignment_blank_idx=att_weight_ref_alignment_blank_idx
#       )
#
#     calc_search_errors(
#       config_builder,
#       variant_params=variant_params,
#       checkpoint=checkpoint,
#       ground_truth_hdf_targets=ground_truth_hdf,
#       search_hdf_targets=forward_search_job.out_default_hdf,
#       corpus_key=corpus_key,
#       alias=alias,
#     )


# def run_train_recog(
#         config_builder: ConfigBuilder,
#         variant_params: Dict,
#         n_epochs: int,
#         train_opts: Dict,
#         recog_opts_list: List[Dict],
#         alias: str,
# ):
#   train_config_builder = copy.deepcopy(config_builder)
#   train_config = train_config_builder.get_train_config(opts=train_opts)
#
#   train_job = ReturnnTrainingJob(
#     train_config,
#     num_epochs=n_epochs,
#     keep_epochs=[n_epochs],
#     log_verbosity=5,
#     returnn_python_exe=variant_params["returnn_python_exe"],
#     returnn_root=variant_params["returnn_root"] if "positional_embedding" not in alias and "center-window_att_global_ctc_align_diff_win_sizes_diff_epochs_diff_lrs" not in alias else RETURNN_ROOT,
#     mem_rqmt=24,
#     time_rqmt=30)
#   train_job.add_alias(alias + "/train")
#   tk.register_output(train_job.get_one_alias() + "/models", train_job.out_model_dir)
#   tk.register_output(train_job.get_one_alias() + "/plot_lr", train_job.out_plot_lr)
#
#   recog_config_builder = copy.deepcopy(config_builder)
#
#   for recog_opts in recog_opts_list:
#     recog_config = recog_config_builder.get_recog_config(opts=recog_opts)
#     ReturnnDecodingExperiment(
#       returnn_config=recog_config,
#       variant_params=variant_params,
#       checkpoint=train_job.out_checkpoints[n_epochs],
#       corpus_key=recog_opts["search_corpus_key"],
#       base_alias=alias
#     ).run_eval(use_hub5_score_job=(type(variant_params["dataset"]["corpus"]) == SWBCorpus))
#
#     # if "center-window_att_global_ctc_align_diff_win_sizes_diff_epochs_diff_lrs" in alias:
#     #   recog_config.config["network"]["decision"] = {
#     #     "class": "copy",
#     #     "from": "output_wo_b",
#     #     "target": "targets",
#     #     "is_output_layer": True
#     #   }
#     #   ReturnnDecodingExperiment(
#     #     returnn_config=recog_config,
#     #     variant_params=variant_params,
#     #     checkpoint=train_job.out_checkpoints[n_epochs],
#     #     corpus_key=recog_opts["search_corpus_key"],
#     #     base_alias=alias + "n-best-test"
#     #   ).run_eval(use_hub5_score_job=(type(variant_params["dataset"]["corpus"]) == SWBCorpus))
#
#     # if "center-window_att_global_ctc_align_diff_win_sizes_diff_epochs_diff_lrs" in alias or "weight_feedback" in alias or "win-size-32" in alias:
#     if "center-window_att_global_ctc_align_weight_feedback/win-size-16" in alias or "center-window_att_global_ctc_align_weight_feedback_only-train-length-model/win-size-16" in alias or "center-window_att_global_ctc_align_diff_win_sizes_diff_epochs_diff_lrs/win-size-4_100-epochs_const-lr-0.000100" in alias:
#       forward_recog_config = copy.deepcopy(recog_config)
#       forward_recog_config.config.update({
#         "forward_use_search": True,
#         "forward_batch_size": CodeWrapper("batch_size")
#       })
#       forward_recog_config.config["network"]["dump_decision"] = {
#         "class": "hdf_dump",
#         "from": "decision",
#         "is_output_layer": True,
#         "filename": "search_out.hdf"
#       }
#       del forward_recog_config.config["task"]
#       forward_recog_config.config["eval"] = copy.deepcopy(train_config.config["dev"])
#       del forward_recog_config.config["search_data"]
#       forward_recog_config.config["network"]["output_w_beam"] = copy.deepcopy(forward_recog_config.config["network"]["output"])
#       forward_recog_config.config["network"]["output_w_beam"]["name_scope"] = "output/rec"
#       del forward_recog_config.config["network"]["output"]
#       forward_recog_config.config["network"]["output"] = copy.deepcopy(forward_recog_config.config["network"]["decision"])
#       forward_recog_config.config["network"]["output"]["from"] = "output_w_beam"
#       forward_recog_config.config["network"]["output_non_blank"]["from"] = "output_w_beam"
#       forward_recog_config.config["network"]["output_wo_b"]["from"] = "output_w_beam"
#       forward_search_job = ReturnnForwardJob(
#         model_checkpoint=train_job.out_checkpoints[n_epochs],
#         returnn_config=forward_recog_config,
#         returnn_root=variant_params["returnn_root"],
#         returnn_python_exe=variant_params["returnn_python_exe"],
#         hdf_outputs=["search_out.hdf"],
#         eval_mode=False
#       )
#
#       for hdf_alias, hdf_target in zip(
#               ["ground_truth", "search"],
#               [train_opts["dataset_opts"]["hdf_targets"]["cv"], forward_search_job.out_default_hdf]
#       ):
#         dump_att_weights_opts = copy.deepcopy(train_opts)
#         dump_att_weights_opts.update({
#           "dataset_opts": {
#             "hdf_targets": {"cv": hdf_target}
#           }
#         })
#
#         dump_att_weights(
#           config_builder,
#           variant_params=variant_params,
#           checkpoint=train_job.out_checkpoints[n_epochs],
#           opts=dump_att_weights_opts,
#           ref_alignment=train_opts["dataset_opts"]["hdf_targets"]["cv"],
#           corpus_key="cv",
#           alias=alias + "_%s" % hdf_alias,
#           use_search=False
#         )
#
#       dump_scores_opts_ground_truth = copy.deepcopy(train_opts)
#       dump_scores_opts_search = copy.deepcopy(train_opts)
#       dump_scores_opts_ground_truth.update({
#         "dataset_opts": {
#           "hdf_targets": {"cv": train_opts["dataset_opts"]["hdf_targets"]["cv"]}
#         }
#       })
#       dump_scores_opts_search.update({
#         "dataset_opts": {
#           "hdf_targets": {"cv": forward_search_job.out_default_hdf}
#         }
#       })
#       calc_search_errors(
#         config_builder,
#         variant_params=variant_params,
#         checkpoint=train_job.out_checkpoints[n_epochs],
#         dump_scores_opts_ground_truth=dump_scores_opts_ground_truth,
#         dump_scores_opts_search=dump_scores_opts_search,
#         corpus_key="cv",
#         alias=alias,
#       )
#
#     # calc_search_errors(
#     #   config_builder,
#     #   variant_params=variant_params,
#     #   checkpoint=train_job.out_checkpoints[n_epochs],
#     #   train_opts=train_opts,
#     #   alias=alias
#     # )
