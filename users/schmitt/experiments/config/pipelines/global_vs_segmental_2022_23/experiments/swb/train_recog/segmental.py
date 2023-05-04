from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import AlignmentComparer, SegmentalAttentionWeightsPlotter
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.training import TrainExperiment, SegmentalTrainExperiment, GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.segmental import run_returnn_simple_segmental_decoding, run_rasr_segmental_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.realignment import run_rasr_segmental_realignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.train_recog.base import TrainRecogPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb import default_tags_for_analysis

from i6_core.returnn.training import Checkpoint

from abc import abstractmethod, ABC
from typing import Dict, List, Type, Optional, Tuple
from sisyphus import Path


class SegmentalTrainRecogPipeline(TrainRecogPipeline):
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          rasr_recog_epochs: Optional[Tuple] = None,
          realignment_length_scale: float = 1.,
          num_retrain: int = 0,
          retrain_load_checkpoint: bool = False,
          import_model_do_initial_realignment: bool = False,
          import_model_is_global: bool = False,
          **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

    self.rasr_recog_epochs = rasr_recog_epochs if rasr_recog_epochs is not None else (self.num_epochs[-1],)
    for epoch in self.rasr_recog_epochs:
      assert epoch in self.num_epochs, "Cannot do RETURNN recog on epoch %d because it is not set in num_epochs"

    self.realignment_length_scale = realignment_length_scale
    self.num_retrain = num_retrain

    self.alignments = {
      "train": {
        "cv": self.dependencies.alignment_paths["cv"], "train": self.dependencies.alignment_paths["train"]
      }
    }

    self.retrain_load_checkpoint = retrain_load_checkpoint

    assert not import_model_do_initial_realignment or self.import_model_train_epoch1 is not None, "Doing an initial realignment when not importing a model won't work"
    self.import_model_do_initial_realignment = import_model_do_initial_realignment
    if import_model_do_initial_realignment:
      self.base_alias = "%s_initial_realignment" % self.base_alias

    assert not import_model_is_global or self.import_model_train_epoch1 is not None, "Setting 'import_model_is_global' does not have an effect when not importing a model"
    self.import_model_is_global = import_model_is_global

  def compare_alignments(
          self,
          hdf_align_path1: Path,
          align1_name: str,
          hdf_align_path2: Path,
          align2_name: str,
          align_alias: str):
    base_alias = "%s/%s" % (self.base_alias, align_alias)
    AlignmentComparer(
      hdf_align_path1=hdf_align_path1,
      blank_idx1=self.dependencies.model_hyperparameters.blank_idx,
      name1=align1_name,
      vocab_path1=self.dependencies.vocab_path,
      hdf_align_path2=hdf_align_path2,
      blank_idx2=self.dependencies.model_hyperparameters.blank_idx,
      name2=align2_name,
      vocab_path2=self.dependencies.vocab_path,
      seq_tags=default_tags_for_analysis,
      corpus_key="cv",
      base_alias=base_alias).run()

  def plot_att_weights(
          self,
          hdf_align_path: Path,
          align_alias: str,
          checkpoint: Checkpoint,
  ):
    base_alias = "%s/%s" % (self.base_alias, align_alias)
    for seq_tag in default_tags_for_analysis:
      SegmentalAttentionWeightsPlotter(
        dependencies=self.dependencies,
        checkpoint=checkpoint,
        corpus_key="cv",
        seq_tag=seq_tag,
        hdf_target_path=hdf_align_path,
        hdf_alias=align_alias,
        variant_params=self.variant_params,
        base_alias=base_alias,
        length_scale=1.0).run()

  def run_training(
          self,
          train_alias: str = "train",
          cv_alignment: Path = None,
          train_alignment: Path = None,
          import_model_train_epoch1=None) -> Dict[int, Checkpoint]:
    base_alias = "%s/%s" % (self.base_alias, train_alias)
    return SegmentalTrainExperiment(
      dependencies=self.dependencies,
      variant_params=self.variant_params,
      num_epochs=self.num_epochs,
      base_alias=base_alias,
      cv_alignment=cv_alignment,
      train_alignment=train_alignment,
      import_model_train_epoch1=import_model_train_epoch1).run_training()

  def _get_realignment(
          self,
          corpus_key: str,
          epoch: int,
          checkpoint: Checkpoint,
          length_scale: float,
          train_alias: str,
          remove_length_model: bool = False) -> Path:
    base_alias = "%s/%s/epoch_%d" % (self.base_alias, train_alias, epoch)
    return run_rasr_segmental_realignment(
      dependencies=self.dependencies,
      variant_params=self.variant_params,
      checkpoint=checkpoint,
      corpus_key=corpus_key,
      base_alias=base_alias,
      length_scale=length_scale,
      remove_length_model=remove_length_model)

  def run_standard_recog(self, checkpoints: Dict[int, Checkpoint], train_alias: str):
    for epoch in self.returnn_recog_epochs:
      checkpoint = checkpoints[epoch]
      base_alias = "%s/%s/epoch_%d/standard_recog" % (self.base_alias, train_alias, epoch)

      variant_params = self._remove_pretrain_from_config(epoch=epoch)

      run_returnn_simple_segmental_decoding(
        dependencies=self.dependencies,
        variant_params=variant_params,
        base_alias=base_alias,
        checkpoint=checkpoint,
        test_corpora_keys=["dev"],
        use_recomb=False,
        calc_search_errors=True,
        search_error_corpus_key="cv",
        cv_realignment=self._get_realignment(
          corpus_key="cv", checkpoint=checkpoint, length_scale=1., epoch=epoch, train_alias=train_alias))

    for epoch in self.rasr_recog_epochs:
      checkpoint = checkpoints[epoch]
      base_alias = "%s/%s/epoch_%d/standard_recog" % (self.base_alias, train_alias, epoch)

      variant_params = self._remove_pretrain_from_config(epoch=epoch)

      run_rasr_segmental_decoding(
        dependencies=self.dependencies,
        variant_params=variant_params,
        base_alias=base_alias,
        checkpoint=checkpoint,
        test_corpora_keys=["dev"],
        calc_search_errors=True,
        search_error_corpus_key="cv",
        label_pruning_limit=12,
        word_end_pruning_limit=12,
        max_segment_len=20,
        concurrent=4,
        cv_realignment=self._get_realignment(
          corpus_key="cv", checkpoint=checkpoint, length_scale=1., epoch=epoch, train_alias=train_alias))

  def run_returnn_recog_w_recomb(self, checkpoints: Dict[int, Checkpoint], train_alias: str):
    for epoch in self.returnn_recog_epochs:
      checkpoint = checkpoints[epoch]
      base_alias = "%s/%s/epoch_%d/returnn_w_recomb" % (self.base_alias, train_alias, epoch)

      variant_params = self._remove_pretrain_from_config(epoch=epoch)

      run_returnn_simple_segmental_decoding(
        dependencies=self.dependencies,
        variant_params=variant_params,
        base_alias=base_alias,
        checkpoint=checkpoint,
        test_corpora_keys=["dev"],
        use_recomb=True,
        calc_search_errors=True,
        search_error_corpus_key="cv",
        cv_realignment=self._get_realignment(
          corpus_key="cv", checkpoint=checkpoint, length_scale=1., epoch=epoch, train_alias=train_alias))

  def run_rasr_recog_wo_length_model(self, checkpoints: Dict[int, Checkpoint], train_alias: str):
    for epoch in self.rasr_recog_epochs:
      checkpoint = checkpoints[epoch]
      base_alias = "%s/%s/epoch_%d/rasr_recog_wo_length_model" % (self.base_alias, train_alias, epoch)

      variant_params = self._remove_pretrain_from_config(epoch=epoch)

      run_rasr_segmental_decoding(
        dependencies=self.dependencies,
        variant_params=variant_params,
        base_alias=base_alias,
        checkpoint=checkpoint,
        test_corpora_keys=["dev"],
        calc_search_errors=True,
        search_error_corpus_key="cv",
        label_pruning_limit=12,
        word_end_pruning_limit=12,
        max_segment_len=20,
        concurrent=4,
        length_scale=0.,
        length_norm=True,
        cv_realignment=self._get_realignment(
          corpus_key="cv",
          checkpoint=checkpoint,
          length_scale=0.,
          epoch=epoch,
          train_alias=train_alias))

  def run_huge_beam_recog(self, checkpoints: Dict[int, Checkpoint], train_alias: str):
    last_epoch, last_checkpoint = list(checkpoints.items())[-1]
    base_alias = "%s/%s/epoch_%d/huge_beam_recog" % (self.base_alias, train_alias, last_epoch)

    run_rasr_segmental_decoding(
      dependencies=self.dependencies,
      variant_params=self.variant_params,
      base_alias=base_alias,
      checkpoint=last_checkpoint,
      test_corpora_keys=["dev400"],
      calc_search_errors=True,
      search_error_corpus_key="cv300",
      label_pruning_limit=3000,
      word_end_pruning_limit=3000,
      max_segment_len=20,
      plot_att_weights=False,
      compare_alignments=False,
      concurrent=8,
      mem_rqmt=30,
      time_rqmt=24,
      length_scale=0.,
      length_norm=True,
      cv_realignment=self._get_realignment(
        corpus_key="cv", checkpoint=last_checkpoint, length_scale=0., epoch=last_epoch, train_alias=train_alias))

  def run_recog(self, checkpoints: Dict[int, Checkpoint], train_alias: str = "train"):
    if self.recog_type == "standard":
      self.run_standard_recog(checkpoints=checkpoints, train_alias=train_alias)
    elif self.recog_type == "returnn_w_recomb":
      self.run_returnn_recog_w_recomb(checkpoints=checkpoints, train_alias=train_alias)
    elif self.recog_type == "rasr_wo_length_model":
      self.run_rasr_recog_wo_length_model(checkpoints=checkpoints, train_alias=train_alias)
    elif self.recog_type == "huge_beam":
      self.run_huge_beam_recog(checkpoints=checkpoints, train_alias=train_alias)
    else:
      raise NotImplementedError

  def run(self):
    if self.import_model_do_initial_realignment:
      for corpus_key in ("cv", "train"):
        self.alignments["train"][corpus_key] = self._get_realignment(
          corpus_key=corpus_key,
          checkpoint=self.import_model_train_epoch1,
          length_scale=self.realignment_length_scale,
          epoch=self.num_epochs[-1],
          train_alias="import_model",
          remove_length_model=self.import_model_is_global)

    train_alias = "train"
    self.checkpoints["train"] = self.run_training(
      import_model_train_epoch1=self.import_model_train_epoch1,
      train_alias=train_alias,
      cv_alignment=self.alignments["train"]["cv"],
      train_alignment=self.alignments["train"]["train"]
    )
    if self.do_recog:
      self.run_recog(checkpoints=self.checkpoints["train"])

    self.compare_alignments(
      hdf_align_path1=self.alignments["train"]["cv"],
      align1_name="ground-truth",
      hdf_align_path2=self.alignments["train"]["cv"],
      align2_name="ground-truth",
      align_alias="train/ground-truth-alignment")

    for retrain_iter in range(self.num_retrain):
      cur_train_alias = "train" if retrain_iter == 0 else ("retrain%d_realign-epoch%d_realign-length-scale%0.1f" % (retrain_iter, self.num_epochs[-1], self.realignment_length_scale))
      next_train_alias = "retrain%d_realign-epoch%d_realign-length-scale%0.1f" % (retrain_iter + 1, self.num_epochs[-1], self.realignment_length_scale)

      self.alignments[next_train_alias] = {}
      for corpus_key in ("cv", "train"):
        self.alignments[next_train_alias][corpus_key] = self._get_realignment(
          corpus_key=corpus_key,
          checkpoint=self.checkpoints[cur_train_alias][self.num_epochs[-1]],
          length_scale=self.realignment_length_scale,
          epoch=self.num_epochs[-1],
          train_alias=cur_train_alias)
      self.checkpoints[next_train_alias] = self.run_training(
        train_alias=next_train_alias,
        cv_alignment=self.alignments[next_train_alias]["cv"],
        train_alignment=self.alignments[next_train_alias]["train"],
        import_model_train_epoch1=self.checkpoints[cur_train_alias][self.num_epochs[-1]] if self.retrain_load_checkpoint else None)

      self.compare_alignments(
        hdf_align_path1=self.alignments[cur_train_alias]["cv"],
        align1_name=cur_train_alias,
        hdf_align_path2=self.alignments[next_train_alias]["cv"],
        align2_name=next_train_alias,
        align_alias="%s/compare-to-prev-align" % (next_train_alias,))

      if self.do_recog:
        self.run_recog(checkpoints=self.checkpoints[next_train_alias], train_alias=next_train_alias)
