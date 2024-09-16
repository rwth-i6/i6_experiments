import copy

from i6_core.returnn.training import PtCheckpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.ctc.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import recog
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.ctc import (
  realign, train
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT

from sisyphus import Path

def run_exps():
  for model_alias, config_builder in baseline.ctc_baseline_rf(
          label_type="bpe1056",
          num_layers=12,
  ):
    for train_alias, checkpoint in train.train_ctc(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=100,
      checkpoint_path=external_checkpoints["luca-aed-bpe1k-w-ctc"],
      checkpoint_alias="luca-aed-bpe1k-w-ctc",
      lr_scheduling_type="const_then_linear",
      use_mgpu=False,
      ce_aux_loss_layers=(4, 8, 12),
      use_curriculum_learning=False,
    ):
      pass


def register_ctc_alignments():
  for model_alias, config_builder in baseline.ctc_baseline_rf(
          label_type="bpe1056",
          num_layers=8,
  ):
    for train_alias, checkpoint in (
            (f"{model_alias}/import_glob.conformer.luca.bpe1k.w-ctc",
             external_checkpoints["luca-aed-bpe1k-w-ctc-w-aux-layers"]),
    ):
      alignments = {}
      for corpous_key in ("train", "cv"):
        alignments[corpous_key] = realign.ctc_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_alias="best",
          plot=False,
          batch_size=50_000,
          time_rqmt=4,
          corpus_key=corpous_key,
          concurrent=1,
        )
      alignments["devtrain"] = alignments["train"]
      LibrispeechBPE1056_CTC_ALIGNMENT.alignment_paths = copy.deepcopy(alignments)


def train_from_scratch():
  for model_alias, config_builder in baseline.ctc_baseline_rf(
          label_type="bpe1056",
          num_layers=12,
  ):
    for train_alias, checkpoint in train.train_ctc(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
      use_mgpu=True,
      ce_aux_loss_layers=(12,),
      filter_data_len=19.5 * 16_000,
      batch_size=15_000,
      accum_grad_multiple_step=4,
      gpu_mem_rqmt=11,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in [60]:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=PtCheckpoint(Path("/work/smt4/thulke/schmitt/sisyphus_work_dirs/03-09-24_aed_flipped_encoder/i6_core/returnn/training/ReturnnTrainingJob.ek6yr1nCX3B2/output/models/epoch.067_.pt")),
        checkpoint_aliases=(f"epoch-{67}",),
        run_analysis=True,
        analysis_dump_gradients=True,
        only_do_analysis=True,
        corpus_keys=("train",),
        att_weight_seq_tags=None,
      )
