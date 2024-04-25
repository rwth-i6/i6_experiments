from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v2 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import from_scratch_model_def, from_scratch_training
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import MakeModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_import import map_param_func_v2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import _returnn_v2_get_model, _returnn_v2_train_step

from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob
# from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf._moh_att_2023_06_30_import import map_param_func_v2

from i6_core.returnn.training import PtCheckpoint, Checkpoint

from sisyphus import Path


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    train_alias = model_alias + "/recog_from_tf_checkpoint"
    checkpoint = ConvertTfCheckpointToRfPtJob(
      checkpoint=Checkpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/training/ReturnnTrainingJob.bcfsn3yxd8VV/output/models/epoch.001.index")),
      make_model_func=MakeModel(
        in_dim=80,
        target_dim=10025,
        align_target_dim=10026,
      ),
      map_func=map_param_func_v2
    ).out_checkpoint
    checkpoint = PtCheckpoint(checkpoint)

    recog.center_window_returnn_frame_wise_beam_search(
      alias=train_alias,
      config_builder=config_builder,
      checkpoint=checkpoint,
      checkpoint_aliases=("last",)
    )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(10,),
      time_rqmt=4
    ):
      # recog.center_window_returnn_frame_wise_beam_search(
      #   alias=train_alias,
      #   config_builder=config_builder,
      #   checkpoint=checkpoint,
      #   checkpoint_aliases=("last",)
      # )
      pass
