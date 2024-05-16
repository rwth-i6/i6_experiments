from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints as external_checkpoints_tf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import default_import_model_name
from i6_experiments.users.schmitt.returnn_frontend.convert.checkpoint import ConvertTfCheckpointToRfPtJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model_import import map_param_func_v2 as map_param_func_v2_global
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import MakeModel as MakeModelGlobal
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import MakeModel as MakeModelSegmental
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_old.model_import import map_param_func_v2 as map_param_func_v2_segmental
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import (
  train
)

from i6_core.returnn.training import PtCheckpoint

from sisyphus import tk


global_att_checkpoint = ConvertTfCheckpointToRfPtJob(
  checkpoint=external_checkpoints_tf[default_import_model_name],
  make_model_func=MakeModelGlobal(
    in_dim=80,
    target_dim=10025,
  ),
  map_func=map_param_func_v2_global
).out_checkpoint

external_checkpoints = {
  default_import_model_name: PtCheckpoint(global_att_checkpoint)
}


def get_center_window_baseline_v1_tf_checkpoint():
  for model_alias, config_builder in baseline.center_window_att_baseline(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(1,),
    ):
      center_window_checkpoint = ConvertTfCheckpointToRfPtJob(
        checkpoint=checkpoint["checkpoints"][1],
        make_model_func=MakeModelSegmental(
          in_dim=80,
          align_target_dim=10026,
          target_dim=10025,
          center_window_size=5,
        ),
        map_func=map_param_func_v2_segmental
      ).out_checkpoint
      return PtCheckpoint(center_window_checkpoint)
