from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.returnn_frontend.convert.checkpoint import ConvertTfCheckpointToRfPtJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_import import map_param_func_v2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import MakeModel

from i6_core.returnn.training import PtCheckpoint

from sisyphus import tk


global_att_checkpoint = ConvertTfCheckpointToRfPtJob(
  checkpoint=external_checkpoints[default_import_model_name],
  make_model_func=MakeModel(
    in_dim=80,
    target_dim=10025,
  ),
  map_func=map_param_func_v2
).out_checkpoint
tk.register_output("torch_checkpoint_%s" % default_import_model_name, global_att_checkpoint)

external_checkpoints = {
  default_import_model_name: PtCheckpoint(global_att_checkpoint)
}
