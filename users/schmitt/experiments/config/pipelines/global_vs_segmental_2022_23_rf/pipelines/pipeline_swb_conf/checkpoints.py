from i6_experiments.users.schmitt.returnn_frontend.convert.checkpoint import ConvertTfCheckpointToRfPtJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model_import import map_param_func_v2 as map_param_func_v2_global
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import MakeModel as MakeModelGlobal

from i6_core.returnn.training import PtCheckpoint, Checkpoint

from sisyphus import tk, Path

external_checkpoints = {}

default_import_model_name = "glob.conformer.mohammad.5.4"

external_checkpoints_tf = {
  "glob.conformer.mohammad": Checkpoint(Path(
    "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.AQFR9zmXqUo7/output/model/average.index"
  ))
}

def py():
  for checkpoint_name, checkpoint in external_checkpoints_tf.items():
    global_att_checkpoint_job = ConvertTfCheckpointToRfPtJob(
      checkpoint=checkpoint,
      make_model_func=MakeModelGlobal(
        in_dim=50,
        target_dim=534,
        target_embed_dim=256,
        encoder_layer_opts={
          "self_att_opts": {
            "with_pos_bias": False,
            "with_bias": False,
            "with_linear_pos": False,
          },
        }
      ),
      map_func=map_param_func_v2_global
    )
    tk.register_output("swb_aed_checkpoints/" + checkpoint_name, global_att_checkpoint_job.out_checkpoint)
