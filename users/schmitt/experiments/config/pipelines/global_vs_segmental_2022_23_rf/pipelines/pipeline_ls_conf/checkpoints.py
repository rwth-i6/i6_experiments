from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints as external_checkpoints_tf
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

from sisyphus import tk, Path

external_checkpoints = {}

default_import_model_name = "glob.conformer.mohammad.5.4"

for checkpoint_name, checkpoint in external_checkpoints_tf.items():
  global_att_checkpoint = ConvertTfCheckpointToRfPtJob(
    checkpoint=checkpoint,
    make_model_func=MakeModelGlobal(
      in_dim=80,
      target_dim=10025,
    ),
    map_func=map_param_func_v2_global
  ).out_checkpoint

  external_checkpoints[checkpoint_name] = PtCheckpoint(global_att_checkpoint)

external_checkpoints.update({
  "albert-aed-trafo-decoder-bpe10k": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/model_checkpoints/segmental_models_2022_23_rf/v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-spm10k-spmSample07.498_converted.pt")),
  "luca-aed-bpe5k": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/model_checkpoints/segmental_models_2022_23_rf/luca-global-aed-no-ctc-bpe5k-100ep_converted.pt")),
  "luca-aed-bpe5k-w-ctc": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/model_checkpoints/segmental_models_2022_23_rf/luca-global-aed-w-ctc-bpe5k-100ep_converted.pt")),
  "luca-aed-bpe1k-w-ctc": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/model_checkpoints/segmental_models_2022_23_rf/luca-global-aed-w-ctc-bpe1k-100ep_converted.pt")),
  "luca-aed-bpe1k-w-ctc-w-aux-layers": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/model_checkpoints/segmental_models_2022_23_rf/luca-global-aed-w-ctc-bpe1k-100ep_converted_w_aux_layers.pt")),
  "luca-aed-bpe1k-wo-ctc": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/model_checkpoints/segmental_models_2022_23_rf/luca-global-aed-no-ctc-bpe1k-100ep_converted.pt")),
})

lm_checkpoints = {
  "kazuki-10k": "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_experiments/users/schmitt/returnn_frontend/convert/checkpoint/ConvertTfCheckpointToRfPtJob.7haAE0Cx93dA/output/model/network.023.pt",
}

# global_att_checkpoint_w_ctc = ConvertTfCheckpointToRfPtJob(
#   checkpoint=external_checkpoints_tf[default_import_model_name],
#   make_model_func=MakeModelGlobal(
#     in_dim=80,
#     target_dim=10025,
#     enc_aux_logits=(11,)
#   ),
#   map_func=map_param_func_v2_global
# ).out_checkpoint
# external_checkpoints[default_import_model_name + "_w_ctc"] = PtCheckpoint(global_att_checkpoint_w_ctc)


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
