from sisyphus import Path
from i6_core.returnn.training import Checkpoint

external_checkpoints = {
  "glob.conformer.mohammad.5.6": Checkpoint(Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/models-backup/best_att_100/avg_ckpt/epoch.2029.index", cached=True)),
  "glob.conformer.mohammad.5.4": Checkpoint(Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index", cached=True))
}

default_import_model_name = "glob.conformer.mohammad.5.6"
