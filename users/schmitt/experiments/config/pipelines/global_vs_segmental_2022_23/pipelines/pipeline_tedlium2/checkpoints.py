from sisyphus import Path
from i6_core.returnn.training import Checkpoint

external_checkpoints = {
  "aed_wo_ctc": Checkpoint(Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/GetBestTFCheckpointJob.mqLz8o9l75TG/output/model/checkpoint.index")),
  "aed_wo_ctc_mon": Checkpoint(Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.lNBuvo2Scanq/output/model/average.index")),
  "aed_w_ctc": Checkpoint(Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.yB4JK4GDCxWG/output/model/average.index")),
  "aed_w_ctc_mon": Checkpoint(Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.f5OcarXGvRrO/output/model/average.index")),
}
