from i6_experiments.users.schmitt.returnn_frontend.convert.checkpoint import ConvertTfCheckpointToRfPtJob

from .conformer_import_tina.fh_ctc import (
  map_param_func_v2 as map_param_func_v2_fh_ctc,
  map_param_func_v2_ctc,
  map_param_func_v2_post_hmm,
  map_param_func_v2_diphone_fh,
  map_param_func_v2_monophone_fh,
)
from .mohammad_aed_import import map_param_func_v2 as map_param_func_v2_mohammad_aed
from .conformer_import_tina.transducer import map_param_func_v2 as map_param_func_v2_transducer
from .conformer_tina import (
  MakeModel,
  MakeFramewiseProbModel,
  MakeDiphoneFHModel,
  MakeMonophoneFHModel,
  MakePhonTransducerModel,
)
from .aed import MakeModel as MakeAedModel
from ..configs import config_24gb_v1

from i6_core.returnn.training import PtCheckpoint, Checkpoint

from sisyphus import tk, Path

external_tf_checkpoints = {
  # "tina-phon-transducer_vit+fs-35-ep": Checkpoint(Path("/work/asr3/raissi/seq2seq_sisyphus_work_folders/work_bpe_gruev/crnn/custom_sprint_training/CustomCRNNSprintTrainingJob.VLyuOJ1PW9Yn/output/models/epoch.300.index")),
  "tina-ctc_fs-100-ep": Checkpoint(Path("/u/raissi/setups/librispeech/BPE_transducer/work/crnn/custom_sprint_training/CustomCRNNSprintTrainingJob.vWsTR7TuOhLb/output/models/epoch.2000.index")),
  "tina-monoph-post-hmm_fs-25-ep": Checkpoint(Path("/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_experiments/users/raissi/costum/returnn/rasr_returnn_bw/ReturnnRasrTrainingBWJob.PPofezMKIPIb/output/models/epoch.500.index")),
  "tina-monoph-post-hmm_fs-50-ep": Checkpoint(Path("/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_experiments/users/raissi/costum/returnn/rasr_returnn_bw/ReturnnRasrTrainingBWJob.eGKwH6rekwMZ/output/models/epoch.1000.index")),
  "tina-monoph-fh_fs-50-ep": Checkpoint(Path("/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_experiments/users/raissi/costum/returnn/rasr_returnn_bw/ReturnnRasrTrainingBWJob.4zAslTmyAn43/output/models/epoch.998.index")),
  "tina-diph-fh_viterbi-20-ep": Checkpoint(Path("/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_experiments/users/raissi/costum/returnn/rasr_returnn_vit/ReturnnRasrTrainingVITJob.A5ctZ7yM0OeU/output/models/epoch.390.index")),
  "tina-diph-fh_vit+fs-35-ep": Checkpoint(Path("/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_experiments/users/raissi/costum/returnn/rasr_returnn_bw/ReturnnRasrTrainingBWJob.Fxqx6AA8yScR/output/models/epoch.291.index")),
  "mohammad-aed-5.4": Checkpoint(Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"))
}

external_torch_checkpoints = {
  # all are trained on single 24gb GPU for 100 epochs
  "aed_1k-bpe": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.eoGg1OAu9UaY/output/models/epoch.2000.pt")),
  "aed_1k-bpe-ctc": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.yky2HwQUYYUv/output/models/epoch.2000.pt")),
  # all transducers use CTC aux loss
  "context-1-transducer_1k-bpe_full-sum": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.c8QhEQnASBnD/output/models/epoch.800.pt")),
  "context-1-transducer_1k-bpe_fixed-path": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.Jxuq93TCFcBv/output/models/epoch.1200.pt")),
  "full-context-transducer_1k-bpe_fixed-path": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.xiinWxFR4ZUK/output/models/epoch.1200.pt")),
  "full-context-transducer_1k-bpe_full-sum": PtCheckpoint(Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.6HvxwoXK7mFE/output/models/epoch.800.pt")),
}


def convert_checkpoints():
  for checkpoint_name, checkpoint in external_tf_checkpoints.items():
    if checkpoint_name in ["tina-phon-transducer_vit+fs-35-ep"]:
      map_func = map_param_func_v2_transducer
      make_model = MakePhonTransducerModel
      target_dim = 79
      in_dim = 50
      extra = {}
    elif "ctc" in checkpoint_name:
      map_func = map_param_func_v2_ctc
      make_model = MakeFramewiseProbModel
      target_dim = 79
      in_dim = 50
      extra = {}
    elif "post-hmm" in checkpoint_name:
      map_func = map_param_func_v2_post_hmm
      make_model = MakeFramewiseProbModel
      target_dim = 84
      in_dim = 50
      extra = {}
    elif checkpoint_name in ["tina-diph-fh_vit+fs-35-ep", "tina-diph-fh_viterbi-20-ep"]:
      map_func = map_param_func_v2_diphone_fh
      make_model = MakeDiphoneFHModel
      target_dim = 84
      in_dim = 50
      extra = {"left_target_dim": 42}
    elif checkpoint_name == "tina-monoph-fh_fs-50-ep":
      map_func = map_param_func_v2_monophone_fh
      make_model = MakeMonophoneFHModel
      target_dim = 84
      in_dim = 50
      extra = {}
    elif checkpoint_name == "mohammad-aed-5.4":
      map_func = map_param_func_v2_mohammad_aed
      make_model = MakeAedModel
      target_dim = 10_025
      in_dim = 80  # log mel on the fly
      extra = dict(
        encoder_opts=config_24gb_v1["model_opts"]["encoder_opts"],
        decoder_opts=config_24gb_v1["model_opts"]["decoder_opts"],
        enc_aux_logits=(12,)
      )
    else:
      map_func = map_param_func_v2_fh_ctc
      make_model = MakeModel
      target_dim = 10_025
      in_dim = 50
      extra = {}

    torch_checkpoint_job = ConvertTfCheckpointToRfPtJob(
      checkpoint=checkpoint,
      make_model_func=make_model(
        in_dim=in_dim,
        target_dim=target_dim,
        **extra,
      ),
      map_func=map_func,
    )
    torch_checkpoint_job.add_alias(f"convert_checkpoints/{checkpoint_name}")

    torch_checkpoint = torch_checkpoint_job.out_checkpoint
    tk.register_output(torch_checkpoint_job.get_one_alias(), torch_checkpoint)

    external_torch_checkpoints[checkpoint_name] = PtCheckpoint(torch_checkpoint)
