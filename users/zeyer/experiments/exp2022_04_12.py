"""
experiments
"""

from typing import Tuple, Optional
import numpy
from sisyphus import tk
from ..datasets import librispeech
# from i6_core.datasets.tf_datasets import DownloadAndPrepareTfDatasetJob
# from i6_core.datasets.huggingface import DownloadAndPrepareHuggingFaceDatasetJob
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_experiments.common.setups.returnn_common import serialization
from returnn_common import nn


def run():
  """run"""
  for name, path in librispeech.librispeech_ogg_zip_dict.items():
    tk.register_output(f"librispeech/dataset/{name}", path)

  tk.register_output("librispeech/sentencepiece-2k.model", librispeech.spm_2k)

  # tk.register_output(
  #  "librispeech/huggingface-dataset-clean",
  #  DownloadAndPrepareHuggingFaceDatasetJob("librispeech_asr", "clean").out_dir)

  input_dim = nn.FeatureDim("input", 80)
  time_dim = nn.SpatialDim("time")
  targets_time_dim = nn.SpatialDim("targets-time")
  output_dim = nn.FeatureDim("output", 2000)

  from returnn_common.asr import specaugment

  class Model(nn.Module):
    """model"""
    def __init__(self, out_dim: nn.Dim, conformer_dim: Optional[nn.Dim] = None, **kwargs):
      super(Model, self).__init__()
      # Medium size default...
      if conformer_dim is None:
        conformer_dim = nn.FeatureDim("conformer", 256)
      kwargs.setdefault("num_layers", 16)
      kwargs.setdefault("num_heads", 4)
      self.conformer = nn.ConformerEncoder(conformer_dim, **kwargs)
      self.out_dim = out_dim
      self.output = nn.Linear(out_dim)

    def __call__(self, x: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
      x = specaugment.specaugment_v2(x, spatial_dim=in_spatial_dim)
      x, out_spatial_dim_ = self.conformer(x, in_spatial_dim=in_spatial_dim)
      assert isinstance(out_spatial_dim_, nn.Dim)
      if out_spatial_dim_ != in_spatial_dim:
        out_spatial_dim_.declare_same_as(nn.SpatialDim("downsampled-time"))
      x = self.output(x)
      return x, out_spatial_dim_

  model = Model(out_dim=output_dim + 1)  # +1 for blank
  inputs = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))
  logits, out_spatial_dim = model(inputs, in_spatial_dim=time_dim)

  targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, targets_time_dim], sparse_dim=output_dim))
  loss = nn.ctc_loss(logits=logits, targets=targets)
  loss.mark_as_loss()

  decoded, decoded_spatial_dim = nn.ctc_greedy_decode(logits, in_spatial_dim=out_spatial_dim)
  error = nn.edit_distance(a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_time_dim)
  error.mark_as_loss(as_error=True, custom_inv_norm_factor=nn.length(targets, axis=targets_time_dim))
  model_py_code_str = nn.get_returnn_config().get_complete_py_code_str(model)

  returnn_train_config_dict = dict(
    use_tensorflow=True,
    # flat_net_construction=True,

    **librispeech.default_dataset_config,

    batching="random",
    batch_size=20000,
    max_seqs=200,
    max_seq_length={"classes": 75},

    gradient_clip=0,
    # gradient_clip_global_norm = 1.0
    optimizer={"class": "nadam", "epsilon": 1e-8},
    gradient_noise=0.0,
    learning_rate=0.0008,
    learning_rates=[0.0003] * 10 + list(numpy.linspace(0.0003, 0.0008, num=10)),
    learning_rate_control="newbob_multi_epoch",
    # learning_rate_control_error_measure = "dev_score_output"
    learning_rate_control_relative_error_relative_lr=True,
    learning_rate_control_min_num_epochs_per_new_lr=3,
    use_learning_rate_control_always=True,
    newbob_multi_num_epochs=librispeech.default_epoch_split,
    newbob_multi_update_interval=1,
    newbob_learning_rate_decay=0.9,
  )

  returnn_train_config = ReturnnConfig(
    returnn_train_config_dict,
    python_epilog=[serialization.Collection(
      [
        serialization.ExplicitHash("my_model"),
        serialization.PythonEnlargeStackWorkaroundCode,
        serialization.NonhashedCode(model_py_code_str),
      ]
    )],
    post_config=dict(
      log_batch_size=True,
      tf_log_memory_usage=True,
      tf_session_opts={"gpu_options": {"allow_growth": True}},
      cleanup_old_models=True,
      # debug_add_check_numerics_ops = True
      # debug_add_check_numerics_on_output = True
      # stop_on_nonfinite_train_score = False,
    ),
    sort_config=False,
  )
  returnn_train_job = ReturnnTrainingJob(
    returnn_train_config,
    log_verbosity=5, num_epochs=100,
    time_rqmt=80, mem_rqmt=15, cpu_rqmt=4)
  tk.register_output("librispeech/ctc-model/learning-rates", returnn_train_job.out_learning_rates)
