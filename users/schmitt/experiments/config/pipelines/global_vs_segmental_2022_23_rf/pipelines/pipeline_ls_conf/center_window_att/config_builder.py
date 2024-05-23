from typing import Tuple, Optional, List, Dict, Union

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_CTC_ALIGNMENT,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import from_scratch_model_def, _returnn_v2_get_model


def get_center_window_att_config_builder_rf(
        win_size: int,
        label_decoder_version: int,
        blank_decoder_version: Optional[int],
        use_joint_model: bool,
        use_weight_feedback: bool = True,
) -> SegmentalAttConfigBuilderRF:
  variant_params = {
    "dependencies": LibrispeechBPE10025_CTC_ALIGNMENT,
    "dataset": {
      "feature_type": "raw",
      "corpus": LIBRISPEECH_CORPUS
    },
    "config": {
      "train_seq_ordering": "laplace:.1000"
    },
    "network": {"length_scale": 1.0},
    "returnn_python_exe": RETURNN_EXE_NEW,
    "returnn_root": RETURNN_CURRENT_ROOT
  }

  config_builder = SegmentalAttConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    center_window_size=win_size,
    label_decoder_version=label_decoder_version,
    blank_decoder_version=blank_decoder_version,
    use_joint_model=use_joint_model,
    use_weight_feedback=use_weight_feedback,
  )

  return config_builder
