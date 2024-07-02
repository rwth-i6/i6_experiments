from typing import Tuple, Optional, List, Dict, Union

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_CTC_ALIGNMENT,
  LibrispeechBPE1056_ALIGNMENT,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import from_scratch_model_def, _returnn_v2_get_model, _returnn_v2_get_joint_model


def get_center_window_att_config_builder_rf(
        win_size: int,
        use_att_ctx_in_state: bool,
        blank_decoder_version: Optional[int],
        use_joint_model: bool,
        use_weight_feedback: bool = True,
        label_decoder_state: str = "nb-lstm",
        bpe_vocab_size: int = 10025,
        gaussian_att_weight_opts: Optional[Dict] = None,
) -> Tuple[str, SegmentalAttConfigBuilderRF]:
  assert bpe_vocab_size in {10025, 1056}
  variant_params = {
    "dependencies": LibrispeechBPE10025_CTC_ALIGNMENT if bpe_vocab_size == 10025 else LibrispeechBPE1056_ALIGNMENT,
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

  if use_joint_model:
    get_model_func = _returnn_v2_get_joint_model
  else:
    get_model_func = _returnn_v2_get_model

  config_builder = SegmentalAttConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=get_model_func,
    center_window_size=win_size,
    use_att_ctx_in_state=use_att_ctx_in_state,
    blank_decoder_version=blank_decoder_version,
    use_joint_model=use_joint_model,
    use_weight_feedback=use_weight_feedback,
    label_decoder_state=label_decoder_state,
    gaussian_att_weight_opts=gaussian_att_weight_opts
  )

  alias = (
    f"bpe-size-{bpe_vocab_size}/"
    f"win-size-{win_size}/"
    f"{'w' if use_weight_feedback else 'wo'}-wf_"
    f"{'w' if use_att_ctx_in_state else 'wo'}-ctx-in-s/"
    f"bd-{blank_decoder_version}/"
    f"{label_decoder_state}"
  )

  return alias, config_builder
