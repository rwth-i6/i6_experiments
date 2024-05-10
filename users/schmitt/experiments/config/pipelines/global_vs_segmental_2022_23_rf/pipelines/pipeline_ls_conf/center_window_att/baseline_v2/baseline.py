from typing import Tuple, Optional, List, Dict, Union
import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_CTC_ALIGNMENT,
)

LIBRISPEECH_CORPUS = LibrispeechCorpora()


def get_center_window_att_config_builder_rf(
        win_size: int,
        length_model_opts: Dict
) -> SegmentalAttConfigBuilderRF:
  variant_params = {
    "dependencies": LibrispeechBPE10025_CTC_ALIGNMENT,
    "dataset": {
      "feature_type": "raw",
      "corpus": LIBRISPEECH_CORPUS
    },
    "returnn_python_exe": RETURNN_EXE_NEW,
    "returnn_root": RETURNN_CURRENT_ROOT
  },

  config_builder = SegmentalAttConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    center_window_size=win_size,
    length_model_opts=length_model_opts,
  )

  return config_builder


def center_window_att_baseline_rf(
        win_size_list: Tuple[int, ...] = (5, 129),
):
  for win_size in win_size_list:
    alias = f"{base_alias}/baseline_rf/win-size-%d" % (
      win_size
    )
    yield alias, get_center_window_att_config_builder_rf(
      win_size=win_size,
      length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
    )
