from typing import Tuple, Optional, List, Dict, Union

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_CTC_ALIGNMENT,
  LibrispeechBPE1056_ALIGNMENT_JOINT_MODEL,
  LibrispeechBPE1056_ALIGNMENT_SEP_MODEL,
  LibrispeechBPE5048_ALIGNMENT_JOINT_MODEL,
  LibrispeechBPE5048_ALIGNMENT_SEP_MODEL,
  LibrispeechSP10240_ALIGNMENT_SEP_MODEL,
  LibrispeechSP10240_ALIGNMENT_JOINT_MODEL,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechSegmentalAttConformerConfigBuilderRF
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
        separate_blank_from_softmax: bool = False,
        blank_decoder_opts: Optional[Dict] = None,
        use_current_frame_in_readout: bool = False,
        use_current_frame_in_readout_w_gate: bool = False,
        use_current_frame_in_readout_random: bool = False,
        use_correct_dim_tags: bool = False,
        target_embed_dimension: int = 640,
        readout_dimension: int = 1024,
        ilm_dimension: int = 1024,
        use_trafo_att: bool = False,
        behavior_version: Optional[int] = None,
        use_readout: bool = True,
        use_trafo_att_wo_cross_att: bool = False,
) -> Tuple[str, LibrispeechSegmentalAttConformerConfigBuilderRF]:
  assert bpe_vocab_size in {10025, 1056, 5048, 10240}

  if bpe_vocab_size == 10025:
    dependencies = LibrispeechBPE10025_CTC_ALIGNMENT
  elif bpe_vocab_size == 1056:
    if use_joint_model:
      dependencies = LibrispeechBPE1056_ALIGNMENT_JOINT_MODEL
    else:
      dependencies = LibrispeechBPE1056_ALIGNMENT_SEP_MODEL
  elif bpe_vocab_size == 5048:
    if use_joint_model:
      dependencies = LibrispeechBPE5048_ALIGNMENT_JOINT_MODEL
    else:
      dependencies = LibrispeechBPE5048_ALIGNMENT_SEP_MODEL
  else:
    if use_joint_model:
      dependencies = LibrispeechSP10240_ALIGNMENT_JOINT_MODEL
    else:
      dependencies = LibrispeechSP10240_ALIGNMENT_SEP_MODEL

  variant_params = {
    "dependencies": dependencies,
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

  config_builder = LibrispeechSegmentalAttConformerConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=get_model_func,
    center_window_size=win_size,
    use_att_ctx_in_state=use_att_ctx_in_state,
    blank_decoder_version=blank_decoder_version,
    use_joint_model=use_joint_model,
    use_weight_feedback=use_weight_feedback,
    label_decoder_state=label_decoder_state,
    gaussian_att_weight_opts=gaussian_att_weight_opts,
    separate_blank_from_softmax=separate_blank_from_softmax,
    blank_decoder_opts=blank_decoder_opts,
    use_current_frame_in_readout=use_current_frame_in_readout,
    use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
    use_current_frame_in_readout_random=use_current_frame_in_readout_random,
    use_correct_dim_tags=use_correct_dim_tags,
    target_embed_dimension=target_embed_dimension,
    readout_dimension=readout_dimension,
    ilm_dimension=ilm_dimension,
    use_trafo_att=use_trafo_att,
    behavior_version=behavior_version,
    use_readout=use_readout,
    use_trafo_att_wo_cross_att=use_trafo_att_wo_cross_att
  )

  alias = (
    f"behavior-{behavior_version}/"
    f"bpe-size-{bpe_vocab_size}/"
    f"win-size-{win_size}{'_gaussian-std-' + str(gaussian_att_weight_opts['std']) if gaussian_att_weight_opts else ''}/"
    f"{'w' if use_weight_feedback else 'wo'}-wf_"
    f"{'w' if use_att_ctx_in_state else 'wo'}-ctx-in-s/"
    f"bd-{blank_decoder_version}/"
    f"{label_decoder_state}{'_trafo-att' if use_trafo_att else ''}{'-wo-cross-att' if use_trafo_att_wo_cross_att else ''}/"
    f"{'_cur_frame_in_readout' if use_current_frame_in_readout else ''}"
    f"{'_cur_frame_in_readout_w_gate' if use_current_frame_in_readout_w_gate else ''}"
    f"{'_cur_frame_in_readout_random' if use_current_frame_in_readout_random else ''}"
    f"/{'sep-blank-sigmoid' if separate_blank_from_softmax or blank_decoder_version is not None else 'blank-in-softmax'}"
  )

  return alias, config_builder
