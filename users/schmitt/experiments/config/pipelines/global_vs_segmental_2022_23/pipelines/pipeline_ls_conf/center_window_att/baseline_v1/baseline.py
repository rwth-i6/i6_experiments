from typing import Tuple, Optional, List, Dict, Union
import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models


def get_center_window_att_config_builder(
        win_size: int,
        use_weight_feedback: bool = True,
        use_positional_embedding: bool = False,
        att_weight_recog_penalty_opts: Optional[Dict] = None,
        length_model_opts: Optional[Dict] = None,
        length_scale: float = 1.0,
        blank_penalty: Union[float, str] = 0.0,
        gaussian_att_weight_interpolation_opts: Optional[Dict] = None,
        expected_position_aux_loss_opts: Optional[Dict] = None,
        pos_pred_att_weight_interpolation_opts: Optional[Dict] = None,
        search_remove_eos: bool = False,
        decoder_version: Optional[int] = None,
):
  model_type = "librispeech_conformer_seg_att"
  variant_name = "seg.conformer.like-global"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["segment_center_window_size"] = win_size
  variant_params["network"]["use_weight_feedback"] = use_weight_feedback
  variant_params["network"]["use_positional_embedding"] = use_positional_embedding
  variant_params["network"]["att_weight_recog_penalty_opts"] = att_weight_recog_penalty_opts
  variant_params["network"]["gaussian_att_weight_interpolation_opts"] = gaussian_att_weight_interpolation_opts
  variant_params["network"]["pos_pred_att_weight_interpolation_opts"] = pos_pred_att_weight_interpolation_opts
  variant_params["network"]["expected_position_aux_loss_opts"] = expected_position_aux_loss_opts
  variant_params["network"]["length_scale"] = length_scale
  variant_params["network"]["blank_penalty"] = blank_penalty
  variant_params["network"]["search_remove_eos"] = search_remove_eos
  variant_params["network"]["decoder_version"] = decoder_version

  if length_model_opts:
    # make sure that we do not add any params which are not present in the defaults
    assert set(length_model_opts.keys()).issubset(set(variant_params["network"]["length_model_opts"].keys()))
    variant_params["network"]["length_model_opts"].update(length_model_opts)

  config_builder = LibrispeechConformerSegmentalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


def center_window_att_baseline(
        win_size_list: Tuple[int, ...] = (5, 129),
):
  for win_size in win_size_list:
    alias = f"{base_alias}/baseline/win-size-%d" % (
      win_size
    )
    yield alias, get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
    )
