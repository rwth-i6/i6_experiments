from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from typing import Optional, Dict, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.ilm_correction.ilm_correction import add_ilm_correction as base_add_ilm_correction


def add_zero_att(
        network: Dict,
        rec_layer_name: str,
        returnn_config: ReturnnConfig
):
  returnn_config.config["att_feature_dim_tag"] = CodeWrapper(
    'DimensionTag(kind=DimensionTag.Types.Feature, description="att_feature_dim", dimension=512)')
  returnn_config.python_prolog.append("from returnn.tensor import batch_dim")

  network[rec_layer_name]["unit"].update({
    "zero_att": {
      "class": "constant",
      "value": 0.0,
      "shape": CodeWrapper("[batch_dim, att_feature_dim_tag]"),
    },
  })


def add_ilm_correction(
        network: Dict,
        rec_layer_name: str,
        target_num_labels: int,
        opts: Dict,
        label_prob_layer: str,
        use_mask_layer: bool = False,
        returnn_config: Optional[ReturnnConfig] = None,
):
  add_zero_att(
    network=network,
    rec_layer_name=rec_layer_name,
    returnn_config=returnn_config,
  )

  base_add_ilm_correction(
    network=network,
    rec_layer_name=rec_layer_name,
    target_num_labels=target_num_labels,
    opts=opts,
    label_prob_layer=label_prob_layer,
    att_layer_name="zero_att",
    static_att=True,
  )