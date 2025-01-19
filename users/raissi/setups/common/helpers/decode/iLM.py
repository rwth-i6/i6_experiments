from dataclasses import dataclass
from typing import Dict

from i6_experiments.users.raissi.setups.common.data.factored_label import PhoneticContext, LabelInfo
from i6_experiments.users.raissi.setups.common.helpers.network import add_mlp

@dataclass
class ILMRenormINFO:
    renormalize: bool
    label_start_idx: int
    label_end_idx: int


def add_zero_ilm_to_returnn_dict(network: Dict, context_type: PhoneticContext, label_info: LabelInfo, ilm_renorm_info: ILMRenormINFO, ilm_scale: float):

    assert context_type in [PhoneticContext.diphone, PhoneticContext.triphone_forward], "Zero iLM can be done only for factored context-dependent models"

    network["zero_enc"] = {"class": "eval", "from": "encoder-output", "eval": "source(0) * 0"}
    if context_type == PhoneticContext.diphone:

        network["input-ilm"] = {
            "class": "copy",
            "from": ["zero_enc", "pastEmbed"],
        }

        ilm_ff_layer = add_mlp(network=network, layer_name="input-ilm", source_layer="input-ilm", size=network["linear1-diphone"]["n_out"], n_layers=2)
        network[ilm_ff_layer.replace("2", "1")]["reuse_params"] = "linear1-diphone"
        network[ilm_ff_layer]["reuse_params"] = "linear2-diphone"

        ilm_layer = "iLM"
        network[ilm_layer] = {
            "class": "linear",
            "from": ilm_ff_layer,
            "activation": "log_softmax",
            "n_out": label_info.get_n_state_classes(),
            "reuse_params": "center-output",
        }

        if ilm_renorm_info.renormalize:
            start = ilm_renorm_info.label_start_idx
            end = ilm_renorm_info.label_end_idx
            network["iLM-renorm"] = {
                "class": "eval",
                "from": [ilm_layer],
                "eval": f"tf.concat([source(0)[:, :81] - tf.math.log(1.0 - tf.exp(source(0)[:, 81:82])), tf.zeros(tf.shape(source(0)[:, 81:82])), source(0)[:, 82:] - tf.math.log(1.0 - tf.exp(source(0)[:, 81:82]))], axis=1)",
            }
            ilm_layer = "iLM-renorm"

        network["output_sub_iLM"] = {
            "class": "eval",
            "from": ["center-output", ilm_layer],
            "eval": f"tf.exp(safe_log(source(0)) - {ilm_scale} * source(1))",
            "is_output_layer": True
        }

    return network