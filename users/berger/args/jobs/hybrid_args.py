from typing import Optional, Dict

from i6_experiments.users.berger.args.jobs.training_args import get_base_training_args
from i6_experiments.users.berger.args.jobs.recognition_args import get_recognition_args
from i6_experiments.users.berger.args.jobs.alignment_args import get_alignment_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
import i6_experiments.common.setups.rasr.util as rasr_util


def get_nn_args(
    train_networks: Dict[str, dict],
    *,
    num_inputs: int,
    num_outputs: int,
    num_epochs: int,
    search_type: SearchTypes = SearchTypes.AdvancedTreeSearch,
    recog_networks: Dict[str, dict] = {},
    keep_epochs: Optional[list] = None,
    returnn_train_config_args: Optional[dict] = None,
    returnn_recog_config_args: Optional[dict] = None,
    train_args: Optional[dict] = None,
    prior_args: Optional[dict] = None,
    align_name: Optional[str] = None,
    align_args: Optional[dict] = None,
    recog_name: Optional[str] = None,
    recog_args: Optional[dict] = None,
    test_recog_name: Optional[str] = None,
    test_recog_args: Optional[dict] = None,
) -> rasr_util.HybridArgs:

    # By default every 10th epoch starting at 80% of the training
    if keep_epochs is None:
        keep_epochs = list(range(10 * ((num_epochs * 4) // 50), num_epochs + 1, 10))

    returnn_train_configs = {
        name: get_returnn_config(
            net,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_epochs=num_epochs,
            keep_epochs=keep_epochs,
            **(returnn_train_config_args or {}),
        )
        for name, net in train_networks.items()
    }

    returnn_recog_configs = {
        name: get_returnn_config(
            net,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_epochs=num_epochs,
            **(returnn_recog_config_args or {}),
        )
        for name, net in recog_networks.items()
    }

    training_args = get_base_training_args(
        num_epochs=num_epochs, num_outputs=num_outputs, **(train_args or {})
    )

    recog_name = recog_name or "recog"
    recognition_args = {
        recog_name: get_recognition_args(
            search_type, epochs=keep_epochs, **(recog_args or {})
        )
    }

    test_recog_name = test_recog_name or recog_name
    test_recognition_args = {
        test_recog_name: get_recognition_args(
            search_type, epochs=keep_epochs, **(test_recog_args or recog_args or {})
        )
    }

    align_name = align_name or "align"
    alignment_args = {
        align_name: get_alignment_args(
            search_type, epochs=[num_epochs], **(align_args or {})
        )
    }

    return rasr_util.HybridArgs(
        returnn_training_configs=returnn_train_configs,
        returnn_recognition_configs=returnn_recog_configs,
        training_args=training_args,
        prior_args=prior_args,
        alignment_args=alignment_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )
