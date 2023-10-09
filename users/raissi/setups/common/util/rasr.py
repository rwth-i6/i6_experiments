__all__ = ["SystemInput"]

import copy
from typing import Dict, Optional

from i6_experiments.common.setups.rasr.util import (
    ReturnnRasrDataInput,
    ReturnnRasrTrainingArgs,
)


class SystemInput:
    """
    holds all the information generated as input to the system ndependently from a previous GMM system
    """

    def __init__(self):
        self.crp: Optional[rasr.CommonRasrParameters] = None
        self.feature_flows: Dict[str, rasr.FlowNetwork] = {}
        self.features: Dict[str, Union[tk.Path, MultiPath, rasr.FlagDependentFlowAttribute]] = {}
        self.alignments: Optional[Union[tk.Path, MultiPath, rasr.FlagDependentFlowAttribute]] = (None,)
        self.returnn_rasr_training_args: Optional[ReturnnRasrTrainingArgs] = None

    def as_returnn_rasr_data_input(
        self,
        name: str = "init",
        *,
        feature_flow_key: str = "gt",
        shuffling_parameters: Dict = {},
        returnn_rasr_training_args: Optional[ReturnnRasrTrainingArgs] = None,
    ):
        """
        Independently from an existing system, stores all info that can be used for bootstrapping
        a hybrid system

        :param name:
        :param feature_flow_key:
        :param shuffle_data:
        :return:
        :rtype: ReturnnRasrDataInput
        """
        return ReturnnRasrDataInput(
            name=name,
            crp=copy.deepcopy(self.crp),
            alignments=self.alignments,
            feature_flow=self.feature_flows[feature_flow_key],
            features=self.features[feature_flow_key],
            shuffling_parameters=shuffling_parameters,
            returnn_rasr_training_args=returnn_rasr_training_args,
        )
