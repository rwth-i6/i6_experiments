__all__ = ["SystemInput"]

import copy
from typing import Optional, Dict, Union

from sisyphus import tk

import i6_core.rasr as rasr
from i6_core.util import MultiPath

from i6_experiments.common.setups.rasr.util import (
    ReturnnRasrDataInput,
)


class SystemInput:
    """
    holds all the information generated as output to the GMM pipeline
    """

    def __init__(self):
        self.crp: Optional[rasr.CommonRasrParameters] = None
        self.feature_flows: Dict[str, rasr.FlowNetwork] = {}
        self.features: Dict[str, Union[tk.Path, MultiPath, rasr.FlagDependentFlowAttribute]] = {}
        self.alignments: Optional[Union[tk.Path, MultiPath, rasr.FlagDependentFlowAttribute]] = None

    def as_returnn_rasr_data_input(
        self,
        name: str = "init",
        *,
        feature_flow_key: str = "gt",
        shuffle_data: bool = False,
        segment_order_sort_by_time_length: bool = False,
        chunk_size=384,
    ):
        shuffle_params = {"shuffle_data": shuffle_data}
        if segment_order_sort_by_time_length:
            shuffle_params["segment_order_sort_by_time_length_chunk_size"] = chunk_size

        return ReturnnRasrDataInput(
            name=name,
            crp=copy.deepcopy(self.crp),
            alignments=self.alignments,
            feature_flow=self.feature_flows[feature_flow_key],
            features=self.features[feature_flow_key],
            chunk_size=chunk_size,
            shuffling_parameters=shuffle_params,
        )
