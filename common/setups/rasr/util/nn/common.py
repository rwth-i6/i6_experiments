__all__ = ["HybridArgs", "NnForcedAlignArgs"]

from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import i6_core.rasr as rasr
import i6_core.returnn as returnn

from .decode import KeyedRecogArgsType
from .training import ReturnnRasrTrainingArgs, ReturnnTrainingJobArgs


class HybridArgs:
    def __init__(
        self,
        returnn_training_configs: Dict[str, returnn.ReturnnConfig],
        returnn_recognition_configs: Dict[str, returnn.ReturnnConfig],
        training_args: Union[Dict[str, Any], ReturnnRasrTrainingArgs, ReturnnTrainingJobArgs],
        recognition_args: KeyedRecogArgsType,
        test_recognition_args: Optional[KeyedRecogArgsType] = None,
    ):
        """
        ##################################################
        :param returnn_training_configs
            RETURNN config keyed by training corpus.
        ##################################################
        :param returnn_recognition_configs
            If a config is not found here, the corresponding training config is used
        ##################################################
        :param training_args:
        ##################################################
        :param recognition_args:
            Configuration for recognition on dev corpora.
        ##################################################
        :param test_recognition_args:
            Additional configuration for recognition on test corpora. Merged with recognition_args.
        ##################################################
        """
        self.returnn_training_configs = returnn_training_configs
        self.returnn_recognition_configs = returnn_recognition_configs
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args


class NnForcedAlignArgs(TypedDict):
    name: str
    target_corpus_keys: List[str]
    feature_scorer_corpus_key: str
    scorer_model_key: Union[str, List[str], Tuple[str], rasr.FeatureScorer]
    epoch: int
    base_flow_key: str
    tf_flow_key: str
    dump_alignment: bool
