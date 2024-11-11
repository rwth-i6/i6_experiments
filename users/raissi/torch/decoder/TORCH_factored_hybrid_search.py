__all__ = ["TORCHFactoredHybridDecoder", "TORCHFactoredHybridAligner"]

import copy
import dataclasses
import itertools
import numpy as np
import re

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from IPython import embed

from sisyphus import tk
from sisyphus.delayed_ops import Delayed, DelayedBase, DelayedFormat

from i6_core.meta import CorpusObject
from i6_experiments.common.setups.rasr import RasrSystem, RasrDataInput, RasrInitArgs
from i6_experiments.users.raissi.setups.common.decoder import RasrFeatureScorer
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    check_prior_info,
    BASEFactoredHybridDecoder
)

Path = tk.Path

import i6_core.am as am
import i6_core.features as features
import i6_core.lm as lm
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog


from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorConfig,
    PriorInfo,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
)

from i6_experiments.users.raissi.setups.common.decoder.statistics import ExtractSearchStatisticsJob
from i6_experiments.users.raissi.setups.common.util.tdp import format_tdp_val, format_tdp
from i6_experiments.users.raissi.setups.common.util.argmin import ComputeArgminJob
from i6_experiments.users.raissi.setups.common.data.typings import (
    TDP,
)

class FeatureFlowType(Enum):
    SAMPLE = auto()
    RASRFEATURES = auto()



@dataclass(eq=True, frozen=True)
class ONNXDecodeIOMap:
    """Map of tensor names used during decoding."""
    features: str
    """Name of the input features."""
    output: str
    """ Name of the output, ONNX feature scorer so far supports one single output"""
    features_size: Optional[str] = None
    """Might be needed by the Torch model"""

    @classmethod
    def default(cls) -> "ONNXDecodeIOMap":
        return ONNXDecodeIOMap(
            features="data",
            features_size="data_len",
            output="classes",
        )

def get_onnx_feature_scorer(
    context_type: PhoneticContext,
    feature_scorer_type: RasrFeatureScorer,
    io_map: Union[Dict[str, str], ONNXDecodeIOMap],
    mixtures: tk.Path,
    model: tk.Path,
    prior_info: Union[PriorInfo, tk.Variable, DelayedBase],
    posterior_scale: float = 1.0,
    apply_log_on_output: bool = False,
    negate_output: bool = True,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
    **kwargs

):

    assert context_type in [PhoneticContext.monophone, PhoneticContext.joint_diphone]

    if isinstance(io_map, ONNXDecodeIOMap):
        io_map = dataclasses.asdict(io_map, dict_factory=lambda x : {k: v for (k,v) in x if v is not None})

    if isinstance(prior_info, PriorInfo):
        check_prior_info(context_type=context_type, prior_info=prior_info)

    if context_type.is_joint_diphone():
        prior = prior_info.diphone_prior
    elif context_type.is_monophone():
        prior = prior_info.center_state_prior

    return feature_scorer_type.get_fs_class()(
        mixtures=mixtures,
        io_map=io_map,
        label_log_posterior_scale=posterior_scale,
        model=model,
        label_prior_scale=prior.scale,
        label_log_prior_file=prior.file,
        apply_log_on_output=apply_log_on_output,
        negate_output=negate_output,
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
        kwargs=kwargs,
    )



class TORCHFactoredHybridDecoder(BASEFactoredHybridDecoder):


    def __init__(
        self,
        name: str,
        rasr_binary_path: tk.Path,
        rasr_input_mapping: Dict[str, RasrDataInput],
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        feature_flow_type: FeatureFlowType,
        feature_path: Path,
        model_path: Path,
        io_map: Union[Dict[str, str], ONNXDecodeIOMap],
        mixtures: Path,
        eval_args,
        scorer: Optional[Union[recog.ScliteJob, recog.Hub5ScoreJob, recog.Hub5ScoreJob]] = None,
        is_multi_encoder_output: bool = False,
        silence_id: int = 40,
        set_batch_major_for_feature_scorer: bool = True,
        lm_gc_simple_hash=False,
        gpu=False,
    ):


        super().__init__(
            name=name,
            crp=self.get_crp(rasr_binary_path, rasr_input_mapping),
            context_type=context_type,
            feature_scorer_type=feature_scorer_type,
            feature_path=feature_path,
            model_path=model_path,
            graph=None, #only TF based decoding requires it
            mixtures=mixtures,
            eval_args=eval_args,
            scorer=scorer,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=silence_id,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
            lm_gc_simple_hash=lm_gc_simple_hash,
            gpu=gpu,
        )

        self.io_map = io_map
        self.set_feature_flow(feature_flow_type=feature_flow_type)

    def get_crp_and_eval_args(self, rasr_binary_path: tk.Path, rasr_input_mapping: Dict, rasr_init_args: RasrInitArgs)-> rasr.CommonRasrParameters:
        rasr_system = RasrSystem(rasr_binary_path=rasr_binary_path)
        rasr_system.rasr_init_args = rasr_init_args

        corpus_key = list(rasr_input_mapping.keys())[0]
        rasr_data_input = rasr_input_mapping[corpus_key]
        assert rasr_data_input.lm is not None, "You should set the lm for the rasr input"

        rasr_system.add_corpus(corpus_key, data=rasr_data_input, add_lm=True)
        stm_args = rasr_init_args.stm_args if rasr_init_args.stm_args is not None else {}
        rasr_system.create_stm_from_corpus(corpus_key, **stm_args)
        rasr_system._set_scorer_for_corpus(corpus_key)

        return rasr_system.crp[corpus_key],




    def get_base_sample_feature_flow(
        self, audio_format: str = "wav", dc_detection: bool = False, **kwargs
    ):
        args = {
            "audio_format": audio_format,
            "dc_detection": dc_detection,
            "input_options": {"block-size": 1},
            "scale_input": 2**-15,
        }
        args.update(kwargs)
        return features.samples_flow(**args)


    def set_feature_flow(self, feature_flow_type: FeatureFlowType):
        if feature_flow_type == FeatureFlowType.SAMPLE:
            self.feature_scorer_flow = self.get_base_sample_feature_flow()
        else:
            raise NotImplementedError





class TORCHFactoredHybridAligner(TORCHFactoredHybridDecoder):
    def __init__(
        self,
        name: str,
        crp,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        feature_flow_type: FeatureFlowType,
        feature_path: Path,
        io_map: Union[Dict[str, str], ONNXDecodeIOMap],
        model_path: Path,
        mixtures: Path,
        gpu=False,
        is_multi_encoder_output: bool = False,
        silence_id=40,
        set_batch_major_for_feature_scorer: bool = True,
    ):

        super().__init__(
            name=name,
            crp=crp,
            context_type=context_type,
            feature_scorer_type=feature_scorer_type,
            feature_flow_type=feature_flow_type,
            feature_path=feature_path,
            io_map=io_map,
            model_path=model_path,
            mixtures=mixtures,
            eval_args=None,
            gpu=gpu,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=silence_id,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
        )


