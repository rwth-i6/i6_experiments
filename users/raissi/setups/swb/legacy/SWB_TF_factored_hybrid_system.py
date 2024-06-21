__all__ = ["SWBTFFactoredHybridSystem"]

import copy
import dataclasses
import itertools
import numpy as np
import sys
from IPython import embed

from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog
import i6_core.returnn as returnn
import i6_core.text as text

from i6_core.util import MultiPath, MultiOutputPath

# common modules
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)

from i6_experiments.users.berger.network.helpers.conformer_wei import add_initial_conv, add_conformer_stack

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    TrainingCriterion,
    SingleSoftmaxType,
)

from i6_experiments.users.raissi.setups.common.TF_factored_hybrid_system import (
    TFFactoredHybridBaseSystem,
    ExtraReturnnCode,
    Graphs,
    ExtraReturnnCode,
    TFExperiment,
)


import i6_experiments.users.raissi.setups.common.encoder as encoder_archs
import i6_experiments.users.raissi.setups.common.helpers.network as net_helpers
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
import i6_experiments.users.raissi.setups.common.helpers.decode as decode_helpers


# user based modules

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)


from i6_experiments.users.raissi.setups.common.helpers.priors import (
    get_returnn_config_for_center_state_prior_estimation,
    get_returnn_config_for_left_context_prior_estimation,
    get_returnn_configs_for_right_context_prior_estimation,
    smoothen_priors,
    JoinRightContextPriorsJob,
    ReshapeCenterStatePriorsJob,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    RasrFeatureScorer,
)

from i6_experiments.users.raissi.setups.swb.legacy.decoder.SWB_factored_hybrid_search import (
    SWBFactoredHybridDecoder,
)

from i6_experiments.users.raissi.setups.swb.legacy.decoder.config import (
    SWBSearchParameters,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    BASEFactoredHybridAligner,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
)

from i6_experiments.users.raissi.experiments.swb.legacy.data_preparation.legacy_constants_and_paths_swb1 import (
    feature_bundles,
    concurrent,
)


# -------------------- Init --------------------

Path = tk.setup_path(__package__)


# -------------------- Systems --------------------
class SWBTFFactoredHybridSystem(TFFactoredHybridBaseSystem):
    """
    this class supports both cart and factored hybrid
    """

    def __init__(
        self,
        returnn_root: Optional[str] = None,
        returnn_python_home: Optional[str] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        rasr_binary_path: Optional[tk.Path] = None,
        rasr_init_args: RasrInitArgs = None,
        train_data: Dict[str, RasrDataInput] = None,
        dev_data: Dict[str, RasrDataInput] = None,
        test_data: Dict[str, RasrDataInput] = None,
        initial_nn_args: Dict = None,
    ):
        super().__init__(
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_init_args=rasr_init_args,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            initial_nn_args=initial_nn_args,
        )

        self.dependencies_path = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies"
        self.recognizers = {"base": SWBFactoredHybridDecoder}
        self.legacy_stm_files = {
            "hub500": tk.Path(f"{self.dependencies_path}/stm-files/hub5e_00.2.stm"),
            "hub501": tk.Path(f"{self.dependencies_path}/stm-files/hub5e_01.2.stm"),
        }
        self.glm_files = dict(zip(["hub500", "hub501"], [tk.Path("/u/corpora/speech/hub5e_00/xml/glm")] * 2))

        self.segments_to_exclude = [
            "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
            "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
            "switchboard-1/sw02691A/sw2691A-ms98-a-0017",
            "switchboard-1/sw04091A/sw4091A-ms98-a-0063",
            "switchboard-1/sw04103A/sw4103A-ms98-a-0022",
            "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
            "switchboard-1/sw04318A/sw4318A-ms98-a-0024",
            "switchboard-1/sw02691A/sw2691A-ms98-a-0017",
            "switchboard-1/sw03266B/sw3266B-ms98-a-0055",
            "switchboard-1/sw04103A/sw4103A-ms98-a-0022",
            "switchboard-1/sw04181A/sw4181A-ms98-a-0036",
            "switchboard-1/sw04318A/sw4318A-ms98-a-0024",
            "switchboard-1/sw04624A/sw4624A-ms98-a-0055" "hub5e_00/en_6189a/36",
            "hub5e_00/en_4852b/77",
            "hub5e_00/en_6189b/66",
        ]

        self.cross_validation_info = {
            "pre_path": ("/").join([self.dependencies_path, "cv-from-hub5-00"]),
            "merged_corpus_path": ("/").join(["merged_corpora", "train-dev.corpus.gz"]),
            "merged_corpus_segment": ("/").join(["merged_corpora", "segments"]),
            "cleaned_dev_corpus_path": ("/").join(["zhou-files-dev", "hub5_00.corpus.cleaned.gz"]),
            "cleaned_dev_segment_path": ("/").join(["zhou-files-dev", "segments"]),
            "features_path": ("/").join(["features", "gammatones", "FeatureExtraction.Gammatone.pp9W8m2Z8mHU"]),
        }
        self.prior_transcript_estimates = {
            "monostate": {
                "state_prior": ("/").join(
                    [self.dependencies_path, "haotian/monostate/monostate.transcript.prior.pickle"]
                ),
                "state_EOW_prior": ("/").join(
                    [self.dependencies_path, "haotian/monostate/monostate.we.transcript.prior.pickle"]
                ),
                "speech_forward": 0.125,
                "silence_forward": 0.025,
            },
            # 1/8 phoneme
            "threepartite": {
                "state_prior": ("/").join(
                    [self.dependencies_path, "haotian/threepartite/threepartite.transcript.prior.pickle"]
                ),
                "state_EOW_prior": ("/").join(
                    [self.dependencies_path, "haotian/threepartite/threepartite.we.transcript.prior.pickle"]
                ),
                "speech_forward": 0.350,
                "silence_forward": 0.025,
            }
            # 1/9 for 3-state, same amount of silence
        }
        self.transcript_prior_xml = {
            "monostate": ("/").join([self.dependencies_path, "haotian/monostate/monostate.we.transcript.prior.xml"]),
        }

    # -------------------- External helpers --------------------

    def set_gammatone_features(self):
        feature_name = self.feature_info.feature_type.get()
        for corpus_key in ["train", "hub500", "hub501"]:
            self.feature_bundles[corpus_key] = {feature_name: feature_bundles[corpus_key]}
            self.feature_flows[corpus_key] = {feature_name: features.basic_cache_flow(feature_bundles[corpus_key])}
            mapping = {"train": "train", "hub500": "dev", "hub501": "eval"}
            cache_pattern = feature_bundles[corpus_key].get_path().split(".bundle")[0]
            caches = [tk.Path(f"{cache_pattern}.{i}") for i in range(1, concurrent[mapping[corpus_key]] + 1)]
            self.feature_caches[corpus_key] = {feature_name: caches}

    def set_stm_and_glm(self):
        for corpus in ["hub500", "hub501"]:
            self.scorer_args[corpus] = {
                "ref": self.legacy_stm_files[corpus],
                "glm": self.glm_files[corpus],
            }

    def prepare_data_with_separate_cv_legacy(self, cv_key="train.cvtrain", bw_key="bw"):
        cv_corpus = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["cleaned_dev_corpus_path"]]
        )
        cv_segment = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["cleaned_dev_segment_path"]]
        )
        cv_feature_path = Path(
            ("/").join(
                [
                    self.cross_validation_info["pre_path"],
                    self.cross_validation_info["features_path"],
                    "output",
                    "gt.cache.bundle",
                ]
            )
        )
        self.cv_input_data[cv_key].update_crp_with(corpus_file=cv_corpus, segment_path=cv_segment, concurrent=1)
        self.feature_flows[cv_key] = self.cv_input_data[cv_key].feature_flow = features.basic_cache_flow(
            cv_feature_path
        )

        if self.training_criterion == TrainingCriterion.VITERBI:
            return

        self.crp[self.crp_names[bw_key]].corpus_config.file = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["merged_corpus_path"]]
        )
        self.crp[self.crp_names[bw_key]].corpus_config.segments.file = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["merged_corpus_segment"]]
        )

    def get_recognizer_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        epoch: int,
        crp_corpus: str,
        recognizer_key: str = "base",
        model_path: Optional[Path] = None,
        gpu=False,
        is_multi_encoder_output=False,
        set_batch_major_for_feature_scorer: bool = True,
        tf_library: Union[Path, str, List[Path], List[str], None] = None,
        dummy_mixtures: Optional[Path] = None,
        lm_gc_simple_hash: Optional[bool] = None,
        crp: Optional[rasr.RasrConfig] = None,
        **decoder_kwargs,
    ):
        if context_type in [
            PhoneticContext.mono_state_transition,
            PhoneticContext.diphone_state_transition,
            PhoneticContext.tri_state_transition,
        ]:
            name = f'{self.experiments[key]["name"]}-delta/e{epoch}/{crp_corpus}'
        else:
            name = f"{self.experiments[key]['name']}/e{epoch}/{crp_corpus}"

        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        if (
            feature_scorer_type == RasrFeatureScorer.nn_precomputed
            and self.experiments[key]["returnn_config"] is not None
        ):

            self.setup_returnn_config_and_graph_for_single_softmax(
                key=key, state_tying=self.label_info.state_tying, softmax_type=SingleSoftmaxType.DECODE
            )
        else:
            crp_list = [n for n in self.crp_names if "train" not in n]
            self.reset_state_tying(crp_list=crp_list, state_tying=self.label_info.state_tying)

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        recog_args = SWBSearchParameters.default_for_ctx(
            context=context_type, priors=p_info, frame_rate=self.frame_rate_reduction_ratio_info.factor
        )

        if dummy_mixtures is None:
            n_labels = (
                self.cart_state_tying_args["cart_labels"]
                if self.label_info.state_tying == RasrStateTying.cart
                else self.label_info.get_n_of_dense_classes()
            )
            dummy_mixtures = mm.CreateDummyMixturesJob(
                n_labels,
                self.initial_nn_args["num_input"],
            ).out_mixtures

        assert self.label_info.sil_id is not None

        if model_path is None:
            model_path = self.get_model_checkpoint(self.experiments[key]["train_job"], epoch)

        recognizer = self.recognizers[recognizer_key](
            name=name,
            crp=self.crp[crp_corpus] if crp is None else crp,
            context_type=context_type,
            feature_scorer_type=feature_scorer_type,
            feature_path=self.feature_flows[crp_corpus],
            model_path=model_path,
            graph=graph,
            mixtures=dummy_mixtures,
            eval_files=self.scorer_args[crp_corpus],
            scorer=self.scorers[crp_corpus],
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
            lm_gc_simple_hash=lm_gc_simple_hash if (lm_gc_simple_hash is not None and lm_gc_simple_hash) else None,
            **decoder_kwargs,
        )

        return recognizer, recog_args

    def get_best_recog_scales_and_transition_values(
        self,
        key: str,
        num_encoder_output: int,
        recog_args: SWBSearchParameters,
        lm_scale: float,
    ) -> SWBSearchParameters:

        assert self.experiments[key]["decode_job"]["runner"] is not None, "Please set the recognizer"
        recognizer = self.experiments[key]["decode_job"]["runner"]

        tune_args = recog_args.with_lm_scale(lm_scale)
        best_config_scales = recognizer.recognize_optimize_scales_v2(
            label_info=self.label_info,
            search_parameters=tune_args,
            num_encoder_output=num_encoder_output,
            altas_value=2.0,
            altas_beam=16.0,
            tdp_sil=[(11.0, 0.0, "infinity", 20.0)],
            tdp_speech=[(8.0, 0.0, "infinity", 0.0)],
            tdp_nonword=[(8.0, 0.0, "infinity", 0.0)],
            prior_scales=[[v] for v in np.arange(0.1, 0.8, 0.1).round(1)],
            tdp_scales=[0.1, 0.2],
        )

        nnsp_tdp = [(l, 0.0, "infinity", e) for l in [8.0, 11.0, 13.0] for e in [10.0, 15.0, 20.0]]
        sp_tdp = [(l, 0.0, "infinity", e) for l in [5.0, 8.0, 11.0] for e in [0.0, 5.0]]
        best_config= recognizer.recognize_optimize_transtition_values(
            label_info=self.label_info,
            search_parameters=best_config_scales,
            num_encoder_output=512,
            altas_value=2.0,
            altas_beam=16.0,
            tdp_sil=nnsp_tdp,
            tdp_speech=sp_tdp,
        )

        return best_config




    def get_aligner_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        epoch: int,
        crp_corpus: str,
        aligner_key: str = "base",
        model_path: Optional[Path] = None,
        feature_path: Optional[Path] = None,
        gpu: bool = False,
        is_multi_encoder_output: bool = False,
        set_batch_major_for_feature_scorer: bool = True,
        tf_library: Union[Path, str, List[Path], List[str], None] = None,
        dummy_mixtures: Optional[Path] = None,
        crp: Optional[rasr.RasrConfig] = None,
        **aligner_kwargs,
    ):
        if context_type in [
            PhoneticContext.mono_state_transition,
            PhoneticContext.diphone_state_transition,
            PhoneticContext.tri_state_transition,
        ]:
            name = f'{self.experiments[key]["name"]}-delta/e{epoch}/{crp_corpus}'
        else:
            name = f"{self.experiments[key]['name']}/e{epoch}/{crp_corpus}"

        if (
            feature_scorer_type == RasrFeatureScorer.nn_precomputed
            and self.experiments[key]["returnn_config"] is not None
        ):

            self.setup_returnn_config_and_graph_for_single_softmax(
                key=key, state_tying=self.label_info.state_tying, softmax_type=SingleSoftmaxType.DECODE
            )

        else:
            crp_list = [n for n in self.crp_names if "align" in n]
            self.reset_state_tying(crp_list=crp_list, state_tying=self.label_info.state_tying)

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(
                self.label_info.get_n_of_dense_classes(),
                self.initial_nn_args["num_input"],
            ).out_mixtures

        assert self.label_info.sil_id is not None

        if model_path is None:
            model_path = self.get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        if feature_path is None:
            feature_path = self.feature_flows[crp_corpus]

        # consider if you need to create separate alignment params
        align_args = self.get_parameters_for_aligner(context_type=context_type, prior_info=p_info)
        align_args = dataclasses.replace(align_args, non_word_phonemes="[LAUGHTER],[NOISE],[VOCALIZEDNOISE]")

        aligner = self.aligners[aligner_key](
            name=name,
            crp=self.crp[crp_corpus] if crp is None else crp,
            context_type=context_type,
            feature_scorer_type=feature_scorer_type,
            feature_path=feature_path,
            model_path=model_path,
            graph=graph,
            mixtures=dummy_mixtures,
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
            **aligner_kwargs,
        )

        return aligner, align_args
