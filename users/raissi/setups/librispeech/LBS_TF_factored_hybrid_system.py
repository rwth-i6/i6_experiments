__all__ = ["LBSTFFactoredHybridSystem"]

import copy
import dataclasses
import itertools
import numpy as np

from typing import Dict, List, Optional, Tuple, TypedDict, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

Path = tk.setup_path(__package__)

# -------------------- Recipes --------------------
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog

import i6_experiments.users.raissi.setups.librispeech.decoder as lbs_decoder

# --------------------------------------------------------------------------------
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    SingleSoftmaxType,
)

from i6_experiments.users.raissi.setups.common.TF_factored_hybrid_system import (
    TFFactoredHybridBaseSystem,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)



from i6_experiments.users.raissi.setups.librispeech.decoder import (
    LBSSearchParameters
)

from i6_experiments.users.raissi.setups.common.decoder import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    RasrFeatureScorer,

)

from i6_experiments.users.raissi.experiments.librispeech.configs.LFR_factored.baseline.config import (
    ALIGN_GMM_TRI_ALLOPHONES_NOUNK,
    ALIGN_GMM_TRI_10MS,

)

from i6_experiments.users.raissi.setups.librispeech.config import CV_SEGMENTS, P_HMM_AM7T1_ALIGNMENT_40ms


class LBSTFFactoredHybridSystem(TFFactoredHybridBaseSystem):
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
        self.recognizers = {"base": lbs_decoder.LBSFactoredHybridDecoder}
        self.cv_info = {"segment_list": CV_SEGMENTS, "alignment": {"dev-other_dev-clean": P_HMM_AM7T1_ALIGNMENT_40ms}}
        self.reference_alignment = {
            "GMM": {
                "alignment": ALIGN_GMM_TRI_10MS,
                "allophones": ALIGN_GMM_TRI_ALLOPHONES_NOUNK,
            }
        }
        self.alignment_example_segments = [
                            "train-other-960/2920-156224-0013/2920-156224-0013",
                            "train-other-960/2498-134786-0003/2498-134786-0003",
                            "train-other-960/6178-86034-0008/6178-86034-0008",
                            "train-other-960/5983-39669-0034/5983-39669-0034",
                        ]
        self.ivectors_prepath = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/ivectors"


    def concat_features_with_ivectors_for_feature_flow(self):
        assert self.ivectors_prepath, "Set the ivectors prepath"
        for k in self.feature_bundles.keys():
            ivec_cached_bundle = Path(f'{self.ivectors_prepath}/{k}/ivec.bundle', cached=True)

            ivector_cached_path = rasr.FlagDependentFlowAttribute("cache_mode",
                                                                  {
                                                                      "bundle":ivec_cached_bundle,
                                                                    "task_dependent": ivec_cached_bundle
                                                                  }
                                                                  )
            ft_k = self.feature_info.feature_type.get()
            self.feature_flows[k][ft_k] = self.concat_features_with_ivec(feature_net=self.feature_flows[k][ft_k], ivec_path=ivector_cached_path)



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
        joint_for_factored_loss: bool = False,
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
                key=key, state_tying=self.label_info.state_tying, softmax_type=SingleSoftmaxType.DECODE, joint_for_factored_loss=joint_for_factored_loss,
            )
        else:
            crp_list = [n for n in self.crp_names if "train" not in n]
            self.reset_state_tying(crp_list=crp_list, state_tying=self.label_info.state_tying)

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        recog_args = lbs_decoder.LBSSearchParameters.default_for_ctx(
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
        recog_args: LBSSearchParameters,
        lm_scale: float,
        context_type: PhoneticContext = None,
        feature_scorer_type: RasrFeatureScorer = None,
        tdp_scales: List = None,
        transition_loop_sil: List = None,
        transition_loop_speech: List = None,
        transition_exit_sil: List = None,
        transition_exit_speech: List = None,
        extend: bool = True,
    ) -> LBSSearchParameters:

        assert self.experiments[key]["decode_job"]["runner"] is not None, "Please set the recognizer"
        recognizer = self.experiments[key]["decode_job"]["runner"]

        context_type = PhoneticContext.diphone if context_type is None else context_type
        feature_scorer_type = RasrFeatureScorer.nn_precomputed if feature_scorer_type is None else feature_scorer_type
        if context_type == PhoneticContext.triphone_forward:
            assert feature_scorer_type == feature_scorer_type.factored, "no triphone with nn precomputed yet"

        tdp_scales = [0.1, 0.2] if tdp_scales is None else tdp_scales
        if feature_scorer_type == RasrFeatureScorer.factored:
            if context_type == PhoneticContext.triphone_forward:
                prior_scales = list(
                    itertools.product(
                        [v for v in np.arange(0.1, 0.6, 0.1).round(1)],
                        [v for v in np.arange(0.1, 0.6, 0.1).round(1)],
                        [v for v in np.arange(0.1, 0.6, 0.1).round(1)],
                    )
                )
            else:
                raise NotImplementedError("You were not supposed to run monophone decoding with factored decoder")
        else:
            prior_scales = [[v] for v in np.arange(0.1, 0.8, 0.1).round(1)]


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
            prior_scales=prior_scales,
            tdp_scales=tdp_scales,

        )

        sil_loop = [8.0, 11.0, 13.0]
        if transition_loop_sil is not None:
            if extend:
                sil_loop.extend(transition_loop_sil)
            else: sil_loop = transition_loop_sil
        sil_exit = [10.0, 15.0, 20.0]
        if transition_exit_sil is not None:
            if extend:
                sil_exit.extend(transition_exit_sil)
            else: sil_exit = transition_exit_sil
        speech_loop = [5.0, 8.0, 11.0]
        if transition_loop_speech is not None:
            if extend:
                speech_loop.extend(transition_loop_speech)
            else: speech_loop = transition_loop_speech
        speech_exit = [0.0, 5.0]
        if transition_exit_speech is not None:
            if extend:
                speech_exit.extend(transition_exit_speech)
            else: speech_exit = transition_exit_speech



        nnsp_tdp = [(l, 0.0, "infinity", e) for l in sil_loop for e in sil_exit]
        sp_tdp = [(l, 0.0, "infinity", e) for l in speech_loop for e in speech_exit]
        best_config = recognizer.recognize_optimize_transtition_values(
            label_info=self.label_info,
            search_parameters=best_config_scales,
            num_encoder_output=num_encoder_output,
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

        align_args = self.get_parameters_for_aligner(context_type=context_type, prior_info=p_info)

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
