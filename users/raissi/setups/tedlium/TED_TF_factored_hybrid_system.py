__all__ = ["TEDTFFactoredHybridSystem"]

import copy
import dataclasses
import itertools
import sys
from IPython import embed

from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

Path = tk.setup_path(__package__)

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog

import i6_experiments.users.raissi.setups.tedlium.decoder as ted_decoder

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

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    RasrFeatureScorer,
)


from i6_experiments.users.raissi.setups.common.features.taxonomy import FeatureType


class TEDTFFactoredHybridSystem(TFFactoredHybridBaseSystem):
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
        self.recognizers = {"base": ted_decoder.TEDFactoredHybridDecoder}
        self.feature_info = dataclasses.replace(self.feature_info, feature_type=FeatureType.filterbanks)

    def _get_merged_corpus_for_corpora(
        self, corpora, name="TED-LIUM-realease2", strategy=corpus_recipe.MergeStrategy.FLAT
    ):
        # due to current bug in the i6_core I need to hack the name as it is
        merged_corpus_job = corpus_recipe.MergeCorporaJob(corpora, name="TED-LIUM-realease2", merge_strategy=strategy)
        return merged_corpus_job.out_merged_corpus

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

        recog_args = ted_decoder.TEDSearchParameters.default_for_ctx(
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
            eval_args=self.scorer_args[crp_corpus],
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
