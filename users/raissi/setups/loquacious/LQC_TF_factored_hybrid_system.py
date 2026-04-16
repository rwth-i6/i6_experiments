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

from i6_experiments.users.raissi.args.system import get_tdp_values

Path = tk.setup_path(__package__)

# -------------------- Recipes --------------------
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog

import i6_experiments.users.raissi.setups.loquacious.decoder as lqc_decoder

# --------------------------------------------------------------------------------
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    SingleSoftmaxType,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.decoder import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    RasrFeatureScorer,
)

from i6_experiments.users.raissi.setups.common.TF_factored_hybrid_system import (
    TFFactoredHybridBaseSystem,
)


from i6_experiments.users.raissi.setups.loquacious.decoder import LQCSearchParameters, LQCFactoredHybridDecoder


from i6_experiments.users.raissi.setups.loquacious.config import (
    CV_SEGMENTS,
    CV_ALIGNMENT,
    ALIGN_GMM_MONO,
    ALIGN_GMM_MONO_ALLOPHONES,
)


class LQCTFFactoredHybridSystem(TFFactoredHybridBaseSystem):
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
        self.recognizers = {"base": LQCFactoredHybridDecoder}
        self.cv_info = {"segment_list": CV_SEGMENTS, "alignment": {"dev.short": CV_ALIGNMENT}}  # ToDo provide
        self.reference_alignment = None
        self.alignment_example_segments = []
        self.stms = {
                'dev.all':tk.Path("/u/raissi/setups/loquacious/2025-02--paper/s250h/work/i6_core/corpus/convert/CorpusToStmJob.zJhFuygafnf2/output/corpus.stm"),
                'test.all': tk.Path("/u/raissi/setups/loquacious/2025-02--paper/s250h/work/i6_core/corpus/convert/CorpusToStmJob.BFjrQqGsbXtR/output/corpus.stm")
        }
        self.segments_to_exclude = ['loquacious-train-small/20151216-0900-PLENARY-4-en_20151216-09:14:08_49/0',
             'loquacious-train-small/20191218-0900-PLENARY-en_20191218-10:16:38_22/0',
             'loquacious-train-small/MZ2yoqEbFFA-00042-00010614-00011018.wav/0',
             'loquacious-train-small/BaRWvVqA9hA-00230-00065978-00066352.wav/0',
             'loquacious-train-small/G1bs59Sqhwc-00071-00075722-00076517.wav/0',
             'loquacious-train-small/YAHF9jm-_GA-00709-00156042-00156381.wav/0',
             'loquacious-train-small/CaNeB6reDZc-00358-00544700-00545000.wav/0',
             'loquacious-train-small/YAHF9jm-_GA-00824-00179023-00179486.wav/0',
             'loquacious-train-small/2g1X-xs9WPo-00400-00107588-00107908.wav/0',
             'loquacious-train-small/BaRWvVqA9hA-00180-00053346-00053753.wav/0',
             'loquacious-train-small/6pnmrmxoeIu-00008-00005460-00005900.wav/0',
             'loquacious-train-small/T2itrpH3upY-00068-00050200-00050566.wav/0',
             'loquacious-train-small/YAHF9jm-_GA-00237-00059441-00059828.wav/0',
             'loquacious-train-small/crMNcug9IkE-00004-00001324-00001707.wav/0',
             'loquacious-train-small/yZR9h111IGU-00603-00288147-00288501.wav/0',
             'loquacious-train-small/v3xG9Irp6DA-00254-00610200-00610900.wav/0',
             'loquacious-train-small/ctkMjwhoxaE-00068-00035111-00035436.wav/0',
             'loquacious-train-small/ogapnepS-zE-00006-00003797-00004498.wav/0',
             'loquacious-train-small/wVJnhE6UjVU-00020-00002312-00002615.wav/0']



    def evaluate_all(self, dev_ctms: Dict[str, tk.Path], test_ctms: Optional[dict[str, tk.Path]], prefix_name: str = "decoding/combined/"):
        """
        #by Nick Rossenbach
        Compute the full Loquacious WER based on the given subset ctm dicts
        """

        dev_stm = self.stms['dev.all']
        test_stm = self.stms['test.all']

        from i6_core.text.processing import PipelineJob
        from i6_core.recognition.scoring import ScliteJob
        dev_ctm_all = PipelineJob(list(dev_ctms.values()), [], zip_output=False, mini_task=True).out

        SCTK_PATH = tk.Path("/u/beck/programs/sctk-2.4.0/bin/")
        dev_sclite_job = ScliteJob(ref=dev_stm, hyp=dev_ctm_all, sort_files=True, sctk_binary_path=SCTK_PATH,
                                   precision_ndigit=1)

        tk.register_output(prefix_name + "dev.all/sclite/wer", dev_sclite_job.out_wer)
        tk.register_output(prefix_name + "dev.all/sclite/report", dev_sclite_job.out_report_dir)

        if test_ctms is not None:
            test_ctm_all = PipelineJob(list(test_ctms.values()), [], zip_output=False, mini_task=True).out
            test_sclite_job = ScliteJob(ref=test_stm, hyp=test_ctm_all, sort_files=True, sctk_binary_path=SCTK_PATH,
                                        precision_ndigit=1)
            tk.register_output(prefix_name + "test.all/sclite/wer", test_sclite_job.out_wer)
            tk.register_output(prefix_name + "test.all/sclite/report", test_sclite_job.out_report_dir)


    def get_recognizer_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        epoch: int,
        crp_corpus: str,
        recognizer_key: str = "base",
        model_path: Optional[Path] = None,
        graph_path: Optional[Path] = None,
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

        assert self.label_info.sil_id is not None

        if model_path is None:
            model_path = self.get_model_checkpoint(self.experiments[key]["train_job"], epoch)

        if (
            feature_scorer_type == RasrFeatureScorer.nn_precomputed
            and self.experiments[key]["returnn_config"] is not None
        ):

            self.setup_returnn_config_and_graph_for_single_softmax(
                key=key,
                state_tying=self.label_info.state_tying,
                softmax_type=SingleSoftmaxType.DECODE,
                joint_for_factored_loss=joint_for_factored_loss,
            )
        else:
            crp_list = [n for n in self.crp_names if "train" not in n]
            self.reset_state_tying(crp_list=crp_list, state_tying=self.label_info.state_tying)

        if graph_path is None:
            graph = self.experiments[key]["graph"].get("inference", None)
            if graph is None:
                self.set_graph_for_experiment(key=key)
                graph = self.experiments[key]["graph"]["inference"]
        else:
            graph = graph_path

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

        recog_args = lqc_decoder.LQCSearchParameters.default_for_ctx(
            context=context_type, priors=p_info, frame_rate=self.frame_rate_reduction_ratio_info.factor
        )
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
