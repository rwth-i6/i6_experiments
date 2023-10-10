__all__ = ["TFFactoredHybridBaseSystem"]

import copy
import itertools
import sys
from IPython import embed

from dataclasses import asdict
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
from i6_core.lexicon.allophones import DumpStateTyingJob, StoreAllophonesJob

# common modules
from i6_experiments.common.setups.rasr.nn_system import NnSystem

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    BASEFactoredHybridBaseSystem,
    Experiment,
    TrainingCriterion,
)


from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    RasrSteps,
    ReturnnRasrDataInput,
)

import i6_experiments.users.raissi.setups.common.encoder.blstm as blstm_setup
import i6_experiments.users.raissi.setups.common.encoder.conformer as conformer_setup
import i6_experiments.users.raissi.setups.common.helpers.network.augment as fh_augmenter
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
import i6_experiments.users.raissi.setups.common.helpers.decode as decode_helpers

from i6_experiments.users.raissi.setups.common.helpers.network.extern_data import get_extern_data_config

from i6_experiments.users.raissi.setups.common.helpers.train.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)


# user based modules
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo

from i6_experiments.users.raissi.setups.common.decoder.factored_hybrid_search import FactoredHybridBaseDecoder

from i6_experiments.users.raissi.setups.common.decoder.config import PriorInfo, PosteriorScales, SearchParameters

from i6_experiments.users.raissi.setups.common.util.hdf import RasrFeaturesToHdf

# -------------------- Init --------------------

Path = tk.setup_path(__package__)


class ExtraReturnnCode(TypedDict):
    epilog: str
    prolog: str


class Graphs(TypedDict):
    train: Optional[tk.Path]
    inference: Optional[tk.Path]


class ExtraReturnnCode(TypedDict):
    epilog: str
    prolog: str


class TFExperiment(Experiment):
    """
    The class is used in the config files as a single experiment
    """

    extra_returnn_code: ExtraReturnnCode
    name: str
    graph: Graphs
    priors: Optional[PriorInfo]
    prior_job: Optional[returnn.ReturnnRasrComputePriorJobV2]
    returnn_config: Optional[returnn.ReturnnConfig]
    train_job: Optional[returnn.ReturnnRasrTrainingJob]


# -------------------- Systems --------------------
class TFFactoredHybridBaseSystem(BASEFactoredHybridBaseSystem):
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

        self.graphs = {}
        # inference related
        self.native_lstm2_path: Optional[tk.Path] = None

    def _set_native_lstm_path(self):
        compile_native_op_job = returnn.CompileNativeOpJob(
            "NativeLstm2",
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            blas_lib=None,
        )
        self.native_lstm2_path = compile_native_op_job.out_op

    # -------------------- External helpers --------------------
    def get_epilog_for_train(self, specaug_args=None):
        # this is for FH when one needs to define extern data
        if specaug_args is not None:
            spec_augment_epilog = get_specaugment_epilog(**specaug_args)
        else:
            spec_augment_epilog = None
        return get_epilog_code_dense_label(
            n_input=self.initial_nn_args["num_input"],
            n_contexts=self.label_info.n_contexts,
            n_states=self.label_info.n_states_per_phone,
            specaugment=spec_augment_epilog,
        )

    def get_model_checkpoint(self, model_job, epoch):
        return model_job.out_checkpoints[epoch]

    def get_model_path(self, model_job, epoch):
        return model_job.out_checkpoints[epoch].ckpt_path

    def set_local_flf_tool_for_decoding(self, path=None):
        self.csp["base"].flf_tool_exe = path

    # -------------------------------------------- Training --------------------------------------------------------
    def get_config_with_legacy_prolog_and_epilog(
        self,
        config: Dict,
        prolog_additional_str: str = None,
        epilog_additional_str: str = None,
        use_frame_wise_label=True,
    ):
        # this is not a returnn config, but the dict params
        assert self.initial_nn_args["num_input"] is not None, "set the feature input dimension"
        time_prolog, time_tag_name = train_helpers.returnn_time_tag.get_shared_time_tag()
        config["extern_data"] = {
            "data": {
                "dim": self.initial_nn_args["num_input"],
                "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)},
            }
        }
        if use_frame_wise_label:
            config["extern_data"].update(
                **get_extern_data_config(label_info=self.label_info, time_tag_name=time_tag_name)
            )

        # these two are gonna get popped and stored during returnn config object creation
        config["python_prolog"] = {"numpy": "import numpy as np", "time": time_prolog}

        if prolog_additional_str is not None:
            config["python_prolog"]["str"] = prolog_additional_str

        if epilog_additional_str is not None:
            config["python_epilog"] = {"str": epilog_additional_str}

        return config

    # -------------encoder architectures -------------------------------
    def get_blstm_network(self, **kwargs):
        # this is without any loss and output layers
        network = blstm_setup.blstm_network(**kwargs)
        if self.training_criterion != TrainingCriterion.fullsum:
            network = augment_net_with_label_pops(network, label_info=self.label_info)

        return network

    def get_conformer_network(self, chunking, conf_model_dim, aux_loss_args):
        # this only includes auxilaury losses
        network_builder = conformer_setup.get_best_conformer_network(
            conf_model_dim,
            chunking=chunking,
            focal_loss_factor=aux_loss_args["focal_loss_factor"],
            label_smoothing=aux_loss_args["label_smoothing"],
            num_classes=s.label_info.get_n_of_dense_classes(),
        )
        network = network_builder.network
        if self.training_criterion != TrainingCriterion.fullsum:
            network = augment_net_with_label_pops(network, label_info=s.label_info)
        return network

    # -------------------- Decoding --------------------
    def _compute_returnn_rasr_priors(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        share: float,
        time_rqmt: Optional[int] = None,
    ):
        self.set_graph_for_experiment(key)

        model_checkpoint = self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)

        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        if share != 1.0:
            train_crp = copy.deepcopy(train_crp)
            segment_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
                segment_file=train_crp.segment_path,
                split={"priors": share, "rest": 1 - share},
                shuffle=True,
            )
            train_crp.segment_path = segment_job.out_segments["priors"]

        # assert train_data.feature_flow == dev_data.feature_flow
        # assert train_data.features == dev_data.features
        # assert train_data.alignments == dev_data.alignments

        if train_data.feature_flow is not None:
            feature_flow = train_data.feature_flow
        else:
            if isinstance(train_data.features, rasr.FlagDependentFlowAttribute):
                feature_path = train_data.features
            elif isinstance(train_data.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": train_data.features,
                    },
                )
            elif isinstance(train_data.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": train_data.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(train_data.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        if isinstance(train_data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(train_data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(train_data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(train_data.alignments, tk.Path):
            alignments = train_data.alignments
        else:
            raise NotImplementedError

        assert isinstance(returnn_config, returnn.ReturnnConfig)

        prior_job = returnn.ReturnnRasrComputePriorJobV2(
            train_crp=train_crp,
            dev_crp=dev_crp,
            model_checkpoint=model_checkpoint,
            feature_flow=feature_flow,
            alignment=alignments,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            mem_rqmt=12,
            time_rqmt=time_rqmt if time_rqmt is not None else 12,
        )

        return prior_job

    def set_mono_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: Optional[returnn.ReturnnConfig] = None,
        output_layer_name: str = "output",
        data_share: float = 0.1,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        config = copy.deepcopy(returnn_config)
        config.config["forward_output_layer"] = output_layer_name

        job = self._compute_returnn_rasr_priors(
            key,
            epoch,
            train_corpus_key=train_corpus_key,
            dev_corpus_key=dev_corpus_key,
            returnn_config=config,
            share=data_share,
        )

        job.add_alias(f"priors/{name}/c")
        tk.register_output(f"priors/{name}/center-state.xml", job.out_prior_xml_file)

        s.experiments[key]["priors"] = [job.out_prior_xml_file]

    def set_diphone_priors(
        self,
        key,
        epoch,
        tf_library=None,
        nStateClasses=None,
        nContexts=None,
        gpu=1,
        time=20,
        isSilMapped=True,
        hdf_key=None,
    ):
        assert self.label_info.sil_id is not None
        if nStateClasses is None:
            nStateClasses = self.label_info.get_n_state_classes()
        if nContexts is None:
            nContexts = self.label_info.n_contexts

        if tf_library is None:
            tf_library = self.tf_library

        name = f"{self.experiments[key]['name']}-epoch-{epoch}"
        model_checkpoint = self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        graph = self.experiments[key]["graph"]["inference"]

        hdf_paths = self.get_hdf_path(hdf_key)

        estimateJob = EstimateRasrDiphoneAndContextPriors(
            graphPath=graph,
            model=model_checkpoint,
            dataPaths=hdf_paths,
            datasetIndices=list(range(len(hdf_paths) // 3)),
            libraryPath=tf_library,
            nStates=nStateClasses,
            tensorMap=self.tf_map,
            nContexts=nContexts,
            nStateClasses=nStateClasses,
            gpu=gpu,
            time=time,
        )

        estimateJob.add_alias(f"priors-{name}")
        xmlJob = DumpXmlRasrForDiphone(
            estimateJob.diphoneFiles,
            estimateJob.contextFiles,
            estimateJob.numSegments,
            nContexts=nContexts,
            nStateClasses=nStateClasses,
            adjustSilence=isSilMapped,
            silBoundaryIndices=[0, self.label_info.sil_id],
        )

        priorFiles = [xmlJob.diphoneXml, xmlJob.contextXml]
        if name is not None:
            xmlName = f"priors/{name}-xmlpriors"
        else:
            xmlName = "diphone-priors"
        tk.register_output(xmlName, priorFiles[0])
        self.experiments[key]["priors"] = priorFiles

    def set_graph_for_experiment(self, key, override_cfg: Optional[returnn.ReturnnConfig] = None):
        config = copy.deepcopy(override_cfg if override_cfg is not None else self.experiments[key]["returnn_config"])

        name = self.experiments[key]["name"]
        python_prolog = self.experiments[key]["extra_returnn_code"]["prolog"]
        python_epilog = self.experiments[key]["extra_returnn_code"]["epilog"]

        if "source" in config.config["network"].keys():  # specaugment
            for v in config.config["network"].values():
                if v["class"] == "eval":
                    continue
                if v["from"] == "source":
                    v["from"] = "data"
                elif isinstance(v["from"], list):
                    v["from"] = ["data" if val == "source" else val for val in v["from"]]
            del config.config["network"]["source"]

        infer_graph = decode_helpers.compile_graph.compile_tf_graph_from_returnn_config(
            config,
            python_prolog=python_prolog,
            python_epilog=python_epilog,
            returnn_root=self.returnn_root,
        )

        self.experiments[key]["graph"]["inference"] = infer_graph
        tk.register_output(f"graphs/{name}-infer.pb", infer_graph)

    def get_recognizer_and_args(
        self,
        key,
        context_type,
        epoch,
        crp_corpus=None,
        gpu=True,
        is_min_duration=False,
        is_multi_encoder_output=False,
        tf_library=None,
        dummy_mixtures=None,
    ):

        name = ("-").join([self.experiments[key]["name"], crp_corpus, f"e{epoch}-"])
        if context_type.value in [self.context_mapper.get_enum(i) for i in range(6, 9)]:
            name = f'{self.experiments[key]["name"]}-delta-e{epoch}-'

        model_path = self._get_model_path(self.experiments[key]["train_job"], epoch)
        num_encoder_output = self.experiments[key]["returnn_config"].config["network"]["fwd_1"]["n_out"] * 2
        p_info = self._get_prior_info_dict()
        assert self.experiments[key]["priors"] is not None

        isSpecAug = True if "source" in self.experiments[key]["returnn_config"].config["network"].keys() else False
        if context_type.value in [
            self.context_mapper.get_enum(1),
            self.context_mapper.get_enum(7),
        ]:
            if isSpecAug:
                recog_args = get_recog_mono_specAug_args()
            else:
                recog_args = get_recog_mono_args()
            scales = recog_args["priorScales"]
            del recog_args["priorScales"]
            p_info["center-state-prior"]["scale"] = scales["center-state"]
            p_info["center-state-prior"]["file"] = self.experiments[key]["priors"][0]
            recog_args["priorInfo"] = p_info

        elif context_type.value in [self.context_mapper.get_enum(2), self.context_mapper.get_enum(8)]:
            recog_args = get_recog_diphone_fromGmm_specAug_args()
            scales = recog_args["shared_args"]["priorScales"]
            del recog_args["shared_args"]["priorScales"]
            p_info["center-state-prior"]["scale"] = scales["center-state"]
            p_info["left-context-prior"]["scale"] = scales["left-context"]
            p_info["center-state-prior"]["file"] = self.experiments[key]["priors"][0]
            p_info["left-context-prior"]["file"] = self.experiments[key]["priors"][1]
            recog_args["shared_args"]["priorInfo"] = p_info
        else:
            print("implement other contexts")
            assert False

        recog_args["use_word_end_classes"] = self.label_info.use_word_end_classes
        recog_args["n_states_per_phone"] = self.label_info.n_states_per_phone
        recog_args["n_contexts"] = self.label_info.n_contexts
        recog_args["is_min_duration"] = is_min_duration
        recog_args["num_encoder_output"] = num_encoder_output

        if context_type.value not in [
            self.context_mapper.get_enum(1),
            self.context_mapper.get_enum(7),
        ]:
            recog_args["4gram_args"].update(recog_args["shared_args"])
            recog_args["lstm_args"].update(recog_args["shared_args"])

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(
                self.label_info.get_n_of_dense_classes(), self.initial_nn_args["num_input"]
            ).out_mixtures  # gammatones

        assert self.label_info.sil_id is not None

        recognizer = FHDecoder(
            name=name,
            search_crp=self.crp[crp_corpus],
            context_type=context_type,
            context_mapper=self.context_mapper,
            feature_path=self.feature_flows[crp_corpus],
            model_path=model_path,
            graph=self.experiments[key]["graph"]["inference"],
            mixtures=dummy_mixtures,
            eval_files=self.scorer_args[crp_corpus],
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
        )

        return recognizer, recog_args

    def run_decoding_for_cart(
        self,
        name,
        corpus,
        feature_flow,
        feature_scorer,
        tdp_scale=1.0,
        exit_sil=20.0,
        norm_pron=True,
        pron_scale=3.0,
        lm_scale=10.0,
        beam=18.0,
        beam_limit=500000,
        we_pruning=0.8,
        we_pruning_limit=10000,
        altas=None,
        only_lm_opt=True,
    ):

        pre_path = "grid" if (altas is not None and beam < 16.0) else "decoding"

        search_crp = copy.deepcopy(self.crp[corpus])
        search_crp.acoustic_model_config.tdp.scale = tdp_scale
        search_crp.acoustic_model_config.tdp["silence"]["exit"] = exit_sil

        # lm
        search_crp.language_model_config.scale = lm_scale

        name += f"-{corpus}-beaminfo{beam}-{beam_limit}-{we_pruning}"
        name += f"-lmScale-{lm_scale}"
        if tdp_scale != 1.0:
            name += f"_tdpscale-{tdp_scale}"
        if exit_sil != 20.0:
            name += f"_exitSil-{tdp_scale}"

        if altas is not None:
            name += f"_altas-{altas}"
        sp = {
            "beam-pruning": beam,
            "beam-pruning-limit": beam_limit,
            "word-end-pruning": we_pruning,
            "word-end-pruning-limit": we_pruning_limit,
        }

        adv_search_extra_config = None
        if altas is not None:
            adv_search_extra_config = rasr.RasrConfig()
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                altas
            )

        modelCombinationConfig = None
        if norm_pron:
            modelCombinationConfig = rasr.RasrConfig()
            modelCombinationConfig.pronunciation_scale = pron_scale

        search = recog.AdvancedTreeSearchJob(
            crp=search_crp,
            feature_flow=feature_flow,
            feature_scorer=feature_scorer,
            search_parameters=sp,
            lm_lookahead=True,
            eval_best_in_lattice=True,
            use_gpu=True,
            rtf=12,
            mem=8,
            model_combination_config=modelCombinationConfig,
            model_combination_post_config=None,
            extra_config=adv_search_extra_config,
            extra_post_config=None,
        )
        search.rqmt["cpu"] = 2
        if corpus == "russian":
            search.rqmt["time"] = 1

        search.add_alias(f"{pre_path}/recog_{name}")

        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"
        lat2ctm = recog.LatticeToCtmJob(
            crp=search_crp,
            lattice_cache=search.out_lattice_bundle,
            parallelize=True,
            fill_empty_segments=True,
            best_path_algo="bellman-ford",
            extra_config=lat2ctm_extra_config,
        )

        sKwrgs = copy.deepcopy(self.scorer_args[corpus])
        sKwrgs["sort_files"] = True
        sKwrgs[self.scorer_hyp_arg[corpus]] = lat2ctm.out_ctm_file
        scorer = self.scorers[corpus](**sKwrgs)

        self.jobs[corpus]["scorer_%s" % name] = scorer
        tk.register_output(f"{pre_path}/{name}.reports", scorer.out_report_dir)

        if beam > 15.0 and altas is None:
            opt = recog.OptimizeAMandLMScaleJob(
                crp=search_crp,
                lattice_cache=search.out_lattice_bundle,
                initial_am_scale=pron_scale,
                initial_lm_scale=lm_scale,
                scorer_cls=recog.ScliteJob,
                scorer_kwargs=sKwrgs,
                opt_only_lm_scale=only_lm_opt,
            )
            tk.register_output(f"optLM/{name}.onlyLmOpt{only_lm_opt}.optlm.txt", opt.out_log_file)
