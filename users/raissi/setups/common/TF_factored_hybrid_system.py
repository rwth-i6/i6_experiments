__all__ = ["TFFactoredHybridBaseSystem"]

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
from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    BASEFactoredHybridSystem,
    Experiment,
    TrainingCriterion,
)

from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)

import i6_experiments.users.raissi.setups.common.encoder as encoder_archs
import i6_experiments.users.raissi.setups.common.helpers.network as net_helpers
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
import i6_experiments.users.raissi.setups.common.helpers.decode as decode_helpers


# user based modules
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    get_lexicon_args,
    get_tdp_values,
)

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
    BASEFactoredHybridDecoder,
    RasrFeatureScorer,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    SearchParameters,
)

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
class TFFactoredHybridBaseSystem(BASEFactoredHybridSystem):
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
                **net_helpers.extern_data.get_extern_data_config(
                    label_info=self.label_info, time_tag_name=time_tag_name
                )
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
        network = encoder_archs.blstm.blstm_network(**kwargs)
        if self.training_criterion != TrainingCriterion.fullsum:
            network = net_helpers.augment.augment_net_with_label_pops(network, label_info=self.label_info)

        return network

    def get_conformer_network(self, chunking: str, conf_model_dim: int, label_smoothing: float = 0.0, **kwargs):
        # this only includes auxilaury losses
        network_builder = encoder_archs.conformer.get_best_conformer_network(
            size=conf_model_dim,
            num_classes=self.label_info.get_n_of_dense_classes(),
            num_input_feature=self.initial_nn_args["num_input"],
            chunking=chunking,
            label_smoothing=label_smoothing,
            additional_args=kwargs,
        )
        network = network_builder.network
        if self.training_criterion != TrainingCriterion.fullsum:
            network = net_helpers.augment.augment_net_with_label_pops(network, label_info=self.label_info)
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
        checkpoint: Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        model_checkpoint = (
            checkpoint
            if checkpoint is not None
            else self.get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        )

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

    def _compute_returnn_rasr_priors_via_hdf(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        share: float,
        time_rqmt: Union[int, float] = 12,
        checkpoint: Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        model_checkpoint = (
            checkpoint
            if checkpoint is not None
            else self.get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        )
        returnn_config = self.get_hdf_config_from_returnn_rasr_data(
            alignment_allophones=None,
            dev_corpus_key=dev_corpus_key,
            include_alignment=False,
            laplace_ordering=False,
            num_tied_classes=None,
            partition_epochs={"train": 1, "dev": 1},
            returnn_config=returnn_config,
            train_corpus_key=train_corpus_key,
        )

        if share != 1.0:
            segment_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
                segment_file=returnn_config.config["train"]["seq_list_file"],
                split={"priors": share, "rest": 1 - share},
                shuffle=True,
            )
            returnn_config.config["train"]["seq_list_file"] = segment_job.out_segments["priors"]

        prior_job = returnn.ReturnnComputePriorJobV2(
            model_checkpoint=model_checkpoint,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            mem_rqmt=12,
            time_rqmt=time_rqmt,
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

        self.experiments[key]["priors"] = [job.out_prior_xml_file]

    def set_diphone_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: Optional[returnn.ReturnnConfig] = None,
        left_context_output_layer_name: str = "left-output",
        center_state_output_layer_name: str = "center-output",
        data_share: float = 1.0 / 3.0,
        smoothen: bool = False,
        via_hdf: bool = False,
        checkpoint: Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        left_config = get_returnn_config_for_left_context_prior_estimation(
            returnn_config,
            left_context_softmax_layer=left_context_output_layer_name,
        )
        center_config = get_returnn_config_for_center_state_prior_estimation(
            returnn_config,
            label_info=self.label_info,
            center_state_softmax_layer=center_state_output_layer_name,
        )

        prior_jobs = {
            ctx: self._compute_returnn_rasr_priors_via_hdf(
                key=key,
                epoch=epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                checkpoint=checkpoint,
            )
            if via_hdf
            else self._compute_returnn_rasr_priors(
                key,
                epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                checkpoint=checkpoint,
            )
            for (ctx, cfg) in [("l", left_config), ("c", center_config)]
        }

        for (ctx, job) in prior_jobs.items():
            job.add_alias(f"priors/{name}/{ctx}")

        center_priors = ReshapeCenterStatePriorsJob(prior_jobs["c"].out_prior_txt_file, label_info=self.label_info)
        center_priors_xml = center_priors.out_prior_xml

        p_info = PriorInfo(
            center_state_prior=PriorConfig(file=center_priors_xml, scale=0.0),
            left_context_prior=PriorConfig(file=prior_jobs["l"].out_prior_xml_file, scale=0.0),
            right_context_prior=None,
        )
        p_info = smoothen_priors(p_info) if smoothen else p_info

        results = [
            ("center-state", p_info.center_state_prior.file),
            ("left-context", p_info.left_context_prior.file),
        ]
        for context_name, file in results:
            xml_name = f"priors/{name}/{context_name}.xml" if name is not None else f"priors/{context_name}.xml"
            tk.register_output(xml_name, file)

        self.experiments[key]["priors"] = p_info

    def set_triphone_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: Optional[returnn.ReturnnConfig] = None,
        left_context_output_layer_name: str = "left-output",
        center_state_output_layer_name: str = "center-output",
        right_context_output_layer_name: str = "right-output",
        data_share: float = 1.0 / 3.0,
        smoothen: bool = False,
        via_hdf: bool = False,
        checkpoint: Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        left_config = get_returnn_config_for_left_context_prior_estimation(
            returnn_config,
            left_context_softmax_layer=left_context_output_layer_name,
        )
        center_config = get_returnn_config_for_center_state_prior_estimation(
            returnn_config,
            label_info=self.label_info,
            center_state_softmax_layer=center_state_output_layer_name,
        )
        right_configs = get_returnn_configs_for_right_context_prior_estimation(
            returnn_config,
            label_info=self.label_info,
            right_context_softmax_layer=right_context_output_layer_name,
        )

        prior_jobs = {
            ctx: self._compute_returnn_rasr_priors_via_hdf(
                key=key,
                epoch=epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                checkpoint=checkpoint,
            )
            if via_hdf
            else self._compute_returnn_rasr_priors(
                key,
                epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                time_rqmt=8,
                checkpoint=checkpoint,
            )
            for (ctx, cfg) in (
                ("l", left_config),
                ("c", center_config),
                *((f"r{i}", cfg) for i, cfg in enumerate(right_configs)),
            )
        }

        for (ctx, job) in prior_jobs.items():
            job.add_alias(f"priors/{name}/{ctx}")

        center_priors = ReshapeCenterStatePriorsJob(prior_jobs["c"].out_prior_txt_file, label_info=self.label_info)
        center_priors_xml = center_priors.out_prior_xml

        right_priors = [prior_jobs[f"r{i}"].out_prior_txt_file for i in range(len(right_configs))]
        right_prior_xml = JoinRightContextPriorsJob(right_priors, label_info=self.label_info).out_prior_xml

        p_info = PriorInfo(
            center_state_prior=PriorConfig(file=center_priors_xml, scale=0.0),
            left_context_prior=PriorConfig(file=prior_jobs["l"].out_prior_xml_file, scale=0.0),
            right_context_prior=PriorConfig(file=right_prior_xml, scale=0.0),
        )
        p_info = smoothen_priors(p_info) if smoothen else p_info

        results = [
            ("center-state", p_info.center_state_prior.file),
            ("left-context", p_info.left_context_prior.file),
            ("right-context", p_info.right_context_prior.file),
        ]
        for context_name, file in results:
            xml_name = f"priors/{name}/{context_name}.xml" if name is not None else f"priors/{context_name}.xml"
            tk.register_output(xml_name, file)

        self.experiments[key]["priors"] = p_info

    def set_graph_for_experiment(self, key, override_cfg: Optional[returnn.ReturnnConfig] = None):
        config = copy.deepcopy(override_cfg if override_cfg is not None else self.experiments[key]["returnn_config"])

        name = self.experiments[key]["name"]
        python_prolog = self.experiments[key]["extra_returnn_code"]["prolog"]
        python_epilog = self.experiments[key]["extra_returnn_code"]["epilog"]

        if "source" in config.config["network"].keys():  # specaugment
            for v in config.config["network"].values():
                if v["class"] in ["eval", "range"]:
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
            returnn_python_exe=self.returnn_python_exe,
        )

        self.experiments[key]["graph"]["inference"] = infer_graph
        tk.register_output(f"graphs/{name}-infer.pb", infer_graph)

    def setup_returnn_config_and_graph_for_diphone_joint_decoding(
        self, key: str = None, returnn_config: returnn.ReturnnConfig = None
    ):

        # Joint diphone will only work with diphone-dense state-tying
        self.label_info = dataclasses.replace(self.label_info, state_tying=RasrStateTying.diphone)
        for crp_k in self.crp_names.keys():
            if "train" not in crp_k:
                self._update_crp_am_setting_for_decoding(self.crp_names[crp_k])

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        clean_returnn_config = net_helpers.augment.remove_label_pops_and_losses_from_returnn_config(returnn_config)
        context_size = self.label_info.n_contexts
        context_time_tag, _, _ = train_helpers.returnn_time_tag.get_context_dim_tag_prolog(
            spatial_size=context_size,
            feature_size=context_size,
            spatial_dim_variable_name="__center_state_spatial",
            feature_dim_variable_name="__center_state_feature",
            context_type="L",
        )

        # used for decoding
        decoding_returnn_config = net_helpers.diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=clean_returnn_config,
            label_info=self.label_info,
            out_joint_score_layer="output",
            log_softmax=True,
        )
        self.reset_returnn_config_for_experiment(
            key=key,
            config_dict=decoding_returnn_config.config,
            extra_dict_key="context",
            additional_python_prolog=context_time_tag,
        )
        self.set_graph_for_experiment(key)

    def setup_returnn_config_and_graph_for_diphone_joint_prior(
        self, key: str = None, returnn_config: returnn.ReturnnConfig = None
    ):

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        clean_returnn_config = net_helpers.augment.remove_label_pops_and_losses_from_returnn_config(returnn_config)
        context_size = self.label_info.n_contexts
        context_time_tag, _, _ = train_helpers.returnn_time_tag.get_context_dim_tag_prolog(
            spatial_size=context_size,
            feature_size=context_size,
            spatial_dim_variable_name="__center_state_spatial",
            feature_dim_variable_name="__center_state_feature",
            context_type="L",
        )
        # used for decoding
        prior_returnn_config = net_helpers.diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=clean_returnn_config,
            label_info=self.label_info,
            out_joint_score_layer="output",
            log_softmax=False,
        )
        self.reset_returnn_config_for_experiment(
            key=key,
            config_dict=prior_returnn_config.config,
            extra_dict_key="context",
            additional_python_prolog=context_time_tag,
        )
        self.set_graph_for_experiment(key)

    def get_recognizer_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        epoch: int,
        crp_corpus: str,
        recognizer_key: str = "base",
        gpu=True,
        is_multi_encoder_output=False,
        tf_library: Union[tk.Path, str, List[tk.Path], List[str], None] = None,
        dummy_mixtures: Optional[tk.Path] = None,
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

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        recog_args = SearchParameters.default_for_ctx(context_type, priors=p_info)

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(
                self.label_info.get_n_of_dense_classes(),
                self.initial_nn_args["num_input"],
            ).out_mixtures  # gammatones

        assert self.label_info.sil_id is not None

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
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
            lm_gc_simple_hash=lm_gc_simple_hash
            if (lm_gc_simple_hash is not None and lm_gc_simple_hash) or self.lm_gc_simple_hash
            else None,
            **decoder_kwargs,
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
