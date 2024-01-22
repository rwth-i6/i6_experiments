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
    AlignmentParameters,
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
    # -------------------- External helpers --------------------
    def get_model_checkpoint(self, model_job, epoch):
        return model_job.out_checkpoints[epoch]

    def get_model_path(self, model_job, epoch):
        return model_job.out_checkpoints[epoch].ckpt_path

    def set_local_flf_tool_for_decoding(self, path=None):
        self.csp["base"].flf_tool_exe = path

    # -------------------------------------------- Training --------------------------------------------------------

    # -------------encoder architectures -------------------------------
    def get_blstm_network(self, **kwargs):
        # this is without any loss and output layers
        network = encoder_archs.blstm.blstm_network(**kwargs)

        if self.frame_rate_reduction_ratio_info.factor != 1:
            assert not self.frame_rate_reduction_ratio_info.factor % 2, "Only even number is supported here"
            network = encoder_archs.blstm.add_subsmapling_via_max_pooling(
                network_dict=network, pool_factor=self.frame_rate_reduction_ratio_info.factor // 2
            )

        if self.training_criterion != TrainingCriterion.fullsum:
            network = net_helpers.augment.augment_net_with_label_pops(
                network,
                label_info=self.label_info,
                frame_rate_reduction_ratio_info=self.frame_rate_reduction_ratio_info,
            )

        return network

    def get_conformer_network(
        self,
        chunking: str,
        conf_model_dim: int,
        frame_rate_reduction_ratio_info: Optional[net_helpers.FrameRateReductionRatioinfo] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        if frame_rate_reduction_ratio_info is None:
            frame_rate_reduction_ratio_info = self.frame_rate_reduction_ratio_info

        frame_rate_args = {
            "reduction_factor": (1, frame_rate_reduction_ratio_info.factor),
        }
        kwargs.update(frame_rate_args)
        # this only includes auxilaury losses
        network_builder = encoder_archs.conformer.get_best_conformer_network(
            size=conf_model_dim,
            num_classes=self.label_info.get_n_of_dense_classes(),
            num_input_feature=self.initial_nn_args["num_input"],
            time_tag_name=frame_rate_reduction_ratio_info.time_tag_name,
            upsample_by_transposed_conv=self.frame_rate_reduction_ratio_info.factor == 1,
            chunking=chunking,
            label_smoothing=label_smoothing,
            additional_args=kwargs,
        )
        network = network_builder.network
        if self.training_criterion != TrainingCriterion.fullsum:
            network = net_helpers.augment.augment_net_with_label_pops(
                network, label_info=self.label_info, frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info
            )
            if frame_rate_reduction_ratio_info.factor > 1 and frame_rate_reduction_ratio_info.single_state_alignment:
                network["slice_classes"] = {
                    "class": "slice",
                    "from": network["classes_"]["from"],
                    "axis": "T",
                    "slice_step": frame_rate_reduction_ratio_info.factor,
                }
                network["classes_"]["from"] = "slice_classes"
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

        if self.experiments[key]["graph"].get("inference", None) is None:
            self.set_graph_for_experiment(key=key)

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

        assert isinstance(returnn_config, returnn.ReturnnConfig)
        if "num_outputs" in returnn_config.config:
            if "classes" in returnn_config.config["num_outputs"]:
                del returnn_config.config["num_outputs"]["classes"]

        prior_job = returnn.ReturnnRasrComputePriorJobV2(
            train_crp=train_crp,
            dev_crp=train_crp,
            model_checkpoint=model_checkpoint,
            feature_flow=feature_flow,
            alignment=None,
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
        if self.experiments[key]["graph"].get("inference", None) is None:
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

    def set_single_prior_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        context_type: PhoneticContext = PhoneticContext.monophone,
        returnn_config: Optional[returnn.ReturnnConfig] = None,
        output_layer_name: str = "output",
        smoothen: bool = False,
        zero_weight: float = 1e-8,
        data_share: float = 0.3,
    ):
        if self.experiments[key]["graph"].get("inference", None) is None:
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

        job.add_alias(f"priors/{name}/single_prior")
        if context_type == PhoneticContext.monophone:
            p_info = PriorInfo(
                center_state_prior=PriorConfig(file=job.out_prior_xml_file, scale=1.0),
            )
            tk.register_output(f"priors/{name}/center-state.xml", p_info.center_state_prior.file)
        elif context_type == PhoneticContext.joint_diphone:
            p_info = PriorInfo(
                diphone_prior=PriorConfig(file=job.out_prior_xml_file, scale=1.0),
            )
            tk.register_output(f"priors/{name}/joint_diphone.xml", p_info.diphone_prior.file)
        else:
            raise NotImplementedError("Unknown PhoneticContext, i.e. context_type")

        p_info = smoothen_priors(p_info, zero_weight=zero_weight) if smoothen else p_info
        self.experiments[key]["priors"] = p_info

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
        if self.experiments[key]["graph"].get("inference", None) is None:
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
        if self.experiments[key]["graph"].get("inference", None) is None:
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

    def set_graph_for_experiment(
        self, key, override_cfg: Optional[returnn.ReturnnConfig] = None, graph_type_name: Optional[str] = None
    ):
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
        if graph_type_name is None:
            graph_type_name = "infer"
        tk.register_output(f"graphs/{name}-{graph_type_name}.pb", infer_graph)

    def setup_returnn_config_and_graph_for_precomputed_decoding(
        self,
        key: str = None,
        returnn_config: returnn.ReturnnConfig = None,
        state_tying: RasrStateTying = RasrStateTying.diphone,
        out_layer_name: str = None,
    ):

        assert state_tying in [
            RasrStateTying.monophone,
            RasrStateTying.diphone,
        ], "triphone state tying not possible in precomputed feature scorer due to memory constraint"
        self.set_state_tying_for_decoder_fsa(state_tying=state_tying)
        if out_layer_name is None:
            out_layer_name = "output" if state_tying == RasrStateTying.diphone else "center-output"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]

        if state_tying == RasrStateTying.diphone:
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
            decoding_returnn_config = net_helpers.diphone_joint_output.augment_returnn_config_to_joint_diphone_softmax(
                returnn_config=clean_returnn_config,
                label_info=self.label_info,
                out_joint_score_layer="output",
                log_softmax=True,
            )
        elif state_tying == RasrStateTying.monophone:
            decoding_returnn_config = copy.deepcopy(returnn_config)
            context_time_tag = None
            decoding_returnn_config.config["network"][out_layer_name] = {
                **decoding_returnn_config.config["network"][out_layer_name],
                "class": "linear",
                "activation": "log_softmax",
            }

        self.reset_returnn_config_for_experiment(
            key=key,
            config_dict=decoding_returnn_config.config,
            extra_dict_key="context",
            additional_python_prolog=context_time_tag,
        )
        self.set_graph_for_experiment(key, graph_type_name="precomputed-infer")

    def setup_returnn_config_and_graph_for_diphone_joint_prior(
        self, key: str = None, returnn_config: returnn.ReturnnConfig = None
    ):

        self.set_state_tying_for_decoder_fsa()
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
        prior_returnn_config = net_helpers.diphone_joint_output.augment_returnn_config_to_joint_diphone_softmax(
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
        self.set_graph_for_experiment(key, graph_type_name="joint-prior")

    def setup_returnn_config_and_graph_for_diphone_joint_training(self, key: str, network: Dict):

        context_size = self.label_info.n_contexts
        context_time_tag, _, _ = train_helpers.returnn_time_tag.get_context_dim_tag_prolog(
            spatial_size=context_size,
            feature_size=context_size,
            spatial_dim_variable_name="__center_state_spatial",
            feature_dim_variable_name="__center_state_feature",
            context_type="L",
        )

        # used for decoding
        decoding_returnn_config = net_helpers.diphone_joint_output.augment_returnn_config_to_joint_diphone_softmax(
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

    def get_recognizer_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        epoch: int,
        crp_corpus: str,
        recognizer_key: str = "base",
        model_path: Optional[tk.Path] = None,
        gpu=False,
        is_multi_encoder_output=False,
        set_batch_major_for_feature_scorer: bool = True,
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

        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        if (
            feature_scorer_type == RasrFeatureScorer.nn_precomputed
            and self.experiments[key]["returnn_config"] is not None
        ):

            self.setup_returnn_config_and_graph_for_precomputed_decoding(
                key=key, state_tying=self.label_info.state_tying
            )

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        recog_args = self.get_parameters_for_decoder(context_type=context_type, prior_info=p_info)

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(
                self.label_info.get_n_of_dense_classes(),
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
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
            lm_gc_simple_hash=lm_gc_simple_hash
            if (lm_gc_simple_hash is not None and lm_gc_simple_hash) or self.lm_gc_simple_hash
            else None,
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
        model_path: Optional[tk.Path] = None,
        feature_path: Optional[tk.Path] = None,
        gpu: bool = False,
        is_multi_encoder_output: bool = False,
        set_batch_major_for_feature_scorer: bool = True,
        tf_library: Union[tk.Path, str, List[tk.Path], List[str], None] = None,
        dummy_mixtures: Optional[tk.Path] = None,
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

            self.setup_returnn_config_and_graph_for_precomputed_decoding(
                key=key, state_tying=self.label_info.state_tying
            )

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
