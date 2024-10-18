__all__ = ["TFFactoredHybridBaseSystem"]

import copy
import dataclasses
import itertools
import numpy as np
from IPython import embed


from typing import Dict, List, Optional, Tuple, TypedDict, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.lexicon as lexicon
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath, MultiOutputPath

# common modules
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)
from i6_experiments.users.mann.experimental.helpers import Train
from i6_experiments.users.raissi.args.system import get_tdp_values

from i6_experiments.users.raissi.setups.common.analysis import (
    ComputeAveragePhonemeLengthJob,
    ComputeSilenceRatioJob,
    ComputeTSEJob,
    PlotViterbiAlignmentsJob,
)

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    BASEFactoredHybridSystem,
    Experiment,
    SingleSoftmaxType,
    TrainingCriterion,
)

import i6_experiments.users.raissi.setups.common.encoder as encoder_archs
import i6_experiments.users.raissi.setups.common.helpers.network as net_helpers
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
import i6_experiments.users.raissi.setups.common.helpers.decode as decode_helpers
from i6_experiments.users.raissi.setups.common.helpers.network import LogLinearScales

from i6_experiments.users.raissi.setups.common.helpers.priors.factored_estimation import get_triphone_priors
from i6_experiments.users.raissi.setups.common.helpers.priors.util import PartitionDataSetup

# user based modules
from i6_experiments.users.raissi.setups.common.data.backend import BackendInfo

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import InputKey

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
    BASEFactoredHybridAligner, DecodingInput,
)

from i6_experiments.users.raissi.setups.common.data.backend import Backend, BackendInfo

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap
from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
    default_posterior_scales,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_dicts.zeineldeen_ted2_global_att_w_ctc_mon import (
    network,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)


class ExtraReturnnCode(TypedDict):
    epilog: str
    prolog: str


class Graphs(TypedDict):
    train: Optional[Path]
    inference: Optional[Path]


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
        returnn_python_exe: Optional[Path] = None,
        rasr_binary_path: Optional[Path] = None,
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
        self.backend_info = BackendInfo.default()
        # inference related
        self.native_lstm2_path: Optional[Path] = None

    # -------------------- External helpers --------------------
    def get_model_checkpoint(self, model_job, epoch):
        return model_job.out_checkpoints[epoch]

    def get_model_path(self, model_job, epoch):
        return model_job.out_checkpoints[epoch].ckpt_path

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

        if self.training_criterion != TrainingCriterion.FULLSUM:
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

        from i6_experiments.users.raissi.setups.common.encoder.conformer.layers import DEFAULT_INIT

        weights_init = DEFAULT_INIT if "weights_init" not in kwargs else kwargs.pop("weights_init")

        network_builder = encoder_archs.conformer.get_best_conformer_network(
            size=conf_model_dim,
            num_classes=self.label_info.get_n_of_dense_classes(),
            num_input_feature=self.initial_nn_args["num_input"],
            time_tag_name=frame_rate_reduction_ratio_info.time_tag_name,
            upsample_by_transposed_conv=self.frame_rate_reduction_ratio_info.factor == 1,
            chunking=chunking,
            label_smoothing=label_smoothing,
            clipping=kwargs.pop("clipping", None),
            weights_init=weights_init,
            additional_args=kwargs,
        )
        network = network_builder.network
        if self.training_criterion != TrainingCriterion.FULLSUM:
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

    def get_conformer_network_zhou_variant(
        self,
        conf_model_dim: int,
        out_layer_name: str = "encoder-output",
        spec_augment_as_data: bool = True,
        auxilary_loss_layers: list = [6],
        frame_rate_reduction_ratio_info: Optional[net_helpers.FrameRateReductionRatioinfo] = None,
    ):

        if frame_rate_reduction_ratio_info is None:
            frame_rate_reduction_ratio_info = self.frame_rate_reduction_ratio_info
        encoder_net = {
            "specaug": {
                "class": "eval",
                "from": "data",
                "eval": f"self.network.get_config().typed_value('transform')(source(0, as_data={spec_augment_as_data}), network=self.network)",
            }
        }
        from_list = encoder_archs.add_initial_conv(network=encoder_net, linear_size=conf_model_dim, from_list="specaug")
        encoder_archs.add_conformer_stack(encoder_net, from_list=from_list)
        encoder_net[out_layer_name] = {
            "class": "copy",
            "from": "conformer_12_output",
            "n_out": conf_model_dim,
        }
        for aux_p in auxilary_loss_layers:
            aux_p_str = f"aux_{aux_p:03d}_"
            encoder_net[f"{aux_p_str}encoder"] = {
                "class": "copy",
                "from": f"conformer_{aux_p}_output",
                "n_out": conf_model_dim,
            }

        if self.training_criterion != TrainingCriterion.FULLSUM:
            if self.training_criterion == TrainingCriterion.VITERBI:
                network = net_helpers.augment.augment_net_with_label_pops(
                    encoder_net,
                    label_info=self.label_info,
                    frame_rate_reduction_ratio_info=frame_rate_reduction_ratio_info,
                )
            elif self.training_criterion == TrainingCriterion.sMBR:
                if frame_rate_reduction_ratio_info.factor > 1:
                    # This layer sets the time step ratio between the input and the output of the NN.

                    frr_factors = (
                        [frame_rate_reduction_ratio_info.factor]
                        if isinstance(frame_rate_reduction_ratio_info.factor, int)
                        else frame_rate_reduction_ratio_info.factor
                    )
                    t_tag = f"{frame_rate_reduction_ratio_info.time_tag_name}"
                    for factor in frr_factors:
                        t_tag += f".ceildiv_right({factor})"

                    encoder_net["classes_"] = {
                        "class": "reinterpret_data",
                        "set_dim_tags": {"T": returnn.CodeWrapper(t_tag)},
                        "from": "classes",
                    }
                    network = encoder_net

            if frame_rate_reduction_ratio_info.factor > 1 and frame_rate_reduction_ratio_info.single_state_alignment:
                network["slice_classes"] = {
                    "class": "slice",
                    "from": network["classes_"]["from"],
                    "axis": "T",
                    "slice_step": frame_rate_reduction_ratio_info.factor,
                }
                network["classes_"]["from"] = "slice_classes"

        else:
            network = encoder_net

        return network

    # -------------------------------------------- Training --------------------------------------------------------
    def add_code_to_extra_returnn_code(self, key: str, extra_key: str, extra_dict_key: str, code: str):
        # extra_key can be either prolog or epilog
        assert extra_dict_key is not None, "set the extra dict key for your additional code"
        old_to_update = copy.deepcopy(self.experiments[key]["extra_returnn_code"][extra_key])
        old_to_update[extra_dict_key] = code
        return old_to_update

    def get_config_with_standard_prolog_and_epilog(
        self,
        config: Dict,
        prolog_additional_str: str = None,
        epilog_additional_str: str = None,
        functions=None,
        label_time_tag: str = None,
        add_extern_data_for_fullsum: bool = False,
    ):
        # this is not a returnn config, but the dict params
        assert self.initial_nn_args["num_input"] is not None, "set the feature input dimension"
        config["extern_data"] = {
            "data": {
                "dim": self.initial_nn_args["num_input"],
                "same_dim_tags_as": {"T": returnn.CodeWrapper(self.frame_rate_reduction_ratio_info.time_tag_name)},
            }
        }
        config["python_prolog"] = {
            "numpy": "import numpy as np",
            "time": self.frame_rate_reduction_ratio_info.get_time_tag_prolog_for_returnn_config(),
        }

        if self.training_criterion != TrainingCriterion.FULLSUM or add_extern_data_for_fullsum:
            if self.frame_rate_reduction_ratio_info.factor == 1:
                label_time_tag = self.frame_rate_reduction_ratio_info.time_tag_name
            config["extern_data"].update(
                **net_helpers.extern_data.get_extern_data_config(
                    label_info=self.label_info,
                    time_tag_name=label_time_tag,
                    add_single_state_label=self.frame_rate_reduction_ratio_info.single_state_alignment,
                )
            )

        if functions is None:
            functions = [
                train_helpers.specaugment.mask,
                train_helpers.specaugment.random_mask,
                train_helpers.specaugment.summary,
                train_helpers.specaugment.transform,
            ]
            if self.backend_info.train == Backend.TF:
                for l in config["network"].keys():
                    if (
                        config["network"][l]["class"] == "eval"
                        and "self.network.get_config().typed_value('transform')" in config["network"][l]["eval"]
                    ):
                        config["network"][l][
                            "eval"
                        ] = "self.network.get_config().typed_value('transform')(source(0), network=self.network)"

        config["python_epilog"] = {
            "functions": functions,
        }

        if prolog_additional_str is not None:
            config["python_prolog"]["str"] = prolog_additional_str

        if epilog_additional_str is not None:
            config["python_epilog"]["str"] = epilog_additional_str

        return config

    def get_config_with_legacy_prolog_and_epilog(
        self,
        config: Dict,
        prolog_additional_str: str = None,
        epilog_additional_str: str = None,
        add_extern_data_for_fullsum=False,
    ):
        # this is not a returnn config, but the dict params
        assert self.initial_nn_args["num_input"] is not None, "set the feature input dimension"

        if self.training_criterion != TrainingCriterion.FULLSUM or add_extern_data_for_fullsum:
            config["extern_data"] = {
                "data": {
                    "dim": self.initial_nn_args["num_input"],
                    "same_dim_tags_as": {"T": returnn.CodeWrapper(self.frame_rate_reduction_ratio_info.time_tag_name)},
                }
            }
            config["python_prolog"] = {
                "numpy": "import numpy as np",
                "time": self.frame_rate_reduction_ratio_info.get_time_tag_prolog_for_returnn_config(),
            }
            label_time_tag = None
            if self.frame_rate_reduction_ratio_info.factor == 1:
                label_time_tag = self.frame_rate_reduction_ratio_info.time_tag_name
            config["extern_data"].update(
                **net_helpers.extern_data.get_extern_data_config(
                    label_info=self.label_info,
                    time_tag_name=label_time_tag,
                    add_single_state_label=self.frame_rate_reduction_ratio_info.single_state_alignment,
                )
            )

        if prolog_additional_str is not None:
            config["python_prolog"]["str"] = prolog_additional_str

        if epilog_additional_str is not None:
            config["python_epilog"] = {"str": epilog_additional_str}

        return config

    def set_returnn_config_for_experiment(self, key: str, config_dict: Dict):
        assert key in self.experiments.keys()

        keep_best_n = (
            config_dict.pop("keep_best_n") if "keep_best_n" in config_dict else self.initial_nn_args["keep_best_n"]
        )
        keep_epochs = (
            config_dict.pop("keep_epochs") if "keep_epochs" in config_dict else self.initial_nn_args["keep_epochs"]
        )
        if None in (keep_best_n, keep_epochs):
            assert False, "either keep_epochs or keep_best_n is None, set this in the initial_nn_args"

        python_prolog = config_dict.pop("python_prolog") if "python_prolog" in config_dict else None
        python_epilog = config_dict.pop("python_epilog") if "python_epilog" in config_dict else None

        base_post_config = {
            "cleanup_old_models": {
                "keep_best_n": keep_best_n,
                "keep": keep_epochs,
            },
        }

        returnn_config = returnn.ReturnnConfig(
            config=config_dict,
            post_config=base_post_config,
            hash_full_python_code=True,
            python_prolog=python_prolog,
            python_epilog=python_epilog,
            sort_config=self.sort_returnn_config,
        )
        self.experiments[key]["returnn_config"] = returnn_config
        self.experiments[key]["extra_returnn_code"]["prolog"] = returnn_config.python_prolog
        self.experiments[key]["extra_returnn_code"]["epilog"] = returnn_config.python_epilog

    def reset_returnn_config_for_experiment(
        self,
        key: str,
        config_dict: Dict,
        extra_dict_key: str = None,
        additional_python_prolog: str = None,
        additional_python_epilog: str = None,
    ):
        if additional_python_prolog is not None:
            python_prolog = self.add_code_to_extra_returnn_code(
                key=key, extra_key="prolog", extra_dict_key=extra_dict_key, code=additional_python_prolog
            )
        else:
            python_prolog = self.experiments[key]["extra_returnn_code"]["prolog"]

        if additional_python_epilog is not None:
            python_epilog = self.add_code_to_extra_returnn_code(
                key=key, extra_key="epilog", extra_dict_key=extra_dict_key, code=additional_python_epilog
            )
        else:
            python_epilog = self.experiments[key]["extra_returnn_code"]["epilog"]

        returnn_config = returnn.ReturnnConfig(
            config=config_dict,
            hash_full_python_code=True,
            python_prolog=python_prolog,
            python_epilog=python_epilog,
            sort_config=self.sort_returnn_config,
        )
        self.experiments[key]["returnn_config"] = returnn_config
        self.experiments[key]["extra_returnn_code"]["prolog"] = returnn_config.python_prolog
        self.experiments[key]["extra_returnn_code"]["epilog"] = returnn_config.python_epilog

    def set_staging_info(
        self, checkpoint: returnn.Checkpoint, copy_param_mode: train_helpers.CopyParamMode.subset, stage_epochs: List
    ) -> Dict:
        self.staging_info = dataclasses.replace(
            self.staging_info, checkpoint=checkpoint, copy_param_mode=copy_param_mode, stage_epochs=stage_epochs
        )

    # -------------------- Decoding --------------------
    def _compute_returnn_rasr_priors(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
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
        train_crp = train_data.get_crp()

        if share != 1.0:
            train_crp = copy.deepcopy(train_crp)
            segment_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
                segment_file=train_crp.segment_path,
                split={"priors": share, "rest": 1 - share},
                shuffle=True,
            )
            train_crp.segment_path = segment_job.out_segments["priors"]

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
            elif isinstance(train_data.features, Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": train_data.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(train_data.features, Path):
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
        state_tying: RasrStateTying = RasrStateTying.monophone,
        returnn_config: Optional[returnn.ReturnnConfig] = None,
        output_layer_name: str = "output",
        joint_for_factored_loss: bool = False,
        checkpoint: Optional[Path] = None,
        smoothen: bool = False,
        zero_weight: float = 1e-8,
        data_share: float = 0.3,
    ):
        # if self.experiments[key]["graph"].get("inference", None) is None:
        #    self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        self.setup_returnn_config_and_graph_for_single_softmax(
            key=key,
            returnn_config=returnn_config,
            state_tying=state_tying,
            softmax_type=SingleSoftmaxType.PRIOR,
            joint_for_factored_loss=joint_for_factored_loss,
        )

        config = copy.deepcopy(self.experiments[key]["returnn_config"])
        config.config["forward_output_layer"] = output_layer_name

        job = self._compute_returnn_rasr_priors(
            key,
            epoch,
            train_corpus_key=train_corpus_key,
            returnn_config=config,
            share=data_share,
            checkpoint=checkpoint,
        )

        job.add_alias(f"priors/{name}/single_prior-{data_share}data")
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

    def set_triphone_priors_factored(
        self,
        key: str,
        epoch: int,
        tensor_map: DecodingTensorMap,
        partition_data_setup: PartitionDataSetup = None,
        model_path: tk.Path = None,
    ):
        self.create_hdf()
        if self.experiments[key]["graph"].get("inference", None) is None:
            self.set_graph_for_experiment(key)
        if partition_data_setup is None:
            partition_data_setup = PartitionDataSetup()

        if model_path is None:
            model_path = DelayedFormat(self.get_model_path(model_job=self.experiments[key]["train_job"], epoch=epoch))
        triphone_priors = get_triphone_priors(
            name=f"{self.experiments[key]['name']}/e{epoch}",
            graph_path=self.experiments[key]["graph"]["inference"],
            model_path=model_path,
            data_paths=self.hdfs[self.train_key],
            tensor_map=tensor_map,
            partition_data_setup=partition_data_setup,
            label_info=self.label_info,
        )

        p_info = PriorInfo(
            center_state_prior=PriorConfig(file=triphone_priors[1], scale=0.0),
            left_context_prior=PriorConfig(file=triphone_priors[2], scale=0.0),
            right_context_prior=PriorConfig(file=triphone_priors[0], scale=0.0),
        )
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
            job.add_alias(f"priors/{name}/{data_share}data/{ctx}")

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

    def setup_returnn_config_and_graph_for_single_softmax(
        self,
        key: str = None,
        returnn_config: returnn.ReturnnConfig = None,
        state_tying: RasrStateTying = RasrStateTying.diphone,
        out_layer_name: str = None,
        softmax_type: SingleSoftmaxType = SingleSoftmaxType.DECODE,
        cv_corpus_key_for_train: str = None,
        joint_for_factored_loss: bool = False,
    ):
        prepare_for_train = False
        log_softmax = False
        if softmax_type in [SingleSoftmaxType.TRAIN, SingleSoftmaxType.DECODE]:
            log_softmax = True
            if softmax_type == SingleSoftmaxType.TRAIN:
                prepare_for_train = True

        assert state_tying in [
            RasrStateTying.monophone,
            RasrStateTying.diphone,
        ], "triphone state tying not possible in precomputed feature scorer due to memory constraint"

        if softmax_type == SingleSoftmaxType.TRAIN:
            assert cv_corpus_key_for_train is not None, "you need to specify the cv corpus for fullsum training"
            assert self.training_criterion in [
                TrainingCriterion.FULLSUM,
                TrainingCriterion.sMBR,
            ], "you forgot to set the correct training criterion"
            self.label_info = dataclasses.replace(self.label_info, state_tying=state_tying)
            self.lexicon_args["norm_pronunciation"] = False
            self.set_rasr_returnn_input_datas(
                input_key=InputKey.BASE,
                is_cv_separate_from_train=True,
                cv_corpus_key=cv_corpus_key_for_train,
            )
            # update all transition models and data
            shift_factor = self.frame_rate_reduction_ratio_info.factor
            tdp_type = "heuristic" if shift_factor == 1 else f"heuristic-{shift_factor}0ms"
            self.update_am_setting_for_all_crps(
                train_tdp_type=tdp_type,
                eval_tdp_type="default",
                add_base_allophones=False,
            )
        else:
            crp_list = [n for n in self.crp_names if "train" not in n or "align" in n]
            self.reset_state_tying(crp_list=crp_list, state_tying=state_tying)

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

            if joint_for_factored_loss:
                f = net_helpers.diphone_joint_output.augment_returnn_config_to_joint_factored_monophone_softmax
            else:
                f = net_helpers.diphone_joint_output.augment_returnn_config_to_joint_diphone_softmax
            final_returnn_config = f(
                returnn_config=clean_returnn_config,
                label_info=self.label_info,
                out_joint_score_layer="output",
                log_softmax=log_softmax,
                prepare_for_train=prepare_for_train,
            )

        elif state_tying == RasrStateTying.monophone:
            final_returnn_config = copy.deepcopy(returnn_config)
            context_time_tag = None
            if log_softmax:
                final_returnn_config.config["network"][out_layer_name] = {
                    **final_returnn_config.config["network"][out_layer_name],
                    "class": "linear",
                    "activation": "log_softmax",
                }

        else:
            assert False, "Only monophone and diphone state tying are supported for single softmax"

        self.reset_returnn_config_for_experiment(
            key=key,
            config_dict=final_returnn_config.config,
            extra_dict_key="context",
            additional_python_prolog=context_time_tag,
        )

        self.set_graph_for_experiment(key, graph_type_name=f"precomputed-{softmax_type}")

    def setup_returnn_config_and_graph_for_precomputed_decoding(
        self,
        key: str = None,
        returnn_config: returnn.ReturnnConfig = None,
        state_tying: RasrStateTying = RasrStateTying.diphone,
        out_layer_name: str = None,
    ):

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
                key=key,
                state_tying=self.label_info.state_tying,
                softmax_type=SingleSoftmaxType.DECODE,
                joint_for_factored_loss=joint_for_factored_loss,
            )
        else:
            crp_list = [n for n in self.crp_names if "train" not in n]
            self.reset_state_tying(crp_list=crp_list, state_tying=self.label_info.state_tying)

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        recog_args = self.get_parameters_for_decoder(context_type=context_type, prior_info=p_info)

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

    def get_lattice_generator_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        crp_corpus: str,
        decoding_input: DecodingInput,
        log_linear_scales: LogLinearScales = None,
        lattice_generator_key: str = "base",
        gpu=False,
        num_encoder_output: int = 512,
        set_batch_major_for_feature_scorer: bool = True,
        tf_library: Union[Path, str, List[Path], List[str], None] = None,
        dummy_mixtures: Optional[Path] = None,
        crp: Optional[rasr.RasrConfig] = None,
        **decoder_kwargs,
    ):

        assert context_type in [
            PhoneticContext.monophone,
            PhoneticContext.joint_diphone,
        ], "only non-autoregressive models are allowed for now"

        recog_args = self.get_parameters_for_decoder(context_type=context_type, prior_info=decoding_input.get_prior())
        posterior_scales = default_posterior_scales()
        if context_type == PhoneticContext.monophone:
            posterior_scales["center-state-scale"] = log_linear_scales.label_posterior_scale
            prior_cfg = recog_args.with_prior_scale(center=log_linear_scales.label_prior_scale)
        else:
            posterior_scales["joint-diphone-scale"] = log_linear_scales.label_posterior_scale
            prior_cfg = recog_args.with_prior_scale(diphone=log_linear_scales.label_prior_scale)

        lattice_generator_cfg = prior_cfg.with_tdp_scale(log_linear_scales.transition_scale).with_posterior_scales(
            posterior_scales
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

        lattice_generator_crp = BASEFactoredHybridAligner.correct_transition_applicator(
            rasr.CommonRasrParameters(self.crp[crp_corpus]) if crp is None else crp
        )

        lattice_generator = self.lattice_generators[lattice_generator_key](
            name=self.experiments[key]["name"],
            crp=lattice_generator_crp,
            context_type=context_type,
            feature_scorer_type=feature_scorer_type,
            feature_path=self.feature_flows[crp_corpus],
            model_path=decoding_input.get_model(),
            graph=decoding_input.get_graph(),
            mixtures=dummy_mixtures,
            tf_library=tf_library,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
            gpu=gpu,
            **decoder_kwargs,
        )

        feature_scorer = lattice_generator.get_feature_scorer(
            label_info=self.label_info, search_parameters=lattice_generator_cfg, num_encoder_output=num_encoder_output
        )

        return lattice_generator, lattice_generator_cfg, feature_scorer

    def calculate_alignment_statistics(
        self,
        key,
        non_speech_labels: [str],
        name: str = None,
        hyp_silence_phone: str = "[SILENCE]",
        ref_silence_phone: str = "[SILENCE]",
        silence_label: str = "[SILENCE]{#+#}@i@f",
        reference_alignment_key: str = "GMM",
        alignment_bundle: tk.Path = None,
        allophones: tk.Path = None,
        reference_alignment: tk.Path = None,
        reference_allophones: tk.Path = None,
        segments: [str] = None,
        use_legacy_tse_calculation: bool = True,
        segment_file: str = None,
    ):
        assert (
            self.experiments[key]["align_job"] is not None or alignment_bundle is not None
        ), "Please set either the alignment job or provide a bundle"

        if reference_alignment is None:
            if reference_alignment_key in self.reference_alignment:
                reference_alignment = self.reference_alignment[reference_alignment_key].get("alignment")
            assert reference_alignment is not None, "Please provide a reference alignment"

        if reference_allophones is None:
            if reference_alignment_key in self.reference_alignment:
                reference_allophones = self.reference_alignment[reference_alignment_key].get("allophones")
            assert reference_allophones is not None, "Please provide a reference allophone file"

        alignment = (
            alignment_bundle
            if alignment_bundle is not None
            else self.experiments[key]["align_job"].out_alignment_bundle
        )
        allophones = (
            allophones
            if allophones is not None
            else lexicon.StoreAllophonesJob(self.crp[self.crp_names["align.train"]]).out_allophone_file
        )
        exp_name = self.experiments[key]["name"] if name is None else name
        if not isinstance(reference_alignment, tk.Path):
            reference_alignment = tk.Path(reference_alignment, cached=True)
        if not isinstance(reference_allophones, tk.Path):
            reference_allophones = tk.Path(reference_allophones, cached=True)

        if use_legacy_tse_calculation:
            tse_job = ComputeTSEJob(
                allophone_file=allophones,
                alignment_cache=alignment,
                ref_allophone_file=reference_allophones,
                ref_alignment_cache=reference_alignment,
                upsample_factor=self.frame_rate_reduction_ratio_info.factor,
            )
            tse_job.add_alias(f"statistics/alignment/{exp_name}/tse")
            tk.register_output(
                f"statistics/alignment/{exp_name}/word_tse",
                tse_job.out_tse_frames,
            )

        else:
            # seems buggy need to debug
            # logging.warn("We do not execute time stamp error until you debugged it ;-)")
            if segment_file is not None:
                tse_job = mm.ComputeTimeStampErrorJobV2(
                    hyp_alignment_cache=alignment,
                    ref_alignment_cache=reference_alignment,
                    hyp_allophone_file=allophones,
                    ref_allophone_file=reference_allophones,
                    hyp_silence_phone=hyp_silence_phone,
                    ref_silence_phone=ref_silence_phone,
                    hyp_upsample_factor=4,
                    segment_file=segment_file,
                )
            else:
                tse_job = mm.ComputeTimeStampErrorJob(
                    hyp_alignment_cache=alignment,
                    ref_alignment_cache=reference_alignment,
                    hyp_allophone_file=allophones,
                    ref_allophone_file=reference_allophones,
                    hyp_silence_phone=hyp_silence_phone,
                    ref_silence_phone=ref_silence_phone,
                    hyp_upsample_factor=4,
                )
            tse_job.rqmt = {
                "time": 4,
                "cpu": 1,
                "mem": 6,
            }

            tse_job.add_alias(f"statistics/alignment/{exp_name}/tse{'_segment' if segment_file is not None else ''}")
            tk.register_output(
                f"statistics/alignment/{exp_name}/word_tse{'_segment' if segment_file is not None else ''}",
                tse_job.out_tse_frames,
            )

        stat_job_phoneme = ComputeAveragePhonemeLengthJob(
            allophone_file=allophones,
            alignment_files=alignment,
            silence_label=silence_label,
            non_speech_labels=non_speech_labels,
        )
        stat_job_phoneme.add_alias(f"statistics/alignment/{exp_name}/average_phoneme_job")
        tk.register_output(
            f"statistics/alignment/{exp_name}/out_average_phoneme_length.txt",
            stat_job_phoneme.out_average_phoneme_length,
        )

        stat_job_sil = ComputeSilenceRatioJob(
            allophone_file=allophones,
            alignment_files=alignment,
            silence_label=silence_label,
        )
        stat_job_sil.add_alias(f"statistics/alignment/{exp_name}/silence_ratio_job")
        tk.register_output(
            f"statistics/alignment/{exp_name}/silence_ratio_silence_ratio.txt",
            stat_job_sil.out_silence_ratio,
        )

        if segments is not None:
            plots = PlotViterbiAlignmentsJob(
                alignment_bundle_path=alignment,
                allophones_path=allophones,
                segments=segments,
                font_size=8,
                show_labels=True,
                monophone=True,
            )
            tk.register_output(f"alignments/plots/{exp_name}", plots.out_plot_folder)

    def get_best_recog_scales_and_transition_values(
        self,
        key: str,
        num_encoder_output: int,
        recog_args: SearchParameters,
        lm_scale: float,
        context_type: PhoneticContext = None,
        feature_scorer_type: RasrFeatureScorer = None,
        tdp_scales: List = None,
        prior_scales: List = None,
        transition_loop_sil: List = None,
        transition_loop_speech: List = None,
        transition_exit_sil: List = None,
        transition_exit_speech: List = None,
        use_heuristic_tdp: bool = False,
        extend: bool = True,
    ) -> SearchParameters:

        assert self.experiments[key]["decode_job"]["runner"] is not None, "Please set the recognizer"
        recognizer = self.experiments[key]["decode_job"]["runner"]

        context_type = PhoneticContext.diphone if context_type is None else context_type
        feature_scorer_type = RasrFeatureScorer.nn_precomputed if feature_scorer_type is None else feature_scorer_type
        if context_type == PhoneticContext.triphone_forward:
            assert feature_scorer_type == feature_scorer_type.factored, "no triphone with nn precomputed yet"

        tdp_scales = [0.1, 0.2] if tdp_scales is None else tdp_scales
        if prior_scales is None:
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
        if use_heuristic_tdp:
            hmm_topology = "monostate" if self.label_info.n_states_per_phone == 1 else "threepartite"
            if self.frame_rate_reduction_ratio_info.factor > 1:
                assert self.frame_rate_reduction_ratio_info.factor in [3, 4], "you have not decided for this ss rate"
                tdps = get_tdp_values()[f"heuristic-{self.frame_rate_reduction_ratio_info.factor}0ms"][hmm_topology]
            else:
                tdps = get_tdp_values()[f"heuristic"][hmm_topology]
            sil_tdp = tdps["silence"]
            sp_tdp = tdps["*"]

        else:
            sil_tdp = (11.0, 0.0, "infinity", 20.0)
            sp_tdp = (8.0, 0.0, "infinity", 0.0)

        best_config_scales = recognizer.recognize_optimize_scales_v2(
            label_info=self.label_info,
            search_parameters=tune_args,
            num_encoder_output=num_encoder_output,
            altas_value=2.0,
            altas_beam=16.0,
            tdp_sil=[sil_tdp],
            tdp_speech=[sp_tdp],
            tdp_nonword=[sp_tdp],
            prior_scales=prior_scales,
            tdp_scales=tdp_scales,
        )

        if use_heuristic_tdp:
            return best_config_scales

        sil_loop = [8.0, 11.0, 13.0]
        if transition_loop_sil is not None:
            if extend:
                sil_loop.extend(transition_loop_sil)
            else:
                sil_loop = transition_loop_sil
        sil_exit = [10.0, 15.0, 20.0]
        if transition_exit_sil is not None:
            if extend:
                sil_exit.extend(transition_exit_sil)
            else:
                sil_exit = transition_exit_sil
        speech_loop = [5.0, 8.0, 11.0]
        if transition_loop_speech is not None:
            if extend:
                speech_loop.extend(transition_loop_speech)
            else:
                speech_loop = transition_loop_speech
        speech_exit = [0.0, 5.0]
        if transition_exit_speech is not None:
            if extend:
                speech_exit.extend(transition_exit_speech)
            else:
                speech_exit = transition_exit_speech

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
