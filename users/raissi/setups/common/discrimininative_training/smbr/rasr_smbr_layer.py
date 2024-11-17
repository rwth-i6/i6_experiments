import copy
from dataclasses import dataclass
from enum import Enum
from IPython import embed
from typing import Optional, Union, Tuple

from i6_core import corpus, discriminative_training, rasr, returnn
from sisyphus import Path
from sisyphus.delayed_ops import DelayedFormat

from i6_experiments.common.setups.rasr.config.lm_config import ArpaLmRasrConfig
from i6_experiments.users.raissi.setups.common.decoder.config import SearchParameters
from i6_experiments.users.raissi.setups.common.discrimininative_training.config import BIGRAM_LM
from i6_experiments.users.raissi.setups.common.helpers.network import (
    add_fast_bw_layer_to_returnn_config,
    LogLinearScales,
)


class Criterion(Enum):
    ME = "ME"

    def __str__(self):
        return self.value


@dataclass(frozen=True, eq=True)
class SMBRparameters:
    num_classes: int
    num_data_dim: int

    arc_scale: Optional[float] = None
    criterion: Criterion = Criterion.ME
    margin: float = 0.0
    posterior_tolerance: int = 100


@dataclass(frozen=True)
class StateAccuracyLatticeAndAlignment:
    alignment_bundle: Path
    lattice_bundle: Path


def _get_smbr_crp(
    *,
    alignment_and_lattices: StateAccuracyLatticeAndAlignment,
    crp: rasr.CommonRasrParameters,
    feature_flow: rasr.FlowNetwork,
    feature_scorer: rasr.FeatureScorer,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
    params: SMBRparameters,
) -> Tuple[rasr.RasrConfig, rasr.RasrConfig]:
    assert params.num_classes > 0
    assert params.num_data_dim > 0
    assert crp.acoustic_model_config.tdp.applicator_type == "corrected", "you are using the buggy FSA for alignment"

    mapping = {
        "corpus": "lattice-processor.corpus",
        "lexicon": [
            "lattice-processor.topology-reader.model-combination.lexicon",
            "lattice-processor.rescoring.nn-em-rescorer.model-combination.lexicon",
            "lattice-processor.rescoring.lm-rescorer.model-combination.lexicon",
            "lattice-processor.rescoring.segmentwise-alignment.model-combination.lexicon",
        ],
        "acoustic_model": [
            "lattice-processor.rescoring.nn-em-rescorer.model-combination.acoustic-model",
            "lattice-processor.rescoring.segmentwise-alignment.model-combination.acoustic-model",
        ],
        "language_model": "lattice-processor.rescoring.lm-rescorer.model-combination.lm",
    }

    config, post_config = rasr.build_config_from_mapping(crp, mapping)
    post_config["*"].output_channel.file = "seq-train.log"


    #fixes for FSA and current alignment
    config.lattice_processor.rescoring.segmentwise_alignment.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = (
        False
    )
    config.lattice_processor.rescoring.segmentwise_alignment.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = (
        False
    )
    config.lattice_processor.rescoring.segmentwise_alignment.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = (
        True
    )
    config.lattice_processor.rescoring.segmentwise_alignment.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = (
        True
    )

    # Define and name actions
    config.lattice_processor.actions = "read,rescore,linear-combination,accumulate-discriminatively"
    config.lattice_processor.selections = "topology-reader,rescoring,linear-combination,accumulation"

    # subtract prior from file
    if feature_scorer.config.prior_file is not None:
        config.lattice_processor.priori_scale = feature_scorer.config.priori_scale
        config.lattice_processor.prior_file = feature_scorer.config.prior_file

    # Sprint Neural Network
    config.lattice_processor.class_labels.number_of_classes = params.num_classes
    config.lattice_processor.neural_network.links = "0->python-layer:0"
    config.lattice_processor.python_layer.layer_type = "python"
    config.lattice_processor.python_layer.links = "0->bias-layer:0"
    config.lattice_processor.python_layer.dimension_input = params.num_data_dim
    config.lattice_processor.python_layer.dimension_output = params.num_classes
    config.lattice_processor.bias_layer.layer_type = "bias"
    config.lattice_processor.bias_layer.dimension_input = params.num_classes
    config.lattice_processor.bias_layer.dimension_output = params.num_classes

    # Reader
    config.lattice_processor.topology_reader.readers = "tdps,accuracy"
    config.lattice_processor.topology_reader.lattice_archive.path = alignment_and_lattices.lattice_bundle
    config.lattice_processor.topology_reader.lattice_archive.lm_scale = 0.0
    config.lattice_processor.topology_reader.lattice_archive.pronunciation_scale = 0.0

    # rescoring
    config.lattice_processor.rescoring.nn_emission_rescorers = "nn-em-rescorer"
    config.lattice_processor.rescoring.nn_em_rescorer.port_name = "features"
    config.lattice_processor.rescoring.combined_lm_rescorers = "lm-rescorer"
    config.lattice_processor.rescoring.lm_rescorer.fall_back_value = 10000
    config.lattice_processor.rescoring.pass_extractors = "tdps,accuracy"
    config.lattice_processor.rescoring.segmentwise_feature_extraction.feature_extraction.file = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/train.ss.feature.flow"


    # Parameters for Am::ClassicAcousticModel
    # Feature-scorer not used, place dummy here
    config.lattice_processor.rescoring.segmentwise_alignment.model_combination.acoustic_model.mixture_set.feature_scorer_type = (
        "diagonal-maximum"
    )
    config.lattice_processor.rescoring.segmentwise_alignment.model_combination.acoustic_model.mixture_set.file = (
        feature_scorer.config.file
    )
    config.lattice_processor.rescoring.nn_em_rescorer.model_combination.acoustic_model.mixture_set.feature_scorer_type = (
        "diagonal-maximum"
    )
    config.lattice_processor.rescoring.nn_em_rescorer.model_combination.acoustic_model.mixture_set.file = (
        feature_scorer.config.file
    )

    # rescoring aligner
    config.lattice_processor.rescoring.segmentwise_alignment.port_name = "features"
    config.lattice_processor.rescoring.segmentwise_alignment.alignment_cache.alignment_label_type = "emission-ids"
    config.lattice_processor.rescoring.segmentwise_alignment.alignment_cache.path = (
        alignment_and_lattices.alignment_bundle
    )
    config.lattice_processor.rescoring.segmentwise_alignment.alignment_cache.read_only = True
    post_config.lattice_processor.rescoring.segmentwise_alignment.model_acceptor_cache.log.channel = "nil"
    post_config.lattice_processor.rescoring.segmentwise_alignment.aligner.statistics.channel = "nil"

    # linear-combination
    if params.arc_scale is None:
        config["*"].LM_SCALE = crp.language_model_config.scale
    else:
        config["*"].LM_SCALE = params.arc_scale
    config.lattice_processor.linear_combination.outputs = "total accuracy"
    config.lattice_processor.linear_combination.total.scales = ["$[1.0/$(LM-SCALE)]"] * 3 + [params.margin]
    config.lattice_processor.linear_combination.accuracy.scales = [0.0] * 3 + [1.0]

    # Accumulation
    config.lattice_processor.accumulation.model_type = "neural-network"
    config.lattice_processor.accumulation.criterion = params.criterion.value
    config.lattice_processor.accumulation.posterior_tolerance = params.posterior_tolerance
    config.lattice_processor.accumulation.lattice_name = "total"
    config.lattice_processor.accumulation.port_name = "features"
    config.lattice_processor.accumulation.estimator = "dry-run"
    config.lattice_processor.accumulation.batch_mode = False
    post_config.lattice_processor.accumulation.enable_feature_description_check = False

    # additional config
    config._update(extra_rasr_config)
    post_config._update(extra_rasr_post_config)

    return config, post_config


def _generate_lattices(
    *,
    crp: rasr.CommonRasrParameters,
    feature_flow: rasr.FlowNetwork,
    feature_scorer: rasr.FeatureScorer,
    search_parameters: SearchParameters,
) -> StateAccuracyLatticeAndAlignment:
    assert search_parameters.lm_scale > 0

    crp = copy.deepcopy(crp)
    assert crp.acoustic_model_config.tdp.applicator_type == "corrected", "you are using the buggy FSA for alignment"


    if crp.lexicon_config.normalize_pronunciation and search_parameters.pron_scale is not None:
        model_combination_cfg = rasr.RasrConfig()
        model_combination_cfg.pronunciation_scale = search_parameters.pron_scale
    else:
        model_combination_cfg = None


    num_lattice = discriminative_training.NumeratorLatticeJob(
        crp=crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        rtf=2,
    )
    num_lattice.rqmt["cpu"] = 1
    raw_den_lattice = discriminative_training.RawDenominatorLatticeJob(
        crp=crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        model_combination_config=model_combination_cfg,
        search_parameters={
            "beam-pruning": search_parameters.beam,
            "beam-pruning-limit": search_parameters.beam_limit,
            "word-end-pruning": search_parameters.we_pruning,
            "word-end-pruning-limit": search_parameters.we_pruning_limit,
        },
        rtf=10,
    )
    raw_den_lattice.rqmt["cpu"] = 1
    den_lattice = discriminative_training.DenominatorLatticeJob(
        crp=crp,
        numerator_path=num_lattice.lattice_path,
        raw_denominator_path=raw_den_lattice.lattice_path,
        search_options={"pruning-threshold": search_parameters.beam},
    )

    den_lattice.rqmt["cpu"] = 1
    alignment_options = None

    state_acc = discriminative_training.StateAccuracyJob(
        crp=crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        denominator_path=den_lattice.lattice_path,
        alignment_options=alignment_options,
        rtf=2,
    )
    state_acc.rqmt["cpu"] = 1

    return StateAccuracyLatticeAndAlignment(
        alignment_bundle=state_acc.segmentwise_alignment_bundle, lattice_bundle=state_acc.lattice_bundle
    )


def augment_for_smbr(
    *,
    crp: rasr.CommonRasrParameters,
    feature_flow_lattice: rasr.FlowNetwork,
    feature_flow_training: rasr.FlowNetwork,
    feature_scorer: rasr.FeatureScorer,
    returnn_config: returnn.ReturnnConfig,
    smbr_params: SMBRparameters,
    lattice_search_parameters: SearchParameters,
    training_search_parameters: SearchParameters,
    lattice_generation_lm: Union[Path, str] = BIGRAM_LM,
    training_lm: Union[Path, str] = BIGRAM_LM,
    output_layer: str = "output",
    smbr_layer_name: str = "smbr-accuracy",
    fw_ce_smoothing: float = 0.0,
    fs_ce_smoothing: float = 0.0,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
    num_rasr_instances: int = 1,
) -> returnn.ReturnnConfig:
    assert 0.0 <= fw_ce_smoothing + fs_ce_smoothing < 0.5
    assert num_rasr_instances > 0

    lattice_crp = rasr.CommonRasrParameters(crp)

    lattice_crp.language_model_config = ArpaLmRasrConfig(
        lm_path=(
            Path(lattice_generation_lm, cached=True)
            if isinstance(lattice_generation_lm, str)
            else lattice_generation_lm
        ),
        scale=lattice_search_parameters.lm_scale,
    ).get()


    lattice_data = _generate_lattices(
        crp=lattice_crp,
        feature_flow=feature_flow_lattice,
        feature_scorer=feature_scorer,
        search_parameters=lattice_search_parameters,
    )

    smbr_crp = rasr.CommonRasrParameters(lattice_crp)
    smbr_crp.language_model_config = ArpaLmRasrConfig(
        lm_path=(Path(training_lm, cached=True) if isinstance(training_lm, str) else training_lm),
        scale=training_search_parameters.lm_scale,
    ).get()



    config, post_config = _get_smbr_crp(
        alignment_and_lattices=lattice_data,
        crp=smbr_crp,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,
        feature_flow=feature_flow_training,
        feature_scorer=feature_scorer,
        params=smbr_params,
    )
    rasr_cfg_job = rasr.WriteRasrConfigJob(config, post_config)

    returnn_config.config.pop("chunking", None)
    returnn_config.config["network"][smbr_layer_name] = {
            "class": "copy",
            "from": output_layer,
            "loss": "sprint",
            "loss_scale": 1 - fw_ce_smoothing - fs_ce_smoothing,
            "loss_opts": {
                "sprint_opts": {
                    "sprintConfigStr": DelayedFormat("--config={}", rasr_cfg_job.out_config),
                    "sprintControlConfig": {"verbose": True},
                    "sprintExecPath": rasr.RasrCommand.select_exe(crp.lattice_processor_exe, "lattice-processor"),
                    "usePythonSegmentOrder": True,  # must be true!
                    "numInstances": num_rasr_instances,
                }
            },
        }

    if fw_ce_smoothing > 0:
        loss_info = {"loss": "ce",
            "loss_scale": fw_ce_smoothing,
            "target": "classes",
            "loss_opts": {"focal_loss_factor": 2.0},  # this can be hardcoded by now
        }
        returnn_config.config["network"][output_layer].update(**loss_info)



    if fs_ce_smoothing > 0:
        returnn_config = add_fast_bw_layer_to_returnn_config(
            crp=smbr_crp,
            returnn_config=returnn_config,
            reference_layer=output_layer,
            log_linear_scales=LogLinearScales(
                transition_scale=smbr_crp.acoustic_model_config.tdp.scale,
                label_posterior_scale=feature_scorer.config.scale,
            ),
        )

        returnn_config.config["network"]["output_bw"]["loss_scale"] = fs_ce_smoothing



    return returnn_config
