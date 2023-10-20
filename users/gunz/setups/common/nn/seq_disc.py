import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple

from sisyphus import Path
from sisyphus.delayed_ops import DelayedFormat

from i6_core import corpus, discriminative_training, rasr, returnn


BIGRAM_LM = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/lm/bigram.seq_train.gz"


class Criterion(Enum):
    ME = "me"

    def __str__(self):
        return self.value


@dataclass(frozen=True, eq=True)
class SmbrParameters:
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
    params: SmbrParameters,
) -> Tuple[rasr.RasrConfig, rasr.RasrConfig]:
    assert params.num_classes > 0
    assert params.num_data_dim > 0

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

    feature_flow.apply_config(
        "lattice-processor.rescoring.segmentwise-feature-extraction.feature-extraction", config, post_config
    )

    config.lattice_processor.rescoring.segmentwise_feature_extraction.feature_extraction.file = "feature.flow"

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
    config.lattice_processor.accumulation.criterion = str(params.criterion)
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
    lm: Union[str, Path],
    beam_limit: int,
    lm_scale: float,
    pron_scale: float,
    concurrency: int = 300,
) -> StateAccuracyLatticeAndAlignment:
    assert lm_scale > 0

    crp = copy.deepcopy(crp)

    crp.concurrent = concurrency
    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    crp.language_model_config.file = Path(lm) if isinstance(lm, str) else lm
    crp.language_model_config.type = "ARPA"
    crp.language_model_config.scale = lm_scale
    crp.segment_path = corpus.SegmentCorpusJob(crp.corpus_config.file, concurrency).out_segment_path

    model_combination_cfg = rasr.RasrConfig()
    model_combination_cfg.pronunciation_scale = pron_scale

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
        search_parameters={"beam-pruning": beam_limit},
        rtf=2,
    )
    raw_den_lattice.rqmt["cpu"] = 1
    den_lattice = discriminative_training.DenominatorLatticeJob(
        crp=crp,
        numerator_path=num_lattice.lattice_path,
        raw_denominator_path=raw_den_lattice.lattice_path,
        search_options={"pruning-threshold": beam_limit},
    )
    den_lattice.rqmt["cpu"] = 1
    state_acc = discriminative_training.StateAccuracyJob(
        crp=crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        denominator_path=den_lattice.lattice_path,
        rtf=2,
    )
    state_acc.rqmt["cpu"] = 1

    return StateAccuracyLatticeAndAlignment(
        alignment_bundle=state_acc.segmentwise_alignment_bundle, lattice_bundle=state_acc.lattice_bundle
    )


def augment_for_smbr(
    *,
    crp: rasr.CommonRasrParameters,
    feature_flow: rasr.FlowNetwork,
    feature_scorer: rasr.FeatureScorer,
    returnn_config: returnn.ReturnnConfig,
    beam_limit: int,
    lm_scale: float,
    pron_scale: float,
    smbr_params: SmbrParameters,
    lm_needs_to_be_not_good: Union[Path, str] = BIGRAM_LM,
    from_output_layer: str = "output",
    smbr_layer_name: str = "smbr-accuracy",
    ce_smoothing: float = 0.1,
    loss_like_ce: bool = False,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
    concurrency: int = 300,
    num_rasr_instances: int = 4,
) -> returnn.ReturnnConfig:
    assert concurrency > 0
    assert 0.0 <= ce_smoothing < 1.0
    assert num_rasr_instances > 0

    lattice_data = _generate_lattices(
        crp=crp,
        concurrency=concurrency,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        beam_limit=beam_limit,
        lm=lm_needs_to_be_not_good,
        lm_scale=lm_scale,
        pron_scale=pron_scale,
    )

    config, post_config = _get_smbr_crp(
        alignment_and_lattices=lattice_data,
        crp=crp,
        extra_rasr_config=extra_rasr_config,
        extra_rasr_post_config=extra_rasr_post_config,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        params=smbr_params,
    )

    rasr_cfg_job = rasr.WriteRasrConfigJob(config, post_config)

    network = {
        **returnn_config.config["network"],
        from_output_layer: {
            **returnn_config.config["network"][from_output_layer],
            "loss_scale": ce_smoothing,
        },
        smbr_layer_name: {
            "class": "copy",
            "from": from_output_layer,
            "loss": "sprint",
            "loss_like_ce": loss_like_ce,
            "loss_scale": 1 - ce_smoothing,
            "loss_opts": {
                "sprintConfigStr": DelayedFormat("--config={}", rasr_cfg_job.out_config),
                "sprintControlConfig": {"verbose": True},
                "sprintExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
                "usePythonSegmentOrder": False,
                "numInstances": num_rasr_instances,
            },
        },
    }

    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config.pop("chunking", None)
    returnn_config.config.pop("pretrain", None)
    returnn_config.config["network"] = network

    return returnn_config
