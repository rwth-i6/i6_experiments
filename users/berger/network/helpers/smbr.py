from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from sisyphus import tk


from i6_core.am.config import acoustic_model_config
import i6_core.rasr as rasr
import i6_core.corpus as corpus
import i6_core.discriminative_training as disc_train
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from typing import Dict, List, Optional, Union

from i6_experiments.users.berger.helpers.rasr import LMData
from i6_experiments.users.berger.systems.functors.rasr_base import ToolPaths

@dataclass(frozen=True)
class StateAccuracyLatticeAndAlignment:
    alignment_bundle: tk.Path
    lattice_bundle: tk.Path

def _generate_lattices(
    crp: rasr.CommonRasrParameters,
    feature_flow: rasr.FlowNetwork,
    feature_scorer: rasr.FeatureScorer,
    beam_limit: int,
    pron_scale: float,
    concurrency: int = 300,
    cross_speaker_corpus: Optional[tk.Path] = None
) -> StateAccuracyLatticeAndAlignment:
    lattice_crp = copy.deepcopy(crp)
    lattice_crp.concurrency = concurrency
    lattice_crp.segment_path = corpus.SegmentCorpusJob(crp.corpus_config.file, concurrency).out_segment_path  # type: ignore

    model_combination_cfg = rasr.RasrConfig()
    model_combination_cfg.pronunciation_scale = pron_scale

    num_lattice = disc_train.NumeratorLatticeJob(
        crp=lattice_crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        rtf=2,
    )
    num_lattice.rqmt["mem"] = 8  # type: ignore

    raw_den_lattice = disc_train.RawDenominatorLatticeJob(
        crp=crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        model_combination_config=model_combination_cfg,
        search_parameters={"beam-pruning": beam_limit},
        rtf=2,
        mem=8,
    )

    den_lattice = disc_train.DenominatorLatticeJob(
        crp=crp,
        numerator_path=num_lattice.lattice_path,  # type: ignore
        raw_denominator_path=raw_den_lattice.lattice_path,  # type: ignore
        search_options={"pruning-threshold": beam_limit},
        mem=8,
    )

    if cross_speaker_corpus is not None:
        ...

    state_acc = disc_train.StateAccuracyJob(
        crp=crp,
        feature_flow=feature_flow,
        feature_scorer=feature_scorer,
        denominator_path=den_lattice.lattice_path,  # type: ignore
        rtf=2,
    )

    return StateAccuracyLatticeAndAlignment(
        alignment_bundle=state_acc.segmentwise_alignment_bundle,  # type: ignore
        lattice_bundle=state_acc.lattice_bundle,  # type: ignore
    )


def make_smbr_rasr_loss_config(
    num_classes: int,
    num_data_dim: int,
    tool_paths: ToolPaths,
    feature_scorer: rasr.FeatureScorer,
    feature_flow_lattice_generation: rasr.FlowNetwork,
    feature_flow_smbr_training: rasr.FlowNetwork,
    loss_corpus_path: tk.Path,
    loss_lexicon_path: tk.Path,
    lm: LMData,
    am_args: Dict,
    cross_speaker_corpus: Optional[tk.Path] = None,
    arc_scale: Optional[float] = None,
    criterion: str = "ME",
    margin: float = 0.0,
    posterior_tolerance: int = 100,
    concurrency: int = 300,
    beam_limit: int = 16,
    pron_scale: float = 2.0,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
):
    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = loss_corpus_path  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = loss_lexicon_path  # type: ignore

    loss_crp.language_model_config = lm.get_config(tool_paths=tool_paths)  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore
    loss_crp.acoustic_model_config.tdp.applicator_type = "corrected"
    loss_crp.acoustic_model_config.allophones.add_all = True  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_from_lexicon = False  # type: ignore

    lattice_data = _generate_lattices(
        crp=loss_crp,
        concurrency=concurrency,
        feature_flow=feature_flow_lattice_generation,
        feature_scorer=feature_scorer,
        beam_limit=beam_limit,
        pron_scale=pron_scale,
        cross_speaker_corpus=cross_speaker_corpus,
    )

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

    config, post_config = rasr.build_config_from_mapping(loss_crp, mapping)
    post_config["*"].output_channel.file = "seq-train.log"

    # Define and name actions
    config.lattice_processor.actions = "read,rescore,linear-combination,accumulate-discriminatively"
    config.lattice_processor.selections = "topology-reader,rescoring,linear-combination,accumulation"

    # subtract prior from file
    if feature_scorer.config.prior_file is not None:
        config.lattice_processor.priori_scale = feature_scorer.config.priori_scale
        config.lattice_processor.prior_file = feature_scorer.config.prior_file

    # Sprint Neural Network
    config.lattice_processor.class_labels.number_of_classes = num_classes
    config.lattice_processor.neural_network.links = "0->python-layer:0"
    config.lattice_processor.python_layer.layer_type = "python"
    config.lattice_processor.python_layer.links = "0->bias-layer:0"
    config.lattice_processor.python_layer.dimension_input = num_data_dim
    config.lattice_processor.python_layer.dimension_output = num_classes
    config.lattice_processor.bias_layer.layer_type = "bias"
    config.lattice_processor.bias_layer.dimension_input = num_classes
    config.lattice_processor.bias_layer.dimension_output = num_classes

    # Reader
    config.lattice_processor.topology_reader.readers = "tdps,accuracy"
    config.lattice_processor.topology_reader.lattice_archive.path = lattice_data.lattice_bundle
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
        lattice_data.alignment_bundle
    )
    config.lattice_processor.rescoring.segmentwise_alignment.alignment_cache.read_only = True
    post_config.lattice_processor.rescoring.segmentwise_alignment.model_acceptor_cache.log.channel = "nil"
    post_config.lattice_processor.rescoring.segmentwise_alignment.aligner.statistics.channel = "nil"

    feature_flow_smbr_training.apply_config(
        "lattice-processor.rescoring.segmentwise-feature-extraction.feature-extraction", config, post_config
    )

    written_flow_file = rasr.WriteFlowNetworkJob(feature_flow_smbr_training)
    config.lattice_processor.rescoring.segmentwise_feature_extraction.feature_extraction.file = (
        written_flow_file.out_flow_file  # type: ignore
    )

    # linear-combination
    if arc_scale is None:
        config["*"].LM_SCALE = loss_crp.language_model_config.scale
    else:
        config["*"].LM_SCALE = arc_scale
    config.lattice_processor.linear_combination.outputs = "total accuracy"
    config.lattice_processor.linear_combination.total.scales = ["$[1.0/$(LM-SCALE)]"] * 3 + [margin]
    config.lattice_processor.linear_combination.accuracy.scales = [0.0] * 3 + [1.0]

    # Accumulation
    config.lattice_processor.accumulation.model_type = "neural-network"
    config.lattice_processor.accumulation.criterion = str(criterion)
    config.lattice_processor.accumulation.posterior_tolerance = posterior_tolerance
    config.lattice_processor.accumulation.lattice_name = "total"
    config.lattice_processor.accumulation.port_name = "features"
    config.lattice_processor.accumulation.estimator = "dry-run"
    config.lattice_processor.accumulation.batch_mode = False
    post_config.lattice_processor.accumulation.enable_feature_description_check = False

    # additional config
    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


def make_rasr_smbr_loss_opts(
    tool_paths: ToolPaths,
    rasr_arch: str = "linux-x86_64-standard",
    num_instances: int = 1,
    **kwargs,
):
    assert tool_paths.rasr_binary_path is not None
    lattice_processor_exe = tool_paths.rasr_binary_path.join_right(f"lattice-processor.{rasr_arch}")

    config, post_config = make_smbr_rasr_loss_config(tool_paths=tool_paths, **kwargs)

    rasr_cfg_job = rasr.WriteRasrConfigJob(config, post_config)

    loss_opts = {
        "sprint_opts": {
            "sprintExecPath": lattice_processor_exe,
            "sprintConfigStr": DelayedFormat(
                "--config={}", rasr_cfg_job.out_config  # type: ignore
            ),
            "sprintControlConfig": {"verbose": True},
            "numInstances": num_instances,
            "usePythonSegmentOrder": True,
        },
    }
    return loss_opts


def add_rasr_smbr_output_layer(
    network: Dict,
    num_outputs: int,
    from_list: Union[str, List[str]] = "output",
    name: str = "smbr",
    l2: Optional[float] = None,
    scale: float = 1.0,
    **kwargs,
):
    network[name] = {
        "class": "copy",
        "from": from_list,
        "loss": "sprint",
        "loss_opts": make_rasr_smbr_loss_opts(num_classes=num_outputs, **kwargs),
        "n_out": num_outputs,
    }
    if l2:
        network[name]["L2"] = l2
    if scale != 1:
        network[name]["loss_scale"] = scale

    return name


