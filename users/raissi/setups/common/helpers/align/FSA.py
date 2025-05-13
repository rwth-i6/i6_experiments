from dataclasses import dataclass
from typing import Optional

from sisyphus import tk

import i6_core.rasr as rasr


@dataclass(frozen=True, eq=True)
class AlignFSAparameters:
    #we rely on the loop transition
    allow_for_silence_repetitions: Optional[bool] = False
    # for a given word sequence this does not matter and breaks the normalization
    normalize_lemma_sequence_scores: Optional[bool] = False



def correct_rasr_FSA_bug(
    crp: rasr.CommonRasrParameters, apply_lemma_exit_penalty: bool = True
) -> rasr.CommonRasrParameters:
    """
    apply_lemma_exit_penalty: if set to False it, an additional lemma level exit penalty would be added to finial model FSA
                              this is not correct conceptually, since the word sequence is given and the training criterion
                              is constant with respect to this type of penalty
    """
    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    if apply_lemma_exit_penalty:
        transition_types = ["*", "silence"]
        if crp.acoustic_model_config.tdp.tying_type == "global-and-nonword":
            for nw in [0, 1]:
                transition_types.append(f"nonword-{nw}")
        for t in transition_types:
            crp.acoustic_model_config.tdp[t].exit = 0.0

    return crp

def create_rasrconfig_for_alignment_fsa(
    crp: rasr.CommonRasrParameters,
    *,
    extra_rasr_config: Optional[rasr.RasrConfig] = None,
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None,
    align_fsa_parameters: Optional[AlignFSAparameters] = None,
) -> tk.Path:
    mapping = {
        "corpus": "neural-network-trainer.corpus",
        "lexicon": ["neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon"],
        "acoustic_model": ["neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model"],
    }
    config, post_config = rasr.build_config_from_mapping(crp, mapping)
    post_config["*"].output_channel.file = "fastbw.log"

    # Define action
    config.neural_network_trainer.action = "python-control"

    if align_fsa_parameters is None:
        align_fsa_parameters = AlignFSAparameters()

    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions = (
        align_fsa_parameters.allow_for_silence_repetitions
    )
    config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores = (
        align_fsa_parameters.normalize_lemma_sequence_scores
    )
    # Hardcoded since without this the FSA bug solution would not be applied
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries = (
        True
    )
    config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = (
        True
    )

    # additional config
    config._update(extra_rasr_config)
    post_config._update(extra_rasr_post_config)

    automaton_config = rasr.WriteRasrConfigJob(config, post_config).out_config
    tk.register_output("train/bw.config", automaton_config)

    return automaton_config
