from typing import Optional
from i6_core.am.config import acoustic_model_config
import torch
import i6_core.rasr as rasr
from sisyphus import tk


def map_tensor_to_minus1_plus1_interval(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor

    dtype = tensor.dtype
    info = torch.iinfo(dtype)
    min_val = info.min
    max_val = info.max

    return 2.0 * (tensor.float() - min_val) / (max_val - min_val) - 1.0


def make_ctc_loss_config_file(
    lexicon_path: tk.Path,
    corpus_path: tk.Path,
    am_args: dict,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
):
    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()  # type: ignore
    loss_crp.corpus_config.file = corpus_path  # type: ignore
    loss_crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"  # type: ignore

    loss_crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    loss_crp.lexicon_config.file = lexicon_path  # type: ignore

    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_all = True  # type: ignore
    loss_crp.acoustic_model_config.allophones.add_from_lexicon = False  # type: ignore

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        loss_crp,
        mapping,
        parallelize=False,
    )

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True  # type: ignore
    config["*"].fix_allophone_context_at_word_boundaries = True  # type: ignore

    # Automaton manipulation
    if allow_label_loop:
        topology = "ctc"
    else:
        topology = "rna"
    config["*"].allophone_state_graph_builder.topology = topology  # type: ignore

    if min_duration > 1:
        config["*"].allophone_state_graph_builder.label_min_duration = min_duration  # type: ignore

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False  # type: ignore

    config._update(extra_config)
    post_config._update(extra_post_config)

    return rasr.WriteRasrConfigJob(config, post_config).out_config
