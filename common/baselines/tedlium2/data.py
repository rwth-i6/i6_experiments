from collections import defaultdict
from typing import Dict

from sisyphus import tk
from i6_experiments.common.datasets.tedlium2.constants import CONCURRENT
from i6_experiments.common.datasets.tedlium2.corpus import get_corpus_object_dict
from i6_experiments.common.datasets.tedlium2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from i6_experiments.common.setups.rasr.util import RasrDataInput

from i6_experiments.common.setups.rasr.config.lex_config import (
    LexiconRasrConfig,
)
from i6_experiments.common.setups.rasr.config.lm_config import ArpaLmRasrConfig
from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm


def get_corpus_data_inputs(add_unknown_phoneme_and_mapping: bool = False) -> Dict[str, Dict[str, RasrDataInput]]:
    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")
    train_lexicon = LexiconRasrConfig(
        get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping, output_prefix="lexicon"
        ),
        False,
    )

    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    comb_lm = ArpaLmRasrConfig(lm_path=lm.ngram_lm)

    rasr_data_input_dict = defaultdict(dict)

    for name, crp_obj in corpus_object_dict.items():
        rasr_data_input_dict[name][name] = RasrDataInput(
            corpus_object=crp_obj,
            lexicon=train_lexicon.get_dict(),
            concurrent=CONCURRENT[name],
            lm=comb_lm.get_dict() if name == "dev" or name == "test" else None,
        )

    return rasr_data_input_dict
