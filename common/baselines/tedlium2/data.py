from collections import defaultdict
from typing import Dict

from i6_experiments.common.datasets.tedlium2.constants import CONCURRENT
from i6_experiments.common.datasets.tedlium2.corpus import get_corpus_object_dict
from i6_experiments.common.datasets.tedlium2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from i6_experiments.common.setups.rasr.util import RasrDataInput

from i6_experiments.users.luescher.setups.rasr.config.lex_config import (
    LexiconRasrConfig,
)


def get_corpus_data_inputs() -> Dict[str, Dict[str, RasrDataInput]]:
    corpus_object_dict = get_corpus_object_dict(
        audio_format="wav", output_prefix="corpora"
    )

    train_lexicon = LexiconRasrConfig(
        get_g2p_augmented_bliss_lexicon(output_prefix="lexicon"), False
    )

    rasr_data_input_dict = defaultdict(dict)

    for name, crp_obj in corpus_object_dict.items():
        rasr_data_input_dict[name][name] = RasrDataInput(
            corpus_object=crp_obj,
            lexicon=train_lexicon.get_dict(),
            concurrent=CONCURRENT[name],
        )

    return rasr_data_input_dict
