from collections import defaultdict
from typing import Dict

from sisyphus import tk

from i6_experiments.common.datasets.tedlium2.constants import CONCURRENT
from i6_experiments.common.datasets.tedlium2.corpus import get_corpus_object_dict
from i6_experiments.common.datasets.tedlium2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from i6_experiments.common.setups.rasr.util import RasrDataInput

from i6_experiments.users.luescher.setups.rasr.config.lex_config import (
    LexiconRasrConfig,
)
from i6_experiments.users.luescher.setups.rasr.config.lm_config import ArpaLmRasrConfig


def get_corpus_data_inputs() -> Dict[str, Dict[str, RasrDataInput]]:
    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    train_lexicon = LexiconRasrConfig(get_g2p_augmented_bliss_lexicon(output_prefix="lexicon"), False)

    wei_lm = ArpaLmRasrConfig(  # TODO replace with complete sisyphus pipeline
        lm_path=tk.Path(
            "/work/asr4/zhou/data/ted-lium2/lm-data/model-kazuki/countLM/4-gram.lm.dev_opt3.bgd.interpolation.pruned.5e-10.gz",
            cached=True,
        ),
    )

    rasr_data_input_dict = defaultdict(dict)

    for name, crp_obj in corpus_object_dict.items():
        rasr_data_input_dict[name][name] = RasrDataInput(
            corpus_object=crp_obj,
            lexicon=train_lexicon.get_dict(),
            concurrent=CONCURRENT[name],
            lm=wei_lm.get_dict() if name == "dev" or name == "test" else None,
        )

    return rasr_data_input_dict
