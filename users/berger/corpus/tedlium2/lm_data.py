from i6_experiments.users.berger.helpers import rasr_lm_config
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs


def get_lm(name: str) -> rasr_lm_config.LMData:
    lm_dict = {}

    ted_4gram = get_corpus_data_inputs()["dev"]["dev"].lm
    assert ted_4gram is not None

    lm_dict["4gram"] = rasr_lm_config.ArpaLMData(filename=ted_4gram["filename"], scale=ted_4gram.get("scale", 1.0))

    return lm_dict[name]
