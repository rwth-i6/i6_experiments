from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_core.meta.system import CorpusObject

from i6_experiments.common.datasets.librispeech import get_corpus_object_dict, get_arpa_lm_dict, get_bliss_lexicon
from i6_experiments.common.setups.rasr import RasrDataInput


def get_corpus_data_inputs():
    """

    :return:
    """

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    lm = {
        "filename": get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }
    lexicon = {
        "filename": get_bliss_lexicon(),
        "normalize_pronunciation": False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs["train-clean-100"] = RasrDataInput(
        corpus_object=corpus_object_dict["train-clean-100"],
        concurrent=10,
        lexicon=lexicon,
        lm=lm,
    )

    for dev_key in ["dev-clean", "dev-other"]:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dict[dev_key], concurrent=10, lexicon=lexicon, lm=lm
        )

    test_data_inputs["test-clean"] = RasrDataInput(
        corpus_object=corpus_object_dict["test-clean"],
        concurrent=10,
        lexicon=lexicon,
        lm=lm,
    )

    return train_data_inputs, dev_data_inputs, test_data_inputs
