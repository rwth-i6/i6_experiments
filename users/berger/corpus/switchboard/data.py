from functools import lru_cache

from sisyphus import tk

import i6_experiments.common.datasets.switchboard as swb_dataset
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.berger.recipe.datasets.switchboard import (
    PreprocessSwitchboardLexiconJob,
)
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
)
from i6_core.meta.system import CorpusObject


def get_corpus_object_dict():
    """
    Download and create a bliss corpus for the Switchboard training corpora and test sets,
    and return all corpora as a dict of CorpusObjects.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str output_prefix:
    :return: A corpus dict with the following entries:
        - 'switchboard-300h'
        - 'hub5e-00'
        - 'hub5-01'
    :rtype: dict[str, CorpusObject]
    """

    corpus_object_dict = {}

    corpus_object = CorpusObject()
    swb_audio_path = tk.Path("/u/corpora/speech/switchboard-1/audio")
    corpus_object.corpus_file = swb_dataset.get_train_bliss_corpus(tk.Path("/u/corpora/speech/switchboard-1/audio"))
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = 311.78
    corpus_object_dict["switchboard-300h"] = corpus_object

    corpus_object = CorpusObject()
    corpus_object.corpus_file = tk.Path("/u/corpora/speech/hub5e_00/xml/hub5e_00.corpus.gz")
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = 3.65
    corpus_object_dict["hub5e-00"] = corpus_object

    # TODO: Hub5-01 currently doesn't exist anymore?
    # corpus_object = CorpusObject()
    # corpus_object.corpus_file = tk.Path("/u/tuske/work/ASR/switchboard/corpus/xml/hub5e_01.corpus.gz")
    # corpus_object.audio_format = "wav"
    # corpus_object.audio_dir = None
    # corpus_object.duration = 6.20
    # corpus_object_dict["hub5-01"] = corpus_object

    return corpus_object_dict


def get_data_inputs(
    train_corpus: str = "switchboard-300h",
    lm_name: str = "fisher_4gram",
    delete_empty_orth: bool = False,
    preprocess_lexicon: bool = True,
):
    corpus_object_dict = get_corpus_object_dict()

    filename = {
        "zoltan_4gram": "/work/asr4/berger/dependencies/switchboard/lm/zoltan_4gram.gz",
        "fisher_4gram": "/home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz",
    }[lm_name]

    lm = {
        "filename": filename,
        "type": "ARPA",
        "scale": 10,
    }

    lexicon_path = swb_dataset.get_bliss_lexicon()
    if preprocess_lexicon:
        lexicon_path = PreprocessSwitchboardLexiconJob(lexicon_path).out_lexicon
    lexicon_path = EnsureSilenceFirstJob(lexicon_path).out_lexicon
    if delete_empty_orth:
        lexicon_path = DeleteEmptyOrthJob(lexicon_path).out_lexicon

    bliss_lexicon = {
        "filename": lexicon_path,
        "normalize_pronunciation": False,
        "add_all": True,
        "add_from_lexicon": False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[train_corpus] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[train_corpus],
        concurrent=100,
        lexicon=bliss_lexicon,
    )

    dev_keys = ["hub5e-00"]
    for dev_key in dev_keys:
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=10,
            lexicon=bliss_lexicon,
            lm=lm,
        )

    # test_keys = ["hub5-01"]
    # for test_key in test_keys:
    #     test_data_inputs[test_key] = rasr_util.RasrDataInput(
    #         corpus_object=corpus_object_dict[test_key],
    #         concurrent=10,
    #         lexicon=bliss_lexicon,
    #         lm=lm
    #         )

    return train_data_inputs, dev_data_inputs, test_data_inputs


def get_final_gmm_output():
    output_args = rasr_util.OutputArgs("final")

    output_args.define_corpus_type("switchboard-300h", "train")
    output_args.define_corpus_type("hub5e-00", "dev")

    output_args.add_feature_to_extract("gt")

    return output_args
