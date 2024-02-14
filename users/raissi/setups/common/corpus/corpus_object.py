from i6_core.meta.system import CorpusObject

@dataclass()
class CorpusData:
    """
    Helper class to define all RasrDataInputs to be passed to the `System` class
    """

    train_data: Dict[str, RasrDataInput]
    dev_data: Dict[str, RasrDataInput]
    test_data: Dict[str, RasrDataInput]


def get_corpus_object(bliss_corpus, duration, audio_dir =None, audio_format="wav"):
    corpus_object = CorpusObject()
    corpus_object.corpus_file = bliss_corpus
    corpus_object.audio_format = audio_format
    corpus_object.audio_dir = audio_dir
    corpus_object.duration = duration