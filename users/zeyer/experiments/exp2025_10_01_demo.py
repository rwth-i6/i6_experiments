
from sisyphus import tk
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

def py():
    path = get_bliss_lexicon()
    tk.register_output("datasets/LibriSpeech/bliss_lexicon.xml.gz", path)
