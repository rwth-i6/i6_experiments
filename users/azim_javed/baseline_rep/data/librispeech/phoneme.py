from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from sisyphus import tk

from ...tools import returnn_root
from ..base import BlissCorpusToTargetHdfJob
from .lexicon import get_bliss_phoneme_lexicon

PHONEME_SIZE = 79


def get_phoneme_target_hdf_file(corpus_key: str) -> tk.Path:
    bliss_lexicon = get_bliss_phoneme_lexicon()
    bliss_corpus = get_bliss_corpus_dict("wav")[corpus_key]
    hdf_job = BlissCorpusToTargetHdfJob(
        bliss_corpus=bliss_corpus, bliss_lexicon=bliss_lexicon, returnn_root=returnn_root, dim=79
    )
    return hdf_job.out_hdf
