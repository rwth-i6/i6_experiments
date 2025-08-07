from sisyphus import tk

from ...tools import returnn_root
from ..base import BlissCorpusToTargetHdfJob
from .lexicon import get_bliss_phoneme_lexicon

PHONEME_SIZE = 88


def get_phoneme_target_hdf_file(bliss_corpus_file: tk.Path) -> tk.Path:
    bliss_lexicon = get_bliss_phoneme_lexicon()
    hdf_job = BlissCorpusToTargetHdfJob(
        bliss_corpus=bliss_corpus_file, bliss_lexicon=bliss_lexicon, returnn_root=returnn_root, dim=PHONEME_SIZE
    )
    return hdf_job.out_hdf
