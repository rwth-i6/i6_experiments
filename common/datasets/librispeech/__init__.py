"""
This module contains helper functions for the (common) pipeline steps needed to use the LibriSpeech corpus.
It will download and convert the corpus parts that are used in later steps.
(and ONLY those, no unneeded corpus jobs will be registered as output)

The corpora can be accessed in 3 ways:
 - as bliss xml with a specific audio format: get_bliss_corpus_dict
 - as meta.System.CorpusObject with a specific format and duration set: get_corpus_object_dict
 - as ogg zip file (containing .oggs): get_ogg_zip_dict

All corpus functions return a dict with the following keys:
- "dev-clean"
- "dev-other"
- "test-clean"
- "test-other"
- "train-clean-100"
- "train-clean-360"
- "train-clean-460"
- "train-other-500"
- "train-other-960"

Available language models can be accessed with ``get_arpa_lm_dict``:
 - "3gram" for the non-pruned 3-gram LM
 - "4gram" for the non-pruned 4-gram LM

The available lexicas can be accessed with:
 - ``get_bliss_lexicon()`` which returns the original lexicon from OpenSLR, optionally as "folded" version
   with the stress markers removed. Use this lexicon for recognition,
   as otherwise there will be a mismatch with the LM vocabualry.
 - ``get_g2p_augmented_bliss_lexicon_dict()`` which returns a lexicon including the OOVs for the specific training
   dataset. This should be used for training over the "vanilla" lexicon.

If you want to use other subsets (especially with .ogg zips),
please consider to use segment lists to avoid creating new corpus files.

All alias and output paths will be under: ``<output_prefix>/LibriSpeech/....``

For i6-users: physical jobs generated via the "export" functions
are located in: `/work/common/asr/librispeech/data/sisyphus_work_dir/`
"""
from .constants import durations, num_segments
from .corpus import get_bliss_corpus_dict, get_ogg_zip_dict, get_corpus_object_dict
from .export import export_all
from .language_model import get_arpa_lm_dict
from .lexicon import get_bliss_lexicon, get_g2p_augmented_bliss_lexicon_dict
from .vocab import get_lm_vocab, get_subword_nmt_bpe
