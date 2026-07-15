"""SAE Phase 0b — text side: phoneme lexicon + inventory (reuses posterior_hmm).

The LibriSpeech phoneme lexicon / G2P / inventory are the mature, heavily-used implementation in
``posterior_hmm.data.phon``; SAE reuses them instead of duplicating. The non-EOW plain-monophone set
is 39 ARPAbet, matching the frozen-alignment phone set (``repr_audit.ARPABET_39``). The inventory
defines the ARPAbet tokens the Qwen3 vocab is extended by (§0b).
"""

from sisyphus import tk

from i6_core.lexicon.conversion import LexiconToPhonemeListJob
from i6_experiments.users.wu.experiments.posterior_hmm.data.phon import get_phon_lexicon

PREFIX = "sae/0b"


def phon_lexicon() -> tk.Path:
    """Non-EOW plain-monophone (39 ARPAbet) LibriSpeech bliss lexicon (posterior_hmm's provider).

    ``with_g2p=False`` = the base LibriSpeech lexicon (no train-960 G2P augmentation); LM-corpus OOVs
    are covered by the per-corpus Sequitur G2P wired in phonemize.py, matching the original SAE flow.
    """
    return get_phon_lexicon(g2p_librispeech_key=None, with_g2p=False)


def librispeech_phoneme_inventory() -> tk.Path:
    """The stress-free 39-ARPAbet inventory (Qwen3 vocab extension, §0b). Returns the phoneme-list path."""
    inventory = LexiconToPhonemeListJob(phon_lexicon()).out_phoneme_list
    tk.register_output(f"{PREFIX}/phoneme_inventory.txt", inventory)
    return inventory
