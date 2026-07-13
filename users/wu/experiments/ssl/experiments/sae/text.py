"""SAE Phase 0b — text side: LibriSpeech lexicon + stress-free phoneme inventory.

The inventory here defines the ARPAbet tokens the Qwen3 vocab is extended by (§0b) and the phone set
the decipherment/LM and reward-lookup share. Corpus phonemization (G2P over the LM text) is added on
top of this base lexicon in a later step.
"""

from sisyphus import tk

from i6_core.lexicon.conversion import LexiconToPhonemeListJob
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

PREFIX = "sae/0b"


def librispeech_phoneme_inventory():
    """Folded (stress-free) LibriSpeech bliss lexicon + its phoneme inventory. Returns (lexicon, inventory)."""
    lex = get_bliss_lexicon(use_stress_marker=False, add_unknown_phoneme_and_mapping=True, add_silence=True)
    inventory = LexiconToPhonemeListJob(lex).out_phoneme_list
    tk.register_output(f"{PREFIX}/ls_lexicon_folded.xml.gz", lex)
    tk.register_output(f"{PREFIX}/phoneme_inventory.txt", inventory)
    return lex, inventory
