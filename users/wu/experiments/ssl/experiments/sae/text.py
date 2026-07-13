"""SAE Phase 0b — text side: LibriSpeech lexicon + stress-free phoneme inventory.

The inventory here defines the ARPAbet tokens the Qwen3 vocab is extended by (§0b) and the phone set
the decipherment/LM and reward-lookup share. Corpus phonemization (G2P over the LM text) is added on
top of this base lexicon in a later step.
"""

from sisyphus import tk

from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.g2p.train import TrainG2PModelJob
from i6_core.lexicon.conversion import LexiconToPhonemeListJob
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

PREFIX = "sae/0b"


def folded_lexicon():
    """The stress-free LibriSpeech bliss lexicon (39 ARPAbet + [SILENCE]/[UNKNOWN])."""
    return get_bliss_lexicon(use_stress_marker=False, add_unknown_phoneme_and_mapping=True, add_silence=True)


def librispeech_phoneme_inventory():
    """Folded lexicon + its phoneme inventory. Returns (lexicon, inventory)."""
    lex = folded_lexicon()
    inventory = LexiconToPhonemeListJob(lex).out_phoneme_list
    tk.register_output(f"{PREFIX}/ls_lexicon_folded.xml.gz", lex)
    tk.register_output(f"{PREFIX}/phoneme_inventory.txt", inventory)
    return lex, inventory


def librispeech_g2p_model():
    """Sequitur G2P trained on the folded lexicon (first pronunciation per word → deterministic).
    Used to phonemize OOV words of the LM corpus. Returns the best model tk.Path."""
    g2p_lex = BlissLexiconToG2PLexiconJob(folded_lexicon()).out_g2p_lexicon
    train = TrainG2PModelJob(g2p_lex)  # defaults: 4 ramp-ups; g2p_path/python resolved from gs
    tk.register_output(f"{PREFIX}/g2p_model", train.out_best_model)
    tk.register_output(f"{PREFIX}/g2p_error_rate", train.out_best_error_rate)
    return train.out_best_model
