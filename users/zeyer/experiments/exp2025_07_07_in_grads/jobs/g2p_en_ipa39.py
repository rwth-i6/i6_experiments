"""English word -> vitouphy IPA-39 phoneme sequence, via espeak-ng.

For datasets without phone-level ground truth (e.g. Buckeye word_detail),
this provides the phone target sequence per word for the phoneme-CTC model
(:class:`Wav2Vec2PhonemeCtc`) and the phoneme forced-align baseline.

espeak-ng (`espeak-ng -q -v en-us --ipa=3`) emits IPA with `_` between phonemes;
we strip stress/length/diacritics and fold each espeak phoneme into the
vitouphy 39-IPA inventory (same fold spirit as ``_TIMIT61_TO_IPA``:
e.g. ɔ->ɑ, ʌ->ə, ʒ->ʃ, r->ɹ).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

# Compute nodes lack the system espeak-ng; a relocated copy (binary + libs + voice data)
# lives in the home FS, driven via LD_LIBRARY_PATH / ESPEAK_DATA_PATH.
_ESPEAK_HOME = "/home/az668407/tools/espeak-ng"


def _espeak_exe_env() -> Tuple[str, Optional[Dict[str, str]]]:
    exe = shutil.which("espeak-ng")
    if exe:
        return exe, None
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = _ESPEAK_HOME + "/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["ESPEAK_DATA_PATH"] = _ESPEAK_HOME
    return _ESPEAK_HOME + "/bin/espeak-ng", env


# Strip before lookup: stress, length, syllabicity, ties (incl. the zero-width
# joiner espeak uses inside compound phonemes), nasalization.
_STRIP_CHARS = "ˈˌːˑ̩̃͡ʲ ̪‍"

# espeak-ng en-us phoneme -> vitouphy IPA-39 symbol(s). Multi-symbol values expand.
_ESPEAK_TO_IPA39: Dict[str, Sequence[str]] = {
    # vowels / diphthongs
    "i": ["i"],
    "ɪ": ["ɪ"],
    "e": ["ɛ"],
    "ɛ": ["ɛ"],
    "æ": ["æ"],
    "a": ["æ"],
    "ɑ": ["ɑ"],
    "ɒ": ["ɑ"],
    "ɔ": ["ɑ"],
    "ʌ": ["ə"],
    "ɐ": ["ə"],
    "ə": ["ə"],
    "ɚ": ["ɝ"],
    "ɜ": ["ɝ"],
    "ɝ": ["ɝ"],
    "ʊ": ["ʊ"],
    "u": ["u"],
    "ʉ": ["u"],
    # espeak reduced vowels
    "ᵻ": ["ɪ"],
    "ᵿ": ["ʊ"],
    "aɪ": ["aɪ"],
    "aʊ": ["aʊ"],
    "eɪ": ["eɪ"],
    "oʊ": ["oʊ"],
    "əʊ": ["oʊ"],
    "oː": ["oʊ"],
    "o": ["oʊ"],
    "ɔɪ": ["ɔɪ"],
    "ɪə": ["ɪ", "ə"],
    "eə": ["ɛ", "ə"],
    "ʊə": ["ʊ", "ə"],
    "aɪə": ["aɪ", "ə"],
    "aʊə": ["aʊ", "ə"],
    # consonants
    "p": ["p"],
    "b": ["b"],
    "t": ["t"],
    "d": ["d"],
    "k": ["k"],
    "ɡ": ["g"],
    "g": ["g"],
    "tʃ": ["ʧ"],
    "ʧ": ["ʧ"],
    "dʒ": ["ʤ"],
    "ʤ": ["ʤ"],
    "f": ["f"],
    "v": ["v"],
    "θ": ["θ"],
    "ð": ["ð"],
    "s": ["s"],
    "z": ["z"],
    "ʃ": ["ʃ"],
    "ʒ": ["ʃ"],
    "h": ["h"],
    "m": ["m"],
    "n": ["n"],
    "ŋ": ["ŋ"],
    "l": ["l"],
    "ɫ": ["l"],
    "ɹ": ["ɹ"],
    "r": ["ɹ"],
    "ɾ": ["ɾ"],
    "j": ["j"],
    "w": ["w"],
    # glottal stop: TIMIT folds q to silence; as a G2P target unit just drop it.
    "ʔ": [],
}


class G2pEnIpa39:
    """word -> list of vitouphy IPA-39 symbols, espeak-ng-backed, with a cache."""

    def __init__(self, vocab: Dict[str, int]):
        """:param vocab: the model vocab; every emitted symbol is asserted to be in it."""
        self.vocab = vocab
        self._cache: Dict[str, List[str]] = {}

    def __call__(self, word: str) -> List[str]:
        w = word.lower()
        if w not in self._cache:
            self._cache[w] = self._convert(w)
        return self._cache[w]

    def _convert(self, word: str) -> List[str]:
        exe, env = _espeak_exe_env()
        out = subprocess.run(
            [exe, "-q", "-v", "en-us", "--ipa", "--sep=_", word],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.strip()
        phones: List[str] = []
        # espeak separates words by spaces and phonemes by '_' (ipa=3).
        for group in out.split():
            for raw in group.split("_"):
                ph = "".join(c for c in raw if c not in _STRIP_CHARS)
                if not ph:
                    continue
                mapped = _ESPEAK_TO_IPA39.get(ph)
                if mapped is None:
                    # Unknown compound: fall back to per-char mapping.
                    mapped = []
                    for c in ph:
                        m1 = _ESPEAK_TO_IPA39.get(c)
                        assert m1 is not None, f"g2p: unknown espeak phoneme {ph!r} (char {c!r}) in word {word!r}"
                        mapped = list(mapped) + list(m1)
                for sym in mapped:
                    assert sym in self.vocab, f"g2p: {sym!r} (from {ph!r}, word {word!r}) not in model vocab"
                    phones.append(sym)
        return phones
