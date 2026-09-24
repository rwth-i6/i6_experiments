"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/prior.py (phone inventory constants) and
i6_experiments 5207c8adf users/wu/experiments/ssl/analysis/repr_audit.py (gold-phone conventions).

The phone set of the whole phase: the stress-free CMU/ARPAbet monophone set (39) in the fixed order of
``repr_audit.ARPABET_39``, with ``SIL`` appended as the last type (id 39), so every read of the
campaign shares one id space.  ``BOS_ID`` is a context-only symbol of the phone m-gram prior (never a
predicted symbol).

Gold-phone convention (``canonical_phone``): MFA labels are stripped of their trailing stress digits;
the non-speech MFA labels ``_NONSPEECH`` and any label outside the 39-phone set map to ``SIL``.

Both source modules define ``PHONE2ID`` identically (the 39 ARPAbet ids, then ``SIL`` = 39), so one
table serves both.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = [
    "ARPABET_39",
    "SIL",
    "PHONES",
    "PHONE2ID",
    "SIL_ID",
    "N_TYPES",
    "NUM_PHONES",
    "BOS_ID",
    "N_CTX",
    "canonical_phone",
]

# Stress-free CMU/ARPAbet monophone set (39) + silence class. Fixed order -> stable ids across splits.
ARPABET_39 = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
SIL = "SIL"
PHONES: List[str] = ARPABET_39 + [SIL]
PHONE2ID: Dict[str, int] = {p: i for i, p in enumerate(PHONES)}
SIL_ID = PHONE2ID[SIL]
N_TYPES = len(PHONES)  # 40 = 39 ARPAbet + SIL
NUM_PHONES = len(PHONE2ID)  # 40 incl. SIL (repr_audit's name for the same number)
BOS_ID = N_TYPES  # sentence-start context only; never a predicted symbol
N_CTX = N_TYPES + 1

_NONSPEECH = {"sil", "sp", "spn", "", "<unk>", "<eps>", "noise", "nsn", "lau"}


def canonical_phone(label: str) -> str:
    """MFA phone label -> stress-free class. Non-speech -> SIL; strip trailing stress digits."""
    s = str(label)
    if s.lower() in _NONSPEECH:
        return SIL
    s = s.rstrip("0123456789")  # AA0/AA1/AA2 -> AA
    return s if s in PHONE2ID else SIL
