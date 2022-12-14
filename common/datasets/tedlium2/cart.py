__all__ = ["CartQuestions"]

from .constants import BOUNDARY, SILENCE, PHONEMES


class CartQuestions:
    def __init__(
        self,
        max_leaves: int,
        min_obs: int = 1000,
        n_phones: int = 3,
    ):
        self.max_leaves = max_leaves
        self.min_obs = min_obs
        self.n_phones = n_phones
        self.boundary = BOUNDARY
        self.silence = SILENCE
        self.phonemes = PHONEMES
        self.phonemes_str = " ".join(self.phonemes)
        self.phonemes_boundary_silence = (
            [self.boundary] + [self.silence] + self.phonemes
        )
        self.phonemes_boundary_silence_str = " ".join(self.phonemes_boundary_silence)

        assert n_phones in [2, 3], "Only diphone and triphone contexts are allowed"

        self.steps = [
            {
                "name": "silence",
                "action": "cluster",
                "questions": [
                    {
                        "type": "question",
                        "description": "silence",
                        "key": "central",
                        "value": self.silence,
                    }
                ],
            },
            {
                "name": "central",
                "action": "partition",
                "min-obs": self.min_obs,
                "questions": [
                    {
                        "type": "for-each-value",
                        "questions": [
                            {
                                "type": "question",
                                "description": "central-phone",
                                "key": "central",
                            }
                        ],
                    }
                ],
            },
            {
                "name": "hmm-state",
                "action": "partition",
                "min-obs": self.min_obs,
                "questions": [
                    {
                        "type": "for-each-value",
                        "questions": [
                            {
                                "type": "question",
                                "description": "hmm-state",
                                "key": "hmm-state",
                            }
                        ],
                    }
                ],
            },
            {
                "name": "linguistics",
                "min-obs": self.min_obs,
                "questions": [
                    {
                        "type": "for-each-value",
                        "questions": [
                            {
                                "type": "question",
                                "description": "boundary",
                                "key": "boundary",
                            }
                        ],
                    },
                    {
                        "type": "for-each-key",
                        "keys": " ".join(
                            "history[0] central future[0]".split(" ")[:n_phones]
                        ),
                        "questions": [
                            {
                                "type": "for-each-value",
                                "questions": [
                                    {"type": "question", "description": "context-phone"}
                                ],
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT",
                                "values": "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT",
                                "values": "B CH D DH F G HH JH K P S SH T TH V Z ZH",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT-PLOSIVE",
                                "values": "B D G K P T",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT-AFFRICATE",
                                "values": "CH JH",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT-FRICATIVE",
                                "values": "DH F HH S SH TH V Z ZH",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT",
                                "values": "L M N NG R W Y",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT-NASAL",
                                "values": "M N NG",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT-LIQUID",
                                "values": "R L",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT-GLIDE",
                                "values": "W Y",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-APPROX",
                                "values": "R Y",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-BILABIAL",
                                "values": "P B M",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-LABIODENTAL",
                                "values": "F V",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-DENTAL",
                                "values": "TH DH",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-ALVEOLAR",
                                "values": "T D N S Z R L",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-POSTALVEOLAR",
                                "values": "SH ZH",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-VELAR",
                                "values": "K G NG",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL",
                                "values": "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CHECKED",
                                "values": "AE AH EH IH UH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE",
                                "values": "AA AO AW AY ER EY IY OW OY UW",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS1",
                                "values": "AY EY IY OY",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS2",
                                "values": "AW OW UW",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS3",
                                "values": "AA AO ER",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSE",
                                "values": "IY UW IH UH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPEN",
                                "values": "EH ER AH AO AE AA",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENFULL",
                                "values": "AA",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENNEAR",
                                "values": "AE",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENMID",
                                "values": "EH ER AH AO",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSEFULL",
                                "values": "IY UW",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSENEAR",
                                "values": "IH UH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-UNROUNDED",
                                "values": "IY EH AE IH ER AH AA",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-ROUNDED",
                                "values": "UH UW AO",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FRONT",
                                "values": "IY EH AE IH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FRONTNEAR",
                                "values": "IH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CENTRAL",
                                "values": "ER",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-BACK",
                                "values": "UW UH AH AO AA",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-BACKNEAR",
                                "values": "UH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-a",
                                "values": "AW AY",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-U",
                                "values": "UH AW OW",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-I",
                                "values": "IH AY EY OY",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-@",
                                "values": "OW",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-e",
                                "values": "EY",
                            },
                        ],
                    },
                ],
            },
        ]
