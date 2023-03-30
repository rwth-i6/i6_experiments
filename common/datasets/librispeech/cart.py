"""
Contains the prepared CART questions for LibriSpeech,
one variant including the ARPAbet Stress-Markers, the other not.
"""
__all__ = [
    "CartQuestionsWithoutStress",
    "CartQuestionsWithStress",
]

from typing import Dict


class CartQuestionsWithoutStress:
    def __init__(
        self,
        max_leaves: int = 12001,
        min_obs: int = 1000,
        add_unknown: bool = True,
        n_phones: int = 3,
    ):
        self.max_leaves = max_leaves
        self.min_obs = min_obs
        self.boundary = "#"
        self.silence = "[SILENCE]"
        self.unknown = "[UNKNOWN]"
        self.phonemes = [
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "B",
            "CH",
            "D",
            "DH",
            "EH",
            "ER",
            "EY",
            "F",
            "G",
            "HH",
            "IH",
            "IY",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW",
            "OY",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH",
            "UW",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        ]
        self.phonemes_str = " ".join(self.phonemes)
        self.phonemes_boundary = [self.boundary] + self.phonemes
        self.phonemes_boundary_str = " ".join(self.phonemes_boundary)
        self.phonemes_boundary_special = (
            [self.boundary] + [self.silence] + [self.unknown] + self.phonemes
            if add_unknown
            else [self.boundary] + [self.silence] + self.phonemes
        )
        self.phonemes_boundary_special_str = " ".join(self.phonemes_boundary_special)

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
                                "values": self.phonemes_str,
                            }
                        ],
                    },
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
                        "keys": (" ").join("history[0] central future[0]".split(" ")[:n_phones]),
                        "questions": [
                            {
                                "type": "for-each-value",
                                "values": self.phonemes_boundary_str,
                                "questions": [{"type": "question", "description": "context-phone"}],
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
                                "description": "fricat",
                                "values": "F V TH S Z SH ZH HH",
                            },
                            {
                                "type": "question",
                                "description": "voiced",
                                "values": "B D G V Z M N NG L R W JH TH",
                            },
                            {
                                "type": "question",
                                "description": "voiceless",
                                "values": "P T K F S SH CH HH",
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
                                "description": "VOWEL-BACK",
                                "values": "UW UH AH AO AA",
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
                                "description": "pb",
                                "values": "P B",
                            },
                            {
                                "type": "question",
                                "description": "td",
                                "values": "T D",
                            },
                            {
                                "type": "question",
                                "description": "sz",
                                "values": "S Z",
                            },
                            {
                                "type": "question",
                                "description": "kg",
                                "values": "K G",
                            },
                        ],
                    },
                ],
            },
        ]

        if add_unknown:
            unk_dict: Dict[str, str] = {
                "type": "question",
                "description": "unknown",
                "key": "central",
                "values": self.unknown,
            }

            assert self.steps[1]["name"] == "central"
            self.steps[1]["questions"].append(unk_dict)


class CartQuestionsWithStress:
    def __init__(self, max_leaves=12001, min_obs=1000, add_unknown: bool = True, n_phones=3):
        self.max_leaves = max_leaves
        self.min_obs = min_obs
        self.boundary = "#"
        self.silence = "[SILENCE]"
        self.unknown = "[UNKNOWN]"
        self.phonemes = [
            "AA0",
            "AA1",
            "AA2",
            "AE0",
            "AE1",
            "AE2",
            "AH0",
            "AH1",
            "AH2",
            "AO0",
            "AO1",
            "AO2",
            "AW0",
            "AW1",
            "AW2",
            "AY0",
            "AY1",
            "AY2",
            "B",
            "CH",
            "D",
            "DH",
            "EH0",
            "EH1",
            "EH2",
            "ER0",
            "ER1",
            "ER2",
            "EY0",
            "EY1",
            "EY2",
            "F",
            "G",
            "HH",
            "IH0",
            "IH1",
            "IH2",
            "IY0",
            "IY1",
            "IY2",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW0",
            "OW1",
            "OW2",
            "OY0",
            "OY1",
            "OY2",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH0",
            "UH1",
            "UH2",
            "UW0",
            "UW1",
            "UW2",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        ]
        self.phonemes_str = " ".join(self.phonemes)
        self.phonemes_boundary = [self.boundary] + self.phonemes
        self.phonemes_boundary_str = " ".join(self.phonemes_boundary)
        self.phonemes_boundary_special = (
            [self.boundary] + [self.silence] + [self.unknown] + self.phonemes
            if add_unknown
            else [self.boundary] + [self.silence] + self.phonemes
        )
        self.phonemes_boundary_special_str = " ".join(self.phonemes_boundary_special)

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
                                "values": self.phonemes_str,
                            }
                        ],
                    },
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
                        "keys": (" ").join("history[0] central future[0]".split(" ")[:n_phones]),
                        "questions": [
                            {
                                "type": "for-each-value",
                                "values": self.phonemes_boundary_str,
                                "questions": [{"type": "question", "description": "context-phone"}],
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
                                "description": "fricat",
                                "values": "F V TH S Z SH ZH HH",
                            },
                            {
                                "type": "question",
                                "description": "voiced",
                                "values": "B D G V Z M N NG L R W JH TH",
                            },
                            {
                                "type": "question",
                                "description": "voiceless",
                                "values": "P T K F S SH CH HH",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL",
                                "values": "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 IH0 IH1 IH2 IY0 IY1 IY2 OW0 OW1 OW2 OY0 OY1 OY2 UH0 UH1 UH2 UW0 UW1 UW2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CHECKED",
                                "values": "AE0 AE1 AE2 AH0 AH1 AH2 EH0 EH1 EH2 IH0 IH1 IH2 UH0 UH1 UH2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE",
                                "values": "AA0 AA1 AA2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 ER0 ER1 ER2 EY0 EY1 EY2 IY0 IY1 IY2 OW0 OW1 OW2 OY0 OY1 OY2 UW0 UW1 UW2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS1",
                                "values": "AY0 AY1 AY2 EY0 EY1 EY2 IY0 IY1 IY2 OY0 OY1 OY2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS2",
                                "values": "AW0 AW1 AW2 OW0 OW1 OW2 UW0 UW1 UW2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS3",
                                "values": "AA0 AA1 AA2 AO0 AO1 AO2 ER0 ER1 ER2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSE",
                                "values": "IY0 IY1 IY2 UW0 UW1 UW2 IH0 IH1 IH2 UH0 UH1 UH2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPEN",
                                "values": "EH0 EH1 EH2 ER0 ER1 ER2 AH0 AH1 AH2 AO0 AO1 AO2 AE0 AE1 AE2 AA0 AA1 AA2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENFULL",
                                "values": "AA0 AA1 AA2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENNEAR",
                                "values": "AE0 AE1 AE2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENMID",
                                "values": "EH0 EH1 EH2 ER0 ER1 ER2 AH0 AH1 AH2 AO0 AO1 AO2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSEFULL",
                                "values": "IY0 IY1 IY2 UW0 UW1 UW2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSENEAR",
                                "values": "IH0 IH1 IH2 UH0 UH1 UH2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-UNROUNDED",
                                "values": "IY0 IY1 IY2 EH0 EH1 EH2 AE0 AE1 AE2 IH0 IH1 IH2 ER0 ER1 ER2 AH0 AH1 AH2 AA0 AA1 AA2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-ROUNDED",
                                "values": "UH0 UH1 UH2 UW0 UW1 UW2 AO0 AO1 AO2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FRONT",
                                "values": "IY0 IY1 IY2 EH0 EH1 EH2 AE0 AE1 AE2 IH0 IH1 IH2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FRONTNEAR",
                                "values": "IH0 IH1 IH2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CENTRAL",
                                "values": "ER0 ER1 ER2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-BACK",
                                "values": "UW0 UW1 UW2 UH0 UH1 UH2 AH0 AH1 AH2 AO0 AO1 AO2 AA0 AA1 AA2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-BACKNEAR",
                                "values": "UH0 UH1 UH2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-a",
                                "values": "AW0 AW1 AW2 AY0 AY1 AY2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-U",
                                "values": "UH0 UH1 UH2 AW0 AW1 AW2 OW0 OW1 OW2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-I",
                                "values": "IH0 IH1 IH2 AY0 AY1 AY2 EY0 EY1 EY2 OY0 OY1 OY2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-@",
                                "values": "OW0 OW1 OW2",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-e",
                                "values": "EY0 EY1 EY2",
                            },
                            {
                                "type": "question",
                                "description": "stress0",
                                "values": "IY0 AE0 UW0 AA0 EH0 AH0 AO0 IH0 EY0 AW0 AY0 ER0 UH0 OY0 OW0",
                            },
                            {
                                "type": "question",
                                "description": "stress1",
                                "values": "IY1 AE1 UW1 AA1 EH1 AH1 AO1 IH1 EY1 AW1 AY1 ER1 UH1 OY1 OW1",
                            },
                            {
                                "type": "question",
                                "description": "stress2",
                                "values": "IY2 AE2 UW2 AA2 EH2 AH2 AO2 IH2 EY2 AW2 AY2 ER2 UH2 OY2 OW2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_UW",
                                "values": "UW0 UW1 UW2",
                            },
                            {
                                "type": "question",
                                "description": "pb",
                                "values": "P B",
                            },
                            {
                                "type": "question",
                                "description": "vowel_UH",
                                "values": "UH0 UH1 UH2",
                            },
                            {
                                "type": "question",
                                "description": "td",
                                "values": "T D",
                            },
                            {
                                "type": "question",
                                "description": "vowel_AW",
                                "values": "AW0 AW1 AW2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_AY",
                                "values": "AY0 AY1 AY2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_AA",
                                "values": "AA0 AA1 AA2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_AE",
                                "values": "AE0 AE1 AE2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_AH",
                                "values": "AH0 AH1 AH2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_AO",
                                "values": "AO0 AO1 AO2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_IY",
                                "values": "IY0 IY1 IY2",
                            },
                            {
                                "type": "question",
                                "description": "sz",
                                "values": "S Z",
                            },
                            {
                                "type": "question",
                                "description": "kg",
                                "values": "K G",
                            },
                            {
                                "type": "question",
                                "description": "vowel_EH",
                                "values": "EH0 EH1 EH2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_OY",
                                "values": "OY0 OY1 OY2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_OW",
                                "values": "OW0 OW1 OW2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_IH",
                                "values": "IH0 IH1 IH2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_EY",
                                "values": "EY0 EY1 EY2",
                            },
                            {
                                "type": "question",
                                "description": "vowel_ER",
                                "values": "ER0 ER1 ER2",
                            },
                        ],
                    },
                ],
            },
        ]

        if add_unknown:
            unk_dict: Dict[str, str] = {
                "type": "question",
                "description": "unknown",
                "key": "central",
                "values": self.unknown,
            }

            assert self.steps[1]["name"] == "central"
            self.steps[1]["questions"].append(unk_dict)
