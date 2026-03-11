from i6_experiments.users.berger.args.jobs.cart import CartQuestions


class SWBCartQuestions(CartQuestions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phonemes = [
            "#",
            "[LAUGHTER]",
            "[NOISE]",
            "[SILENCE]",
            "[VOCALIZEDNOISE]",
            "aa",
            "ae",
            "ah",
            "ao",
            "aw",
            "ax",
            "ay",
            "b",
            "ch",
            "d",
            "dh",
            "eh",
            "el",
            "en",
            "er",
            "ey",
            "f",
            "g",
            "hh",
            "ih",
            "iy",
            "jh",
            "k",
            "l",
            "m",
            "n",
            "ng",
            "ow",
            "oy",
            "p",
            "r",
            "s",
            "sh",
            "t",
            "th",
            "uh",
            "uw",
            "v",
            "w",
            "y",
            "z",
            "zh",
        ]

        self.steps = [
            {
                "name": "silence",
                "action": "cluster",
                "questions": [
                    {
                        "type": "question",
                        "description": "silence",
                        "key": "central",
                        "value": "[SILENCE]",
                    }
                ],
            },
            {
                "name": "noise",
                "action": "cluster",
                "questions": [
                    {
                        "type": "question",
                        "description": "noise_[LAUGHTER]",
                        "key": "central",
                        "value": "[LAUGHTER]",
                    },
                    {
                        "type": "question",
                        "description": "noise_[NOISE]",
                        "key": "central",
                        "value": "[NOISE]",
                    },
                    {
                        "type": "question",
                        "description": "noise_[VOCALIZEDNOISE]",
                        "key": "central",
                        "value": "[VOCALIZEDNOISE]",
                    },
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
                    },
                    {
                        "type": "question",
                        "description": "noise",
                        "key": "central",
                        "values": self.unknown,
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
                        "keys": "history[0] central future[0]",
                        "questions": [
                            {
                                "type": "for-each-value",
                                "questions": [{"type": "question", "description": "context-phone"}],
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT",
                                "values": "b ch d dh f g hh jh k l el m n en ng p r s sh t th v w y z zh",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT",
                                "values": "b ch d dh f g hh jh k p s sh t th v z zh",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT-PLOSIVE",
                                "values": "b d g k p t",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT-AFFRICATE",
                                "values": "ch jh",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-OBSTRUENT-FRICATIVE",
                                "values": "dh f hh s sh th v z zh",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT",
                                "values": "l el m n en ng r w y ",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT-NASAL",
                                "values": "m n en ng",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT-LIQUID",
                                "values": "r l el",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-SONORANT-GLIDE",
                                "values": "w y",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-APPROX",
                                "values": "r y",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-BILABIAL",
                                "values": "p b m",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-LABIODENTAL",
                                "values": "f v",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-DENTAL",
                                "values": "th dh",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-ALVEOLAR",
                                "values": "t d n en s z r l el",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-POSTALVEOLAR",
                                "values": "sh zh",
                            },
                            {
                                "type": "question",
                                "description": "CONSONANT-VELAR",
                                "values": "k g ng",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL",
                                "values": "aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CHECKED",
                                "values": "ae ah eh ih uh ",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SHORTCENTRAL",
                                "values": "ax ",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE",
                                "values": "aa ao aw ay er ey iy ow oy uw",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS1",
                                "values": "ay ey iy oy",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS2",
                                "values": "aw ow uw ",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FREE-PHTHONGS3",
                                "values": "aa ao er",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSE",
                                "values": "iy uw ih uh",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPEN",
                                "values": "eh er ah ao ae aa",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENFULL",
                                "values": "aa",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENNEAR",
                                "values": "ae",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-OPENMID",
                                "values": "eh er ah ao",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSEFULL",
                                "values": "iy uw",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CLOSENEAR",
                                "values": "ih uh",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-UNROUNDED",
                                "values": "iy eh ae ih er ah aa",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-ROUNDED",
                                "values": "uh uw ao",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FRONT",
                                "values": "iy eh ae ih",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-FRONTNEAR",
                                "values": "ih",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-CENTRAL",
                                "values": "ax er",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-BACK",
                                "values": "uw uh ah ao aa",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-BACKNEAR",
                                "values": "uh",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-a",
                                "values": "aw ay",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-U",
                                "values": "uh aw ow",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-I",
                                "values": "ih ay ey oy",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-@",
                                "values": "ax ow",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL-SAMPA-e",
                                "values": "ey ",
                            },
                        ],
                    },
                ],
            },
        ]

        if not self.add_unknown:
            self.steps[1]["questions"].pop(1)
