from i6_experiments.users.berger.args.jobs.cart import CartQuestions


class WSJCartQuestions(CartQuestions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phonemes = [
            "#",
            "[SILENCE]",
            "aa",
            "ae",
            "ah",
            "aw",
            "awh",
            "ee",
            "eh",
            "ey",
            "ih",
            "oh",
            "oo",
            "ooh",
            "ow",
            "UH",
            "ur",
            "b",
            "ch",
            "d",
            "dh",
            "f",
            "g",
            "h",
            "j",
            "k",
            "l",
            "m",
            "n",
            "ng",
            "p",
            "r",
            "s",
            "sh",
            "t",
            "th",
            "uh",
            "ul",
            "um",
            "un",
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
                        "keys": "history[0] future[0]",
                        "questions": [
                            {
                                "type": "for-each-value",
                                "questions": [{"type": "question", "description": "context-phone"}],
                            },
                            {
                                "type": "question",
                                "description": "ALVEOLAR-STOP",
                                "values": "d t",
                            },
                            {
                                "type": "question",
                                "description": "LABIAL-STOP",
                                "values": "b p",
                            },
                            {
                                "type": "question",
                                "description": "DENTAL",
                                "values": "dh th",
                            },
                            {
                                "type": "question",
                                "description": "LIQUID",
                                "values": "l r ur",
                            },
                            {"type": "question", "description": "LW", "values": "l w"},
                            {
                                "type": "question",
                                "description": "S/SH",
                                "values": "s sh",
                            },
                            {
                                "type": "question",
                                "description": "VELAR-STOP",
                                "values": "g k",
                            },
                            {
                                "type": "question",
                                "description": "LQGL-BACK",
                                "values": "l r ur w",
                            },
                            {
                                "type": "question",
                                "description": "NASAL",
                                "values": "m n",
                            },
                            {
                                "type": "question",
                                "description": "L-NASAL",
                                "values": "m n um un",
                            },
                            {
                                "type": "question",
                                "description": "R-NASAL",
                                "values": "m n ng",
                            },
                            {
                                "type": "question",
                                "description": "L-VELAR",
                                "values": "ng g k",
                            },
                            {
                                "type": "question",
                                "description": "R-VELAR",
                                "values": "g k",
                            },
                            {
                                "type": "question",
                                "description": "R-VOICELESS-FRIC",
                                "values": "th s sh f",
                            },
                            {
                                "type": "question",
                                "description": "L-VOICELESS-FRIC",
                                "values": "th s sh f j",
                            },
                            {
                                "type": "question",
                                "description": "L-VOICED-FRIC",
                                "values": "dh z v j zh ch",
                            },
                            {
                                "type": "question",
                                "description": "R-VOICED-FRIC",
                                "values": "dh z v j zh ch",
                            },
                            {
                                "type": "question",
                                "description": "LIQUID-GLIDE",
                                "values": "s z sh zh",
                            },
                            {
                                "type": "question",
                                "description": "S/Z/SH/ZH",
                                "values": "s z sh zh",
                            },
                            {
                                "type": "question",
                                "description": "W-GLIDE",
                                "values": "aw awh ow w",
                            },
                            {
                                "type": "question",
                                "description": "PALATL",
                                "values": "y ch sh",
                            },
                            {
                                "type": "question",
                                "description": "Y-GLIDE",
                                "values": "ey y",
                            },
                            {
                                "type": "question",
                                "description": "L-LABIAL",
                                "values": "w m b p v f um",
                            },
                            {
                                "type": "question",
                                "description": "R-LABIAL",
                                "values": "w m b p v f",
                            },
                            {
                                "type": "question",
                                "description": "HIGH-VOWEL",
                                "values": "ee ih UH uh y",
                            },
                            {
                                "type": "question",
                                "description": "LAX-VOWEL",
                                "values": "ee eh ih UH uh ah oh ooh oo awh",
                            },
                            {
                                "type": "question",
                                "description": "LOW-VOWEL",
                                "values": "ae aa aw",
                            },
                            {
                                "type": "question",
                                "description": "ORAL-STOP2",
                                "values": "p t k",
                            },
                            {
                                "type": "question",
                                "description": "R-ORAL-STOP3",
                                "values": "b d g",
                            },
                            {
                                "type": "question",
                                "description": "L-ORAL-STOP3",
                                "values": "b d g ng",
                            },
                            {
                                "type": "question",
                                "description": "ALVEOLAR",
                                "values": "n d t s z un",
                            },
                            {
                                "type": "question",
                                "description": "DIPHTHONG",
                                "values": "aw awh ey ow",
                            },
                            {
                                "type": "question",
                                "description": "R-FRICATIVE",
                                "values": "dh th s sh z v f zh",
                            },
                            {
                                "type": "question",
                                "description": "L-FRICATIVE",
                                "values": "dh th s sh z v f zh j ch",
                            },
                            {
                                "type": "question",
                                "description": "ROUND-VOCALIC",
                                "values": "UH uh ow w",
                            },
                            {
                                "type": "question",
                                "description": "FRONT-R",
                                "values": "ae ee eh ih ey ah y aw awh",
                            },
                            {
                                "type": "question",
                                "description": "TENSE-VOWEL",
                                "values": "ey ae ow aa aw awh",
                            },
                            {
                                "type": "question",
                                "description": "BACK-L",
                                "values": "UH uh ow aa l r ur w aw awh oh ooh oo",
                            },
                            {
                                "type": "question",
                                "description": "FRONT-L",
                                "values": "ae ee eh ih ey ah y",
                            },
                            {
                                "type": "question",
                                "description": "BACK-R",
                                "values": "UH uh ow aa l r ur w oh ooh oo",
                            },
                            {
                                "type": "question",
                                "description": "ORAL-STOP1",
                                "values": "b d g p ch j",
                            },
                            {
                                "type": "question",
                                "description": "VOWEL",
                                "values": "ae ee eh ih UH uh ah aa aw ey ow oh ooh oo awh",
                            },
                            {
                                "type": "question",
                                "description": "SONORANT",
                                "values": "ae ee eh ih ey ah UH uh ur ow aa aw l r w y oh ooh awh",
                            },
                            {
                                "type": "question",
                                "description": "VOICED",
                                "values": "ae ee eh ih UH uh ah aa aw ey ow l r w y m n ng j b d dh g v z um un ul ur zh",
                            },
                        ],
                    },
                ],
            },
        ]
