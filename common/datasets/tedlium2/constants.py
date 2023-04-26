DURATIONS = {
    "train": 212.07,
    "dev": 1.60,
    "test": 2.62,
}

NUM_SEGMENTS = {
    "train": 92973,
    "dev": 507,
    "test": 1155,
}

# concurrent should roughly be set to 5 hours per sub-task for training
# and 0.5 hours per subtask for recognition
CONCURRENT = {
    "train": 50,
    "dev": 5,
    "test": 5,
}

BOUNDARY = "#"

SILENCE = "[SILENCE]"

PHONEMES = [
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
