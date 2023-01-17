SUBDIR_PREFIX = "Switchboard"

durations = {"train": 311.62, "hub5e00": 3.61, "hub5e01": 6.17, "rt03s": 6.27}

# about 1 per 0.5 hours for test and 1 per 5 hours for train
concurrent = {
    "train": 60,
    "hub5e00": 7,
    "hub5e01": 12,
    "rt03s": 12,
}

num_segments = {
    "train": 249536,
    "hub5e00": 4466,
    "hub5e01": 5895,
    "rt03s": 8420,
}
