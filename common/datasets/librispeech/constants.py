durations = {
    "train-clean-100": 100.6,
    "train-clean-360": 363.6,
    "train-other-500": 496.7,
    "dev-clean": 5.4,
    "dev-other": 5.3,
    "test-clean": 5.4,
    "test-other": 5.1,
}

num_segments = {
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "train-other-500": 148688,
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
}

durations["train-clean-460"] = (
    durations["train-clean-100"] + durations["train-clean-360"]
)
durations["train-other-960"] = (
    durations["train-clean-460"] + durations["train-other-500"]
)

num_segments["train-clean-460"] = (
    num_segments["train-clean-100"] + num_segments["train-clean-360"]
)
num_segments["train-other-960"] = (
    num_segments["train-clean-460"] + num_segments["train-other-500"]
)
