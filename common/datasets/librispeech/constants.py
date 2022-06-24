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

# concurrent should roughly be set to 5 hours per sub-task for training
# and 0.5 hours per subtask for recognition
concurrent = {
    "train-clean-100": 20,
    "train-clean-360": 80,
    "train-other-500": 100,
    "dev-clean": 10,
    "dev-other": 10,
    "test-clean": 10,
    "test-other": 10,
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

concurrent["train-clean-460"] = (
    concurrent["train-clean-100"] + concurrent["train-clean-360"]
)
concurrent["train-other-960"] = (
    concurrent["train-clean-460"] + concurrent["train-other-500"]
)
