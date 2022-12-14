durations = {
    "train": 212.07,
    "dev": 1.60,
    "test": 2.62,
}

num_segments = {
    "train": 92973,
    "dev": 507,
    "test": 1155,
}

# concurrent should roughly be set to 5 hours per sub-task for training
# and 0.5 hours per subtask for recognition
concurrent = {
    "train": 50,
    "dev": 5,
    "test": 5,
}
