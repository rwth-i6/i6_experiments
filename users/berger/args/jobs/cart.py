from abc import ABC


class CartQuestions(ABC):
    def __init__(self, max_leaves: int = 9001, min_obs: int = 1000, add_unknown: bool = True):
        self.max_leaves = max_leaves
        self.min_obs = min_obs
        self.boundary = "#"
        self.silence = "[SILENCE]"
        self.add_unknown = add_unknown
        self.unknown = "[UNKNOWN]"
        self.phonemes = []
        self.steps = []

    @property
    def phonemes_str(self):
        return " ".join(self.phonemes)

    @property
    def phonemes_boundary(self):
        return [self.boundary] + self.phonemes

    @property
    def phonemes_boundary_str(self):
        return " ".join(self.phonemes_boundary)

    @property
    def phonemes_boundary_extra_str(self):
        return " ".join(self.phonemes_boundary_extra)

    @property
    def phonemes_boundary_extra(self):
        return (
            [self.boundary] + [self.silence] + [self.unknown] + self.phonemes
            if self.add_unknown
            else [self.boundary] + [self.silence] + self.phonemes
        )
