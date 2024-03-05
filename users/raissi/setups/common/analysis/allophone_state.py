from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class AllophoneState:
    """
    A single, parsed allophone state.
    """

    ctx_l: str
    ctx_r: str
    ph: str
    rest: str

    def to_mono(self) -> str:
        return f"{self.ph}{self.rest}"

    def __str__(self):
        return f"{self.ph}{{{self.ctx_l}+{self.ctx_r}}}{self.rest}"

    def as_context(self):
        return "#" if self.ph == "[SILENCE]" else self.ph

    @classmethod
    def from_alignment_state(cls, state: str) -> "AllophoneState":
        import re

        match = re.match(r"^(.*)\{(.*)\+(.*)}(.*)$", state)
        if match is None:
            raise AttributeError(f"{state} is not an allophone state")

        return cls(ph=match.group(1), ctx_l=match.group(2), ctx_r=match.group(3), rest=match.group(4))
