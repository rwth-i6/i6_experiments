import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass(eq=True, frozen=True)
class AllophoneState:
    """
    A single, parsed allophone state.
    """

    ctx_l: str
    ctx_r: str
    ph: str
    rest: str

    def in_context(self, left: Optional["AllophoneState"], right: Optional["AllophoneState"]) -> "AllophoneState":
        if self.ph == "[SILENCE]":
            # Silence does not have context.

            return self

        new_left = left.ph if left is not None and left.ph != "[SILENCE]" else "#"
        new_right = right.ph if right is not None and right.ph != "[SILENCE]" else "#"
        return dataclasses.replace(self, ctx_l=new_left, ctx_r=new_right)

    def to_mono(self) -> str:
        return f"{self.ph}{self.rest}"

    def __str__(self):
        return f"{self.ph}{{{self.ctx_l}+{self.ctx_r}}}{self.rest}"

    @classmethod
    def from_alignment_state(cls, state: str) -> "AllophoneState":
        import re

        match = re.match(r"^(.*)\{(.*)\+(.*)}(.*)$", state)
        if match is None:
            raise AttributeError(f"{state} is not an allophone state")

        return cls(ph=match.group(1), ctx_l=match.group(2), ctx_r=match.group(3), rest=match.group(4))
