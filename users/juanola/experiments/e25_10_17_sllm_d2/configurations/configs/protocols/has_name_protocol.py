from typing import Protocol


class HasNameProtocol(Protocol):

    @property
    def name(self) -> str: ...
