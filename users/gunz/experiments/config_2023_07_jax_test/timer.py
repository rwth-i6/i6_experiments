from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Optional


@dataclass(frozen=True)
class Timestamp:
    name: str
    start: float

    def elapsed(self) -> float:
        return time.process_time() - self.start

    def __str__(self) -> str:
        time = round(self.elapsed(), 1)
        return f"{self.name}={time}s"


class IncAvg:
    n: int = 0
    value: float = 0.0

    def total(self) -> float:
        return self.n * self.value

    def update(self, value: float) -> float:
        self.n += 1
        self.value = self.value + (value - self.value) / self.n
        return self.value


class Timer:
    _avg: IncAvg
    name: Optional[str]

    def __init__(self, name: Optional[str] = ""):
        self._avg = IncAvg()
        self.name = name

    @contextmanager
    def enter(self):
        start = Timestamp(name=self.name, start=time.process_time())
        yield start
        self._avg.update(start.elapsed())

    def total(self) -> float:
        return self._avg.total()

    def value(self) -> float:
        return self._avg.value

    def __str__(self):
        name = self.name or "t"
        time = round(self.value(), 1)
        return f"{name}={time}s"
