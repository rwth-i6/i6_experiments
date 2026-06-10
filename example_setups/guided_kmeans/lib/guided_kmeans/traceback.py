from typing import Protocol

class TracebackItemProtocol(Protocol):
    lemma: str
    start_time: float
    end_time: float
    lm_score: float
    am_score: float
