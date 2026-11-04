"""Compare prepared audio through audited selection-only dataset wrappers."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4096)
def canonical_audio_origin(dataset):
    """Subset selection changes coverage, not samples for the retained IDs.

    Only PrepareBenchmarkSubsetJob is followed. Do not collapse arbitrary
    `.dataset` dependencies: encoding, cropping or preprocessing jobs may change
    the audio. Callers must still check raw references and scoring-unit coverage.
    """
    from .controlled_comparison import parameters_flat

    current = Path(dataset).resolve()
    seen = set()
    for _ in range(16):
        if current in seen:
            raise ValueError("Cycle in prepared-audio subset provenance")
        seen.add(current)
        job = current.parent.parent
        if (current.name != "dataset" or current.parent.name != "output"
                or job.name.split(".", 1)[0] != "PrepareBenchmarkSubsetJob"):
            return str(current)
        parent = parameters_flat(job).get("dataset")
        if not isinstance(parent, str) or not Path(parent).is_absolute():
            raise ValueError(f"Cannot verify selection-only audio origin: {current}")
        current = Path(parent).resolve()
    raise ValueError("Prepared-audio subset provenance is too deep")


def audio_origins(record):
    return tuple(sorted({canonical_audio_origin(path) for path in record.get("dataset_paths", [])}))


def check_audio_origins(a, b):
    aa, bb = audio_origins(a), audio_origins(b)
    if aa and bb and aa != bb:
        raise ValueError(f"Pair has different prepared audio origins: {aa!r} versus {bb!r}")
