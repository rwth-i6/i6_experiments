"""Run names never reach a NEW job's hash (user, 2026-10-13: "we should make sure the run name never makes its way
into any hash").

Ten job classes carry run names / tags in hashed parameters: the self-play labels (``PersonaSelfPlay.label_a/_b``),
the ``{label: path}`` dicts of the multi-run read-outs (``SelfPlayCoherenceJudge``, ``SelfPlayJudgeV2``,
``SelfPlayJudgeV2Table``, ``SelfPlayCopyStats``, ``SelfPlayLexiconRates``, ``PersonaHeldoutTurnPos``), the ``run`` of
``PersonaHeldoutSummary`` entries, and the tags of ``ResultNotify`` / ``VoiceBenchResponses``. With the name in the hash,
renaming an arm re-hashes (= re-runs) all of its read-outs. Each class's ``hash()`` goes through :func:`label_free_hash`:

* its old, label-keyed hash, if that hash is one of ``label_hash_legacy.LEGACY_LABEL_HASHED[cls]`` (every job of the
  class that existed on disk on 2026-10-13: they all keep their work dirs);
* otherwise the hash of the class's label-free view (``canon(parsed_args)``: labels dropped, ``{label: x}`` dicts
  replaced by the sorted hashes of their values) + ``label_free_hash: 1``.

So no existing job moves, and every new job of these classes -- whichever track, family or call site declares it --
is invariant under renaming its runs. Output ROWS still carry the labels the first declaration passed (two
declarations that differ only in labels are now ONE job; :func:`label_collisions` lists them).

Guard: ``tests/check_label_free_hash.py`` (renames real arms in the real graph: no hash may move).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .label_hash_legacy import LEGACY_LABEL_HASHED

#: The key the label-free view adds (so a label-free hash can never equal a label-keyed one).
LABEL_FREE_KEY = "label_free_hash"
LABEL_FREE_VERSION = 1

#: ``{(class name, hash): label signature}`` of the label-free jobs built in this process (collision detection).
_SEEN: dict = {}
_COLLISIONS: list = []


def hash_bytes(value) -> bytes:
    """Sisyphus's own hash representation of ``value`` (a tk.Path hashes as its creator + path, never its alias)."""
    from sisyphus.hash import sis_hash_helper

    return sis_hash_helper(value)


def values_canon(d: dict) -> list:
    """A ``{label: value}`` dict without its labels: the sorted hash bytes of its values (duplicates kept)."""
    return sorted(hash_bytes(v) for v in d.values())


def is_legacy(cls, legacy_hash: str) -> bool:
    """True if ``legacy_hash`` names a job of ``cls`` that existed before label-free hashing (keeps its hash)."""
    return legacy_hash in LEGACY_LABEL_HASHED.get(cls.__name__, ())


def label_free_hash(
    cls,
    parsed_args: dict,
    canon: Callable[[dict], dict],
    base_hash: Callable[[dict], str],
    labels: Callable[[dict], Any],
) -> str:
    """The hash of a label-carrying job (module docstring). ``base_hash``: the class's hash before 2026-10-13 (e.g.
    ``Job.hash``); ``canon``: the label-free view of the parsed arguments; ``labels``: the labels ``canon`` drops
    (only used to report collisions)."""
    legacy = base_hash(parsed_args)
    if is_legacy(cls, legacy):
        return legacy
    h = base_hash({**canon(dict(parsed_args)), LABEL_FREE_KEY: LABEL_FREE_VERSION})
    sig = repr(labels(parsed_args))
    prev = _SEEN.setdefault((cls.__name__, h), sig)
    if prev != sig:
        _COLLISIONS.append((cls.__name__, h, prev, sig))
        logging.warning(
            "label-free %s.%s declared twice with different labels (%s vs %s): one job, the first labels",
            cls.__name__,
            h,
            prev,
            sig,
        )
    return h


def label_collisions() -> list:
    """``[(class, hash, first labels, later labels)]``: declarations that differ only in labels (one job)."""
    return list(_COLLISIONS)


def job_is_label_free(job) -> bool:
    """True for a job of a label-carrying class that hashes label-free (i.e. not one of the 2026-10-13 legacy jobs)."""
    cls = type(job)
    if cls.__name__ not in LEGACY_LABEL_HASHED:
        return False
    return not is_legacy(cls, job._sis_id().rsplit(".", 1)[-1])
