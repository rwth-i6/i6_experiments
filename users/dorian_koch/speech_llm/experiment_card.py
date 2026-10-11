"""Experiment card: a NON-hashed note attached to a training job saying why it was run.

``SpeechFinetune(card=...)`` / ``RLFinetune(card=...)`` take a dict with the keys below; both jobs drop ``card``
from their hash unconditionally (like ``compute``), so writing or rewording a card never moves a run. The
``write_config`` task writes it to ``<job>/output/experiment_card.json`` + ``.md``, so a future reader can open any
run and see what it was for and what to compare it against without digging through docs.

Only ``why`` and ``change`` are required. The card is written once, when ``write_config`` runs: editing the dict
of a job whose ``write_config`` already ran changes nothing on disk (the card records why the run was LAUNCHED).
"""

from __future__ import annotations

import json

#: key -> one-line meaning (also the .md headings, in this order)
CARD_KEYS = {
    "why": "why the run exists (the question it answers)",
    "change": "the ONE thing it changes vs its baseline",
    "baseline": "the arm/tag it is compared against",
    "compare": "what to read, and the numbers to beat",
    "evals": "where its results land (tags / output paths)",
    "asked_by": "who asked for it, with the date (e.g. 'user, 2026-10-11')",
    "notes": "anything else a future reader needs",
}
REQUIRED_KEYS = ("why", "change")


def check_card(card: dict | None) -> dict | None:
    """Validate a card at graph-build time (the manager), so a typo fails before anything is submitted."""
    if card is None:
        return None
    assert isinstance(card, dict), f"card must be a dict, got {type(card).__name__}"
    unknown = set(card) - set(CARD_KEYS)
    assert not unknown, f"card: unknown keys {sorted(unknown)} (allowed: {list(CARD_KEYS)})"
    missing = [k for k in REQUIRED_KEYS if not str(card.get(k) or "").strip()]
    assert not missing, f"card: missing {missing}"
    for k, v in card.items():
        assert isinstance(v, (str, list, tuple)), f"card[{k!r}]: str or list of str, got {type(v).__name__}"
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in card.items()}


def write_card(job, json_path: str, md_path: str) -> None:
    """Write ``job.card`` (plus the job's id) as JSON and Markdown."""
    card = dict(job.card)
    payload = {"job": job._sis_id(), **card}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    lines = [f"# Experiment card: {payload['job']}", ""]
    for k in CARD_KEYS:
        if k not in card:
            continue
        v = card[k]
        lines += [f"**{k}**", ""]
        lines += [f"- {x}" for x in v] if isinstance(v, list) else [v]
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
