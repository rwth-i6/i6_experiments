"""Guard: every dialogue template forbids inventing time- or place-dependent "facts".

A question whose answer depends on when or where it is asked cannot be answered from a fixed corpus,
so a generator can only make one up -- and a confidently invented answer is exactly what a knowledge
model must not be trained on. It also teaches the model to claim capabilities it does not have
("let me check", "according to the latest").

This was not hypothetical. Measured over 8,000 randomly sampled rows of the 2026-07-24
triviaqa_mix corpus, `followup_topic_change` -- which asks for a SECOND question the source row has
no gold answer for -- produced a live/unanswerable question in **33.9%** of its rows and had the
assistant claim live data access in **22.4%**. Every other template was at or below 4%. Typical
output: "do you know if it's going to rain tomorrow?" -> "The forecast says it should stay dry".

The rule lives in `_COMMON_SUFFIX`, so it reaches every template automatically -- but only for as
long as nobody writes a template that bypasses the suffix. That is what this checks. It also pins the
extra grounding on the one template that invents its own question.

Note the template text is NOT part of `HfToDialogue`'s hash (see `templates_version`), so editing a
prompt silently does nothing to an existing corpus. This check therefore guards the *next* corpus,
not the current one; repairing an already-generated corpus needs a `templates_version` bump.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_dialogue_templates.py
"""

import os
import re
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))  # `sisyphus` package lives here
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import (  # noqa: E402
    DIALOGUE_INSTRUCTION_TEMPLATE_NAMES,
    DIALOGUE_INSTRUCTION_TEMPLATES,
)

#: Phrases that must appear in every template's prompt for the ban to be stated at all.
REQUIRED = [
    ("no-live-facts", re.compile(r"NEVER invent information that depends on the current", re.I)),
    ("no-weather", re.compile(r"no weather or forecasts", re.I)),
    ("no-tools", re.compile(r"has NO live data, no internet, and no tools", re.I)),
    ("no-lookup-claim", re.compile(r"never say 'let me check'", re.I)),
]

#: Templates that ask the generator to invent a question of its own, rather than using the supplied
#: one. These need the extra instruction that the invented question must have a settled answer --
#: the general ban says what not to do, this says what to do instead.
SELF_SOURCED = {"followup_topic_change"}
GROUNDED = re.compile(r"durable, well-established general knowledge", re.I)

assert len(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES) == len(DIALOGUE_INSTRUCTION_TEMPLATES)

failures = []
for name, text in zip(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES, DIALOGUE_INSTRUCTION_TEMPLATES):
    missing = [label for label, pat in REQUIRED if not pat.search(text)]
    if missing:
        failures.append(
            f"{name}: missing {missing} -- it does not carry the anti-fabrication rules. If it was "
            "written without _COMMON_SUFFIX, append the suffix rather than restating the rules."
        )
    if name in SELF_SOURCED and not GROUNDED.search(text):
        failures.append(
            f"{name}: invents its own second question but never says the answer must be one it is "
            "confident in. Without that it reaches for weather/time small talk and fabricates."
        )
    print(f"[ok] {name:<24} {len(text):>5} chars, all {len(REQUIRED)} rules present")

# A template that invents a question but is not declared here would be unguarded.
for name, text in zip(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES, DIALOGUE_INSTRUCTION_TEMPLATES):
    invents = re.search(r"SECOND question|another question|question of your own", text, re.I)
    if invents and name not in SELF_SOURCED:
        failures.append(
            f"{name}: asks for a question the source row has no gold answer for, but is not in "
            "SELF_SOURCED, so the grounding requirement is not checked for it."
        )

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"\nall {len(DIALOGUE_INSTRUCTION_TEMPLATES)} templates forbid inventing time/place-dependent facts")
