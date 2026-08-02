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

There are two defences and this checks both: the prompt rule, and -- for the one template that
needs a question its row cannot supply -- actually GIVING it a second gold fact
(`TEMPLATE_EXTRA_FACTS` / `attach_extra_facts`), which removes the reason to invent rather than
forbidding it.

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

import collections
import json
import os
import re
import sys

import jsonschema
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))  # `sisyphus` package lives here
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import (  # noqa: E402
    DIALOGUE_INSTRUCTION_TEMPLATE_NAMES,
    DIALOGUE_INSTRUCTION_TEMPLATES,
    TEMPLATE_EXTRA_FACTS,
    _DIALOGUE_JSON_SCHEMA,
    _SELF_SOURCED_PARAGRAPHS,
    _build_user_message,
    _extras_for,
    _pick_template,
    attach_extra_facts,
    strip_dialogue_markdown_fence,
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

# ---------------------------------------------------------------------------
# Multi-fact dialogues: a template that needs a further question must be GIVEN one
# ---------------------------------------------------------------------------
# Telling a generator "don't invent a question you can't answer" is a rule it can break silently.
# Handing it a real second gold fact removes the reason to invent. `TEMPLATE_EXTRA_FACTS` is that
# wiring, and these checks drive the real caller path (`attach_extra_facts` -> `_extras_for` ->
# `_build_user_message`) rather than re-deriving it -- a guard that builds its own inputs is blind
# to what the caller actually passes.

unknown = set(TEMPLATE_EXTRA_FACTS) - set(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES)
if unknown:
    failures.append(f"TEMPLATE_EXTRA_FACTS names {sorted(unknown)}, which are not templates")

missing_escape = SELF_SOURCED - set(TEMPLATE_EXTRA_FACTS)
if missing_escape:
    failures.append(
        f"{sorted(missing_escape)} invent their own question but cannot be given a real one "
        "(absent from TEMPLATE_EXTRA_FACTS), so the only defence is prompt wording."
    )

# `_extras_for` must respect BOTH caps: the job-level count and the per-template one.
row = {"extra_facts_json": json.dumps([{"question": "Q2?", "answer": "A2", "aliases": []}] * 3)}
for name in DIALOGUE_INSTRUCTION_TEMPLATE_NAMES:
    want = TEMPLATE_EXTRA_FACTS.get(name, 0)
    got = _extras_for(row, name, extra_facts=3)
    n_got = len(got or [])
    if n_got != want:
        failures.append(f"_extras_for({name!r}, extra_facts=3) gave {n_got} facts, expected {want}")
if _extras_for(row, "followup_topic_change", extra_facts=0) is not None:
    failures.append("_extras_for ignored extra_facts=0 -- the job-level switch does not switch off")
if _extras_for({}, "followup_topic_change", extra_facts=1) is not None:
    failures.append("_extras_for invented extras for a row that carries none")

# The supplied facts must actually reach the prompt, and must override the self-sourcing
# instruction the template still carries for the no-extras case.
spec = {"question": "Q1?", "answer": "A1", "aliases": ["a1"], "options": None, "background": None}
tpl = DIALOGUE_INSTRUCTION_TEMPLATES[DIALOGUE_INSTRUCTION_TEMPLATE_NAMES.index("followup_topic_change")]
msg = _build_user_message(spec, tpl, [{"question": "Q2?", "answer": "A2", "aliases": ["a2"]}])
for needle in ("Q2?", "A2", "Do not think up a further question of your own"):
    if needle not in msg:
        failures.append(f"prompt with extras is missing {needle!r} -- supplied facts do not reach the model")
# ...and the instruction to source one itself must be GONE, not merely overridden later.
for name, para in _SELF_SOURCED_PARAGRAPHS.items():
    if para in msg:
        failures.append(
            f"{name}: the prompt still tells the generator to choose its own further question while "
            "also supplying one -- two contradicting instructions in the same prompt"
        )
    if para not in _build_user_message(spec, tpl):
        failures.append(f"{name}: with no extras supplied, the self-sourcing instruction is missing")
if _build_user_message(spec, tpl) != _build_user_message(spec, tpl, None):
    failures.append("passing extras=None changed the prompt -- single-fact corpora are not reproducible")
if "Additional supplied facts" in _build_user_message(spec, tpl):
    failures.append("the extras block leaked into a prompt with no extras")

# Partner assignment: every row gets real OTHER rows, and every fact is used equally often.
from datasets import Dataset  # noqa: E402

N, N_EXTRA = 40, 2
ds = Dataset.from_dict(
    {
        "question_id": [f"q{i}" for i in range(N)],
        "question": [f"question {i}?" for i in range(N)],
        "answer": [{"value": f"answer {i}", "aliases": []} for i in range(N)],
    }
)
with_extras = attach_extra_facts(ds, "triviaqa", N_EXTRA)
used = collections.Counter()
for i, r in enumerate(with_extras):
    picked = json.loads(r["extra_facts_json"])
    if len(picked) != N_EXTRA:
        failures.append(f"row {i} got {len(picked)} partners, expected {N_EXTRA}")
    if r["question"] in {p["question"] for p in picked}:
        failures.append(f"row {i} was given its own fact as a partner")
    if len({p["question"] for p in picked}) != len(picked):
        failures.append(f"row {i} got the same partner twice")
    used.update(p["question"] for p in picked)
if set(used.values()) != {N_EXTRA}:
    failures.append(
        f"partner exposure is uneven ({sorted(set(used.values()))} appearances per fact, want "
        f"{N_EXTRA} for all) -- some facts would be over-represented in the corpus by luck"
    )
if attach_extra_facts(ds, "triviaqa", N_EXTRA)["extra_facts_json"] != with_extras["extra_facts_json"]:
    failures.append("attach_extra_facts is not deterministic -- a shard could not be regenerated")
print(
    f"[ok] extra facts        {len(TEMPLATE_EXTRA_FACTS)} template(s) can take supplied facts, pairing is a derangement"
)


# --- template selection (migrated from the retired test_pipeline.py) ----------------------------
# Corpus reproducibility rests on _pick_template being a pure function of the row uid: a shard that
# is regenerated must pick the same template, or the mixture silently shifts between runs.

if len(set(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES)) != len(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES):
    dupes = [n for n, c in collections.Counter(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES).items() if c > 1]
    failures.append(f"duplicate template name(s) {dupes} -- per-template slices would collide")

for uid in ["42", "hello", "question_id_999", ""]:
    first = _pick_template(uid)
    if _pick_template(uid) != first:
        failures.append(f"_pick_template({uid!r}) is not deterministic -- shards cannot be regenerated")

n_templates = len(DIALOGUE_INSTRUCTION_TEMPLATES)
reached = {_pick_template(str(i))[1] for i in range(n_templates * 100)}
if len(reached) != n_templates:
    missing = sorted(set(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES) - reached)
    failures.append(
        f"_pick_template never selects {missing} over {n_templates * 100} uids -- those templates "
        "would contribute no rows despite being declared"
    )
print(f"[ok] template selection deterministic, all {n_templates} templates reachable")


# --- generator output contract (migrated from test_pipeline.py) ---------------------------------
# The schema is passed to vLLM as guided_json, so it is what actually constrains generation. An
# empty-text turn survives TTS as a zero-length clip and poisons the alignment.

_valid = [{"speaker": "user", "text": "Hello"}, {"speaker": "assistant", "text": "Hi"}]
try:
    jsonschema.validate(_valid, _DIALOGUE_JSON_SCHEMA)
except jsonschema.ValidationError as exc:
    failures.append(f"_DIALOGUE_JSON_SCHEMA rejects a valid dialogue: {exc.message}")

try:
    jsonschema.validate(
        [{"speaker": "user", "text": ""}, {"speaker": "assistant", "text": "Hi"}],
        _DIALOGUE_JSON_SCHEMA,
    )
    failures.append("_DIALOGUE_JSON_SCHEMA accepts an empty-text turn -- TTS would emit a 0-length clip")
except jsonschema.ValidationError:
    pass
print("[ok] dialogue schema     accepts valid turns, rejects empty text")


# --- markdown-fence stripping (migrated from test_pipeline.py) ----------------------------------
# Drives the REAL HfDialogueCleaner helper. It used to be a closure inside the job's run(), and the
# old test re-implemented it, so the "guard" could not have caught a change to the job.

_payload = [{"speaker": "user", "text": "Hello"}]
for label, raw in [
    ("fenced", "```json\n" + json.dumps(_payload) + "\n```"),
    ("plain", json.dumps(_payload)),
    ("fenced+whitespace", "  ```json\n" + json.dumps(_payload) + "\n```  "),
]:
    try:
        if json.loads(strip_dialogue_markdown_fence(raw)) != _payload:
            failures.append(f"strip_dialogue_markdown_fence mangled the {label} payload")
    except json.JSONDecodeError as exc:
        failures.append(f"strip_dialogue_markdown_fence left unparseable JSON for {label}: {exc}")
print("[ok] fence stripping     fenced / plain / padded payloads all parse")


if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"\nall {len(DIALOGUE_INSTRUCTION_TEMPLATES)} templates forbid inventing time/place-dependent facts")
