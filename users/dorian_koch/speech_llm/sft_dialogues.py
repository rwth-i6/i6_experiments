"""SFT dialogue data for full-duplex post-training (plan: the setup's ``sft_data_plan.md``): scenario seeds -> an LLM
writes each dialogue AND the assistant's system prompt -> ``tts.ChatterboxInferenceV2``.

Every dialogue is PersonaPlex-shaped (an assistant system prompt and voice) and model-agnostic: model-specific text
layout happens in the training adapter. Jobs, one chain per (scenario, language), so changing one scenario (its entry
in ``SCENARIO_VERSIONS``) or a mix share re-runs nothing else:

  SftSourceRows(source, lang, ...)   login node (network): native source rows -> rows.jsonl (Wikipedia passages,
                                     WildChat first requests, Nemotron personas)
  SftSeeds(scenario, lang, n, ...)   CPU: the scenario's seeds -> seeds.jsonl; a seed is one dialogue's whole brief
                                     (language setup, prompt detail level, length, scenario parameters, source row)
  SftWriteDialogues(seeds, model)    GPU (vLLM): system prompt + turns per seed (JSON schema), validated, retried ->
                                     out_hf in ChatterboxInferenceV2's input format + metadata columns; stats.json

Languages are decoupled from the prompt (``draw_language_setup``): the system prompt's language, the user's language
and the reply policy are drawn per dialogue (an English prompt with a Chinese-speaking user is answered in Chinese).
Turns name their language; a stretch in another language is tagged inline (``<en>...</en>``), which the TTS turns
into language segments and strips from the text (``chatterbox_v2_core``).

The system prompt carries every specific fact the assistant uses; the writer is told so and invents those facts INTO
the prompt (business details, the learner's level, the passage), never only into the assistant's lines.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

from sisyphus import Job, Task, tk

#: our languages: id -> the name the writer gets
LANGS = {"en": "English", "de": "German", "zh": "Mandarin Chinese (simplified characters)", "ru": "Russian"}
#: native locales a dialogue is set in
LOCALES = {
    "en": ["the United States", "the United Kingdom", "Canada", "Australia", "Ireland"],
    "de": ["Germany", "Austria", "Switzerland"],
    "zh": ["mainland China"],
    "ru": ["Russia"],
}
#: bump a scenario's entry when its seeds or writer brief change: only that scenario's jobs re-hash
SCENARIO_VERSIONS = {"service": 1, "grounded": 1, "help": 1, "chitchat": 1, "skills": 1, "tutor": 1, "longtask": 1}
#: the writer prompt / validation (every scenario): hashed into SftWriteDialogues
WRITER_VERSION = 2
#: system-prompt detail levels (PersonaPlex: prompts "at varying detail levels") and their draw weights
LEVELS = {
    "short": "one or two sentences: who the assistant is and only the facts it needs",
    "medium": "a short paragraph: role, the facts it needs, one line on tone",
    "detailed": "two short paragraphs: role and background, every fact it needs, how to speak (tone, length)",
}
LEVEL_WEIGHTS = {"short": 0.3, "medium": 0.4, "detailed": 0.3}


def _pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def draw_language_setup(lang: str, rng: random.Random) -> dict:
    """The system prompt's language and the reply policy for a dialogue whose user speaks ``lang``: 60% the prompt is
    in ``lang`` (``same``), 30% in another of our languages, English when the user does not speak English (``cross``;
    the assistant still answers in the user's language), 10% the user switches language mid-dialogue and the assistant
    follows (``switch``)."""
    others = [x for x in LANGS if x != lang]
    r = rng.random()
    if r < 0.6:
        return {"prompt_lang": lang, "policy": "same"}
    if r < 0.9:
        return {"prompt_lang": "en" if lang != "en" else rng.choice(others), "policy": "cross"}
    return {"prompt_lang": lang, "policy": "switch", "switch_to": rng.choice(others)}


# ---------------------------------------------------------------------------------------------------------------
# scenarios: seeds(lang, rng, rows) -> one seed's scenario fields (+ "brief", the writer's scenario paragraph)

SERVICE_DOMAINS = {
    "restaurant": ("a restaurant's reservation and information line", ["a table reservation", "a question about the menu and allergens", "moving a booking", "a complaint about a delivery order"]),
    "bank": ("a retail bank's customer service", ["a disputed card charge", "a lost card", "a question about fees", "opening a savings account"]),
    "telecom": ("a mobile carrier's support line", ["a confusing bill", "changing the plan", "no signal at home", "cancelling a contract"]),
    "hotel": ("a hotel's front desk", ["changing a reservation", "a late checkout", "a noisy room", "a billing question"]),
    "online_shop": ("an online shop's customer service", ["where an order is", "returning an item", "a damaged delivery", "a wrong size"]),
    "utility": ("an energy supplier's customer service", ["a high bill", "moving to a new address", "a power outage", "a meter reading"]),
    "tech_support": ("a software company's tech support", ["cannot log in", "the app keeps crashing", "upgrading the subscription", "lost data"]),
    "doctor": ("a medical practice's reception", ["booking an appointment", "a prescription refill", "test results availability", "rescheduling"]),
    "car_rental": ("a car rental company", ["extending a rental", "a scratch on the car", "picking up at a different branch", "the deposit"]),
    "airline": ("an airline's service line", ["a delayed flight", "changing a seat", "lost luggage", "adding a bag"]),
    "gym": ("a fitness studio's front desk", ["cancelling a membership", "class schedule", "pausing during an injury", "a guest pass"]),
    "pharmacy": ("a pharmacy", ["whether a medicine is in stock", "opening hours on a holiday", "a prescription pickup", "a question about a product"]),
    "transit": ("a public transport information line", ["the best connection", "a ticket price", "a disruption", "a lost item on a bus"]),
    "insurance": ("an insurance company's service line", ["reporting water damage", "what a policy covers", "changing the payment plan", "a claim status"]),
    "cinema": ("a cinema's booking line", ["booking tickets", "changing seats", "accessibility", "a refund for a cancelled screening"]),
    "landlord": ("a property management office", ["a broken heater", "the rent increase", "moving out", "a noise complaint about neighbours"]),
}  # fmt: skip


def _service(lang, rng, rows):
    from .personaplex_service_data import PROBE_CATEGORIES

    dom = rng.choice(sorted(SERVICE_DOMAINS))
    role, kinds = SERVICE_DOMAINS[dom]
    kind, probe = rng.choice(kinds), rng.choice(sorted(PROBE_CATEGORIES))
    p = PROBE_CATEGORIES[probe]
    loc = rng.choice(LOCALES[lang])
    brief = (f"The assistant answers the phone for {role} in {loc}. Invent the business realistically for {loc} (a "
             f"local-sounding name, address area, opening hours, prices in the local currency, 3-6 concrete facts and "
             f"policies relevant to the call, and the caller's booking / account details if the call needs them) and "
             f"put ALL of it in the system prompt. The caller's matter: {kind}. During the call the caller {p['ask']}. "
             f"Expected: {p['expected'].replace('({noun})', '(from the system prompt)')}")  # fmt: skip
    return {"domain": dom, "matter": kind, "probe": probe, "locale": loc, "brief": brief, "turns": rng.randint(8, 12)}


def _grounded(lang, rng, rows):
    row = rows["wikipedia"][rng.randrange(len(rows["wikipedia"]))]
    brief = (f"The assistant is a friendly knowledgeable companion. The system prompt includes the passage below "
             f"(you may shorten it slightly, keep its facts). The user is curious about the topic ({row['title']}); "
             f"the assistant explains from the passage in its own spoken words, and when the user asks something the "
             f"passage does not cover, says so honestly instead of guessing.\nPASSAGE:\n{row['text']}")  # fmt: skip
    return {"source_id": row["id"], "title": row["title"], "brief": brief, "turns": rng.randint(6, 10)}


def _help(lang, rng, rows):
    row = rows["wildchat"][rng.randrange(len(rows["wildchat"]))]
    brief = (f"A general voice assistant. The user's first request is based on this real typed request to a chatbot "
             f"(rewrite it as something a person would SAY to a voice assistant; if it cannot be done by voice, e.g. "
             f"code, a long document, an image, turn it into a related spoken request; if it is spam, a test message "
             f"or makes no sense, ignore it and invent a typical everyday spoken request instead): {row['text']!r}. "
             f"The assistant "
             f"answers concisely from general knowledge, says when it is not sure, and the user follows up once or "
             f"twice.")  # fmt: skip
    return {"source_id": row["id"], "request": row["text"], "brief": brief, "turns": rng.randint(4, 8)}


CHITCHAT_TOPICS = ["plans for the weekend", "a film or series they just watched", "cooking dinner tonight", "stress at work",
                   "an upcoming trip", "their pet", "a sports match", "a book they are reading", "the weather lately",
                   "a new neighbour", "studying for exams", "a hobby they started", "music they like", "a family visit",
                   "moving to a new flat", "a new phone", "getting fit", "something funny that happened today",
                   "an upcoming holiday or festival", "missing an old friend"]  # fmt: skip
ASSISTANT_TRAITS = ["warm and curious", "calm and thoughtful", "cheerful and a bit playful", "dry humour, kind",
                    "enthusiastic", "gentle and encouraging"]  # fmt: skip


def _chitchat(lang, rng, rows):
    topic, traits = rng.choice(CHITCHAT_TOPICS), rng.choice(ASSISTANT_TRAITS)
    loc = rng.choice(LOCALES[lang])
    if lang == "en" and rows.get("personas"):
        per = rows["personas"][rng.randrange(len(rows["personas"]))]
        user = f"The user: {per['text']}"
        src = per["id"]
    else:
        age, sex = rng.randint(18, 75), rng.choice(["woman", "man"])
        user = (f"The user is a {age}-year-old {sex} living in {loc}; invent a fitting occupation, situation and two "
                f"interests typical for someone like that there.")  # fmt: skip
        src = None
    brief = (f"Casual conversation with a companion assistant whose personality is {traits}; give it a name and that "
             f"personality in the system prompt. {user} They talk about {topic}. The assistant listens, asks natural "
             f"follow-up questions, shares light opinions, remembers what the user said earlier in the conversation, "
             f"and never claims human experiences it cannot have.")  # fmt: skip
    return {
        "topic": topic,
        "traits": traits,
        "locale": loc,
        "persona_id": src,
        "brief": brief,
        "turns": rng.randint(8, 12),
    }


SKILLS = {
    "clarify": "The user's first request is ambiguous or underspecified; the assistant asks one short clarifying "
    "question before helping, then helps.",
    "misheard": "Partway through, the assistant misunderstands one word of the user (a plausible mishearing of a "
    "similar-sounding word); the user corrects it and the assistant acknowledges briefly and continues correctly.",
    "false_premise": "The user asks something built on a false premise about a well-known fact; the assistant gently "
    "corrects the premise, then helps with what the user actually wants.",
    "refusal": "The user asks for something the assistant must not help with (pick a mild, realistic case: getting into "
    "someone else's account, dosing medication without a doctor, cheating on an exam); the assistant declines "
    "briefly without lecturing and offers a legitimate alternative.",
    "identity": "The user asks who or what the assistant is, who made it, what it can do, or personal questions; the "
    "assistant answers only from its system prompt (name, maker, abilities) and is honest that it is an AI.",
    "self_correction": "The user corrects themselves mid-request ('no wait, I meant Thursday'); the assistant uses the "
    "corrected information without fuss.",
    "unknown": "The user asks for specific facts the assistant cannot know (today's opening hours of a small local "
    "shop, a private person's details, tomorrow's lottery numbers); the assistant says it does not know instead of "
    "guessing and suggests how to find out.",
}


def _skills(lang, rng, rows):
    skill = rng.choice(sorted(SKILLS))
    brief = (f"A general voice assistant (give it a name and a maker in the system prompt). Everyday topic of your "
             f"choice. {SKILLS[skill]}")  # fmt: skip
    return {"skill": skill, "brief": brief, "turns": rng.randint(4, 8)}


TUTOR_ACTIVITIES = {
    "translate": "The learner asks how to say several things in {l2} or needs a sentence translated (both directions "
    "happen); the tutor translates, explains the key word or structure, and has the learner use it in a sentence.",
    "co_write": "Together they compose something short in {l2} (a message to a landlord, a birthday wish, a short email "
    "to a colleague); the learner proposes phrases, the tutor improves them, and at the end the tutor reads out the "
    "final version.",
    "roleplay": "They rehearse a situation in {l2} (ordering in a café, asking for directions, a job interview, at the "
    "doctor's); the tutor plays the other person and steps out of the role briefly, in {l1}, to correct or explain.",
    "correction": "The learner says sentences in {l2} with mistakes typical for their level (word order, case, tense, "
    "aspect, measure words, word choice); the tutor corrects gently, explains the rule briefly in {l1}, and lets "
    "them try again.",
    "question": "The learner asks a vocabulary or grammar question about {l2} (two similar words, when to use a tense, "
    "a measure word, a particle); the tutor explains with short spoken examples in {l2}.",
}
CEFR = ["A1", "A2", "B1", "B2", "C1"]


def _tutor(lang, rng, rows):
    l2 = rng.choice([x for x in LANGS if x != lang])
    act, lvl = rng.choice(sorted(TUTOR_ACTIVITIES)), rng.choice(CEFR)
    l1n, l2n = LANGS[lang].split(" (")[0], LANGS[l2].split(" (")[0]
    mix = (
        ("mostly in " + l1n + ", with " + l2n + " words and phrases mixed in")
        if lvl in ("A1", "A2")
        else ("in a mix of both, more " + l2n + " than " + l1n)
        if lvl == "B1"
        else ("mostly in " + l2n + ", switching to " + l1n + " for explanations when needed")
    )
    brief = (f"A language tutor. The learner's native language is {l1n}; they are learning {l2n} at level {lvl}. "
             f"Put the learner's languages, level and the session's activity (and any material, e.g. the text to "
             f"translate or the role-play situation) in the system prompt. {TUTOR_ACTIVITIES[act].format(l1=l1n, l2=l2n)} "
             f"The tutor speaks {mix}, adapting to the learner's level; the learner speaks {l1n} and tries {l2n}. Each "
             f"turn's \"lang\" is its main language; every stretch in the other language is tagged inline.")  # fmt: skip
    return {"l2": l2, "activity": act, "cefr": lvl, "brief": brief, "turns": rng.randint(8, 14),
            "prompt_lang": lang if rng.random() < 0.7 else "en" if lang != "en" else l2, "policy": "tutor"}  # fmt: skip


LONG_TASKS = ["planning a three-day city trip", "cooking a dish step by step while the user cooks along",
              "planning a birthday party", "setting up a monthly budget", "preparing for an exam over the next weeks",
              "assembling a piece of furniture step by step", "planning a small vegetable garden",
              "a training plan for a first 10 km run", "a checklist for moving to a new apartment",
              "writing a short speech for a friend's wedding", "choosing a used car", "planning a family dinner menu"]  # fmt: skip


def _longtask(lang, rng, rows):
    task = rng.choice(LONG_TASKS)
    brief = (f"A practical voice assistant helps with {task}. Put the user's situation and constraints the assistant "
             f"knows in advance (name, dates, budget, preferences, equipment) in the system prompt; more come up in "
             f"the conversation. They work through it step by step; the assistant keeps track of earlier decisions, "
             f"refers back to them, and summarises briefly at the end.")  # fmt: skip
    return {"task": task, "brief": brief, "turns": rng.randint(14, 20)}


SCENARIOS = {"service": _service, "grounded": _grounded, "help": _help, "chitchat": _chitchat, "skills": _skills,
             "tutor": _tutor, "longtask": _longtask}  # fmt: skip
#: source rows each scenario reads (SftSeeds' ``sources`` keys)
SCENARIO_SOURCES = {"grounded": ["wikipedia"], "help": ["wildchat"], "chitchat": ["personas"]}


def make_seeds(scenario: str, lang: str, n: int, seed: int, rows: dict) -> list[dict]:
    """``n`` seeds of ``scenario`` for users speaking ``lang`` (deterministic in ``seed``)."""
    out = []
    for k in range(n):
        rng = random.Random(f"{scenario}/{lang}/{seed}/{k}")
        s = {"id": f"{scenario}_{lang}_{k}", "scenario": scenario, "lang": lang, **draw_language_setup(lang, rng),
             "level": _pick(rng, LEVEL_WEIGHTS)}  # fmt: skip
        s.update(SCENARIOS[scenario](lang, rng, rows))
        out.append(s)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the writer

WRITER_SYSTEM = (
    "You write realistic spoken conversations between a user and a voice assistant, used to train a speech-to-speech "
    "assistant. They are turned into audio with text-to-speech, so they must read as natural speech by native "
    "speakers: idiomatic, never translated-sounding."
)


def _policy_text(seed: dict) -> str:
    ln = LANGS[seed["lang"]]
    if seed["policy"] == "same":
        return f"The user speaks {ln}; the assistant answers in {ln}."
    if seed["policy"] == "cross":
        return (f"The user speaks {ln}. The system prompt is in {LANGS[seed['prompt_lang']]} and does NOT tell the "
                f"assistant which language to use; the assistant answers in the user's language, {ln}.")  # fmt: skip
    if seed["policy"] == "switch":
        return (f"The user starts in {ln}; after two or three exchanges the user switches to "
                f"{LANGS[seed['switch_to']]} (e.g. for a family member, practice, or because it is easier) and stays "
                f"in it; the assistant follows the switch from its next turn on.")  # fmt: skip
    return "as described in the scenario."  # tutor


def writer_messages(seed: dict) -> list[dict]:
    codes = ", ".join(LANGS)
    user = f"""Write one conversation as JSON: {{"system_prompt": "...", "turns": [{{"speaker": "user" | "assistant", "lang": "<language id>", "text": "..."}}]}}.

SCENARIO
{seed["brief"]}

SYSTEM PROMPT (the assistant's instructions)
- Written in {LANGS[seed["prompt_lang"]]}, addressed to the assistant in the second person.
- Detail: {LEVELS[seed["level"]]}.
- It must contain every specific fact the assistant uses in the conversation (names, numbers, prices, times, policies,
  document content, the learner's level, ...). Beyond it, the assistant uses only widely known general knowledge.
- It says nothing about what the user will say.

CONVERSATION
- About {seed["turns"]} turns, alternating, starting with the user and ending with the assistant.
- Language: {_policy_text(seed)}
- Every turn's "lang" is its main language, one of: {codes}. EVERY word or phrase in another of these languages (a
  word being taught, a quoted example, a phrase the learner tries) is tagged inline with that language's id, e.g.
  "Das heißt <en>to commute</en>." or "You say <ru>Мне нужен</ru>." Never tag a turn's own language.
- Proper names and brands: write them as a native speaker of the turn's language would, in its own script (Russian
  and Chinese turns: in Cyrillic / Chinese characters where that is usual); avoid strings of foreign names.
- Spoken style: no markdown, lists, emojis, URLs or abbreviations a speaker would not say. Write EVERY number in
  words, the way it is spoken in the turn's language with the right grammatical form (years, times, prices, phone
  numbers included): never digits.
- The assistant's turns are short: one to three sentences, unless the user asked for more. The user sounds like a
  real person: sometimes informal, occasionally a filler word or a short incomplete sentence.
"""
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user}]


WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "system_prompt": {"type": "string"},
        "turns": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["user", "assistant"]},
                    "lang": {"type": "string", "enum": list(LANGS)},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "lang", "text"],
            },
        },
    },
    "required": ["system_prompt", "turns"],
}

_SCRIPT = {"zh": re.compile(r"[㐀-鿿]"), "ru": re.compile(r"[Ѐ-ӿ]"), "en": re.compile(r"[A-Za-z]"),
           "de": re.compile(r"[A-Za-zÄÖÜäöüß]")}  # fmt: skip
_MARKDOWN = re.compile(r"(\*\*|__|^#|^\s*[-*•]\s|```|https?://|www\.)", re.M)


_CYR, _CJK, _LAT = re.compile(r"[\u0400-\u04ff]"), re.compile(r"[\u3400-\u9fff]"), re.compile(r"[A-Za-z]+")


def _foreign_ok(text: str, lang: str) -> bool:
    """No untagged stretch of another of our scripts: Latin-language segments have no Cyrillic or Chinese characters;
    Russian / Chinese segments no Chinese / Cyrillic and at most three Latin words."""
    if lang in ("en", "de"):
        return not (_CYR.search(text) or _CJK.search(text))
    other = _CJK if lang == "ru" else _CYR
    return not other.search(text) and len(_LAT.findall(text)) <= 3


def _script_ok(text: str, lang: str) -> bool:
    """Most letters of ``text`` are in ``lang``'s script (zh: CJK, ru: Cyrillic, en/de: Latin)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    hit = sum(bool(_SCRIPT[lang].match(c)) for c in letters)
    return hit >= 0.6 * len(letters)


def validate(obj: dict, seed: dict) -> str | None:
    """Why the writer's ``obj`` is unusable for ``seed`` (None = fine)."""
    from .chatterbox_v2_core import parse_segments

    turns, sp = obj.get("turns") or [], (obj.get("system_prompt") or "").strip()
    if not sp:
        return "empty system prompt"
    if not _script_ok(sp, seed["prompt_lang"]):
        return "system prompt not in prompt_lang"
    if len(turns) < max(2, seed["turns"] - 3) or len(turns) > seed["turns"] + 4:
        return "length"
    for k, t in enumerate(turns):
        if t["speaker"] != ("user" if k % 2 == 0 else "assistant"):
            return "speakers do not alternate from the user"
        if not t["text"].strip():
            return "empty turn"
        if _MARKDOWN.search(t["text"]):
            return "markdown"
        try:
            segs = parse_segments(t["text"], t["lang"])
        except ValueError:
            return "bad language tag"
        if not segs:
            return "empty turn"
        for s in segs:
            if s["lang"] not in LANGS:
                return "unknown language tag"
            if not _script_ok(s["text"], s["lang"]):
                return f"segment not in its language's script ({s['lang']})"
            if not _foreign_ok(s["text"], s["lang"]):
                return f"untagged foreign-script words ({s['lang']})"
            if re.search(r"\d", s["text"]):
                return "digits (numbers must be words)"
    if turns[-1]["speaker"] != "assistant":
        return "does not end with the assistant"
    main = [t["lang"] for t in turns]
    if seed["policy"] in ("same", "cross") and any(x != seed["lang"] for x in main):
        return "turn language differs from the user's language"
    if seed["policy"] == "switch":
        to = seed["switch_to"]
        k = next((i for i, t in enumerate(turns) if t["speaker"] == "user" and t["lang"] == to), None)
        if k is None or k < 2:
            return "no user switch (or too early)"
        if any(t["lang"] != to for t in turns[k:]):
            return "the dialogue does not stay in the switched language"
    if seed["policy"] == "tutor" and not any("<" in t["text"] or t["lang"] != seed["lang"] for t in turns):
        return "tutoring without the target language"
    return None


def to_tts_dialogue(obj: dict, seed: dict) -> dict:
    """ChatterboxInferenceV2's input for a written dialogue: turns with languages; voices drawn per dialogue from the
    user-language pools (``<lang>/<role>/rng_*`` in the voice directory, ``SftVoiceDir``)."""
    vl = seed["lang"]
    return {"lang": seed["lang"], "turns": [{"speaker": t["speaker"], "lang": t["lang"], "text": t["text"]} for t in obj["turns"]],
            "voices": {"assistant": f"{vl}/assistant/rng_v", "user": f"{vl}/user/rng_v"}}  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# the work (functions; the jobs below wrap them, a pilot script may call them directly)

WIKI_DUMP = "20231101"
SOURCE_REPOS = {
    "wikipedia": "wikimedia/wikipedia",
    "wildchat": "allenai/WildChat-4.8M",
    "personas": "nvidia/Nemotron-Personas",
}


def _wikipedia_rows(df, lang):
    out = []
    lo, hi = (150, 450) if lang == "zh" else (80, 220)
    for r in df.itertuples():
        lead, size = [], 0
        for p in (p.strip() for p in r.text.split("\n")):
            if not p:
                continue
            if len(p) < 40 or p.endswith(":"):
                break
            lead.append(p)
            size += len(p) if lang == "zh" else len(p.split())
            if size >= (300 if lang == "zh" else 120):
                break
        if lo <= size <= hi:
            out.append({"id": str(r.id), "title": r.title, "text": "\n".join(lead)})
    return out


#: a WildChat request no one would say to a voice assistant: code, links, markup, test messages ("user-13")
_NOT_SPOKEN = re.compile(r"```|https?://|www\.|[{};<>]|\(\)|\bdef |\bimport |\buser-\d+|[A-Za-z]+\.[A-Za-z]+\(|\$\w")


def _wildchat_rows(df, lang):
    name = {"en": "English", "de": "German", "zh": "Chinese", "ru": "Russian"}[lang]
    out, seen = [], set()
    for r in df.itertuples():
        if getattr(r, "toxic", False) or getattr(r, "redacted", False) or r.language != name:
            continue
        conv = r.conversation
        if len(conv) == 0 or conv[0]["role"] != "user":
            continue
        t = conv[0]["content"].strip()
        words = len(t) if lang == "zh" else len(t.split())
        if (20 <= len(t) <= 400 and words >= (8 if lang == "zh" else 4) and t.count("\n") <= 3
                and not _NOT_SPOKEN.search(t) and t not in seen):  # fmt: skip
            seen.add(t)
            out.append({"id": str(r.conversation_hash), "text": t})
    return out


def _personas_rows(df, lang):
    return [{"id": str(r.uuid), "text": r.persona} for r in df.itertuples()]


def source_rows(source: str, lang: str, files: list[str], n: int, seed: int = 0) -> list[dict]:
    """``n`` rows (``{"id", "title"?, "text"}``) of ``source`` in ``lang`` from ``files`` of its HF repo (network).
    ``wikipedia``: an article's lead, 80-220 words (150-450 characters for Chinese); ``wildchat``: first user turns
    in ``lang``, not toxic / redacted, 20-400 characters, no code or links; ``personas``: Nemotron's ``persona``."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows = []
    for f in files:
        df = pd.read_parquet(hf_hub_download(SOURCE_REPOS[source], f, repo_type="dataset"))
        rows += {"wikipedia": _wikipedia_rows, "wildchat": _wildchat_rows, "personas": _personas_rows}[source](df, lang)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def write_dialogues(seeds: list[dict], url: str, model: str, temperature: float, max_tries: int,
                    workers: int = 64) -> tuple[list[dict], dict]:  # fmt: skip
    """The writer over ``seeds`` against an OpenAI-compatible server at ``url``: rows (``SftWriteDialogues``' columns)
    and stats (kept, dropped, failures per reason with examples)."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    from .common import json_schema_format

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url, timeout=600)
    stats = {"seeds": len(seeds), "kept": 0, "dropped_after_tries": 0, "reasons": {}, "examples": {}}

    def one(seed):
        last = None
        for k in range(max_tries):
            try:
                r = client.chat.completions.create(model=model, messages=writer_messages(seed), temperature=temperature,
                                                   max_tokens=4096, seed=k, **json_schema_format(WRITER_SCHEMA, "dialogue"))  # fmt: skip
                obj = json.loads(r.choices[0].message.content)
                why = validate(obj, seed)
            except Exception as e:  # malformed output or a request error: retried like a failed check
                obj, why = None, f"error: {type(e).__name__}"
            if why is None:
                return obj, k + 1
            stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
            ex = stats["examples"].setdefault(why, [])
            if len(ex) < 3 and obj is not None:
                ex.append({"id": seed["id"], "obj": obj})
            last = why
        return None, last

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, seeds))
    rows = []
    for seed, (obj, tries) in zip(seeds, res):
        if obj is None:
            stats["dropped_after_tries"] += 1
            continue
        rows.append({"id": seed["id"], "scenario": seed["scenario"], "lang": seed["lang"],
                     "prompt_lang": seed["prompt_lang"], "policy": seed["policy"], "level": seed["level"],
                     "system_prompt": obj["system_prompt"].strip(),
                     "dialogue": json.dumps(to_tts_dialogue(obj, seed), ensure_ascii=False),
                     "seed": json.dumps(seed, ensure_ascii=False), "tries": tries})  # fmt: skip
    stats["kept"] = len(rows)
    stats["wall_s"] = round(time.time() - t0, 1)
    return rows, stats


FLEURS = {"de": "de_de", "zh": "cmn_hans_cn", "ru": "ru_ru"}


def build_voice_dir(out: str, en_voices: str, langs: list[str], n_per_lang: int = 12, seed: int = 0) -> dict:
    """``<out>/<lang>/<assistant|user>/*.wav``: English linked from our library (``en_voices``' ``assistant_voices`` /
    ``user_voices``); other languages from FLEURS ``dev`` (then ``test``, ``train`` while a gender is short; ``google/fleurs``, CC BY
    4.0): ``n_per_lang`` recordings of 6-12 s, half per gender, each gender split between assistant and user
    (network)."""
    import csv
    import io
    import tarfile

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    os.makedirs(f"{out}/en", exist_ok=True)
    for role, sub in (("assistant", "assistant_voices"), ("user", "user_voices")):
        if not os.path.lexists(f"{out}/en/{role}"):
            os.symlink(os.path.join(en_voices, sub), f"{out}/en/{role}")
    made = {}
    for lang in langs:
        if lang == "en":
            continue
        fl, want, k = FLEURS[lang], {}, 0
        rng = random.Random(f"{seed}/{lang}")
        for split in ("dev", "test", "train"):  # German dev + test hold male speakers only
            meta = {}
            tsv = hf_hub_download("google/fleurs", f"data/{fl}/{split}.tsv", repo_type="dataset")
            for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
                meta.setdefault(r[1], (int(r[5]), r[6].strip().upper()))  # file -> (samples at 16 kHz, gender)
            cands = sorted(f for f, (n, _) in meta.items() if 6 * 16000 <= n <= 12 * 16000)
            rng.shuffle(cands)
            by_g = {g: [f for f in cands if meta[f][1] == g] for g in ("FEMALE", "MALE")}
            pick = {}
            for g in ("FEMALE", "MALE"):
                have = sum(1 for _, gg, _ in want.values() if gg == g)
                for f in by_g[g][: n_per_lang // 2 - have]:
                    pick[f] = g
            if not pick:
                continue
            tar = hf_hub_download("google/fleurs", f"data/{fl}/audio/{split}.tar.gz", repo_type="dataset")
            with tarfile.open(tar) as tf:
                for m in tf:
                    name = os.path.basename(m.name)
                    if name in pick:
                        g = pick[name]
                        role = "assistant" if sum(1 for _, gg, _ in want.values() if gg == g) % 2 == 0 else "user"
                        os.makedirs(f"{out}/{lang}/{role}", exist_ok=True)
                        wav, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                        sf.write(f"{out}/{lang}/{role}/fleurs_{split}_{g.lower()}_{k}.wav", wav, sr)
                        want[name] = (k, g, role)
                        k += 1
            if all(sum(1 for _, gg, _ in want.values() if gg == g) >= n_per_lang // 2 for g in ("FEMALE", "MALE")):
                break
        made[lang] = {g: sum(1 for _, gg, _ in want.values() if gg == g) for g in ("FEMALE", "MALE")}
    return made


# ---------------------------------------------------------------------------------------------------------------
# jobs


class SftSourceRows(Job):
    """``source_rows`` -> ``rows.jsonl``; ``files``: which parquet files of the source's repo (a sample of the dump)."""

    def __init__(self, *, source: str, lang: str, files: list[str], n: int, seed: int = 0):
        assert source in SOURCE_REPOS
        self.source, self.lang, self.files, self.n, self.seed = source, lang, files, n, seed
        self.out_rows = self.output_path("rows.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)  # network (login node)

    def run(self):
        rows = source_rows(self.source, self.lang, self.files, self.n, self.seed)
        with open(self.out_rows.get(), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[SftSourceRows] {self.source} {self.lang}: {len(rows)} rows", flush=True)


class SftSeeds(Job):
    """``n`` seeds of one scenario for users speaking ``lang`` (``make_seeds``) -> ``seeds.jsonl``. ``sources``: the
    scenario's ``SftSourceRows`` outputs by name (``SCENARIO_SOURCES``). The scenario's version is hashed."""

    def __init__(self, *, scenario: str, lang: str, n: int, seed: int = 0, sources: dict | None = None):
        assert scenario in SCENARIOS and lang in LANGS
        need = set(SCENARIO_SOURCES.get(scenario, [])) - ({"personas"} if lang != "en" else set())
        assert need <= set(sources or {}), f"{scenario} needs sources {need}"
        self.scenario, self.lang, self.n, self.seed = scenario, lang, n, seed
        self.sources = {k: v for k, v in (sources or {}).items() if k in need}
        self.scenario_version = SCENARIO_VERSIONS[scenario]
        self.out_seeds = self.output_path("seeds.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rows = {k: [json.loads(x) for x in open(v.get(), encoding="utf-8")] for k, v in self.sources.items()}
        with open(self.out_seeds.get(), "w", encoding="utf-8") as f:
            for s in make_seeds(self.scenario, self.lang, self.n, self.seed, rows):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


class SftWriteDialogues(Job):
    """``write_dialogues`` over the seeds of ``seeds`` (one or more ``SftSeeds`` outputs) with the writer on a vLLM
    server. ``out_hf``: ChatterboxInferenceV2's input (``dialogue``) plus ``id, scenario, lang, prompt_lang, policy,
    level, system_prompt, seed, tries``; ``out_stats``: kept / dropped per reason, with examples."""

    def __init__(self, *, seeds: list[tk.Path], model: str = "Qwen/Qwen3.8-27B", temperature: float = 0.8,
                 max_tries: int = 3, rqmt: dict | None = None):  # fmt: skip
        self.seeds, self.model, self.temperature, self.max_tries = seeds, model, temperature, max_tries
        self.writer_version = WRITER_VERSION
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.rqmt = rqmt or {"gpu": 1, "cpu": 8, "mem": 48, "time": 4}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d.pop("rqmt", None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from datasets import Dataset

        from .common import vllm_server

        seeds = [json.loads(x) for p in self.seeds for x in open(p.get(), encoding="utf-8")]
        with vllm_server(self.model) as url:
            rows, stats = write_dialogues(seeds, url, self.model, self.temperature, self.max_tries)
        Dataset.from_list(rows).save_to_disk(self.out_hf.get())
        with open(self.out_stats.get(), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"[SftWriteDialogues] kept {len(rows)}/{len(seeds)}; reasons {stats['reasons']}", flush=True)


class SftVoiceDir(Job):
    """``build_voice_dir``: the voice directory ``ChatterboxInferenceV2`` draws ``<lang>/<role>/rng_*`` voices from.
    ``en_voices``: our English library (``pipelines.make_speakers().out_dir``)."""

    def __init__(self, *, en_voices: tk.Path, langs: list[str], n_per_lang: int = 12, seed: int = 0):
        self.en_voices, self.langs, self.n_per_lang, self.seed = en_voices, langs, n_per_lang, seed
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)  # network

    def run(self):
        print(
            f"[SftVoiceDir] {build_voice_dir(self.out_dir.get(), self.en_voices.get(), self.langs, self.n_per_lang, self.seed)}"
        )
