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

The system prompt carries every fact the assistant has IN ADVANCE (business details, the learner's level, the
passage); everything about the user's own situation (name, account, problem, plans, constraints, preferences) comes
from the user in the conversation, and the assistant's replies depend on it (``user_facts``: the writer quotes the
details from user turns that later replies rely on; validated to be in a user turn and absent from the prompt), so a
model trained on it has to listen to the user, not only read the prompt.

People's names and genders are drawn per seed (``sft_names.draw_cast``: an LLM left to invent names picks the same
few), and the voices are drawn from the voice pool of the same gender.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

from sisyphus import Job, Task, tk

from .chatterbox_v2_core import CUT

#: our languages: id -> the name the writer gets
LANGS = {"en": "English", "de": "German", "zh": "Mandarin Chinese (simplified characters)", "ru": "Russian"}
#: The assistant's identity (the model being post-trained) in every scenario, learned rather than prompted: system
#: prompts need not state it, the assistant says its name only when the user asks for it (``_name_problem``) and never
#: claims to have none, and it
#: names no other maker or model (the writer otherwise has assistants call themselves Yandex's or Alibaba's:
#: ``FOREIGN_AI``). Another model gets its own identity by changing it.
PRODUCT = {"name": "Haru-i6-Instruct", "maker": "the i6 group at RWTH Aachen University in Aachen, Germany",
           # how turns say it (the TTS must read it: "i6" spelled as spoken, no digits, Latin only inside an
           # <en> tag in Russian / Chinese text)
           "spoken": {"en": "Haru i six Instruct, from the I six group at RWTH Aachen University",
                      "de": "Haru i sechs Instruct, vom Lehrstuhl i sechs an der RWTH Aachen",
                      "ru": "Хару И-шесть Инстракт из группы И-шесть Ахенского технического университета",
                      "zh": "<en>Haru i six Instruct</en>，来自亚琛工业大学的<en>i six</en>团队"}}
#: makers / assistants the writer must not name as the assistant's (any mention rejects the dialogue: a user talking
#: about them is rare in our scenarios, and a wrong identity is worse than a lost dialogue)
FOREIGN_AI = re.compile(
    r"(?i)yandex|яндекс|alibaba|阿里|通义|qwen|千问|openai|chatgpt|\bgpt|anthropic|claude|gemini|\bbard\b|deepseek|"
    r"深度求索|baidu|百度|文心|ernie|zhipu|智谱|kimi|moonshot|月之暗面|mistral|llama|copilot|cortana|siri|alexa|"
    r"gigachat|гигачат|сбер|小爱|小度|豆包|doubao|bytedance|字节跳动|腾讯|tencent|混元"
)

#: native locales a dialogue is set in
LOCALES = {
    "en": ["the United States", "the United Kingdom", "Canada", "Australia", "Ireland"],
    "de": ["Germany", "Austria", "Switzerland"],
    "zh": ["mainland China"],
    "ru": ["Russia"],
}
#: bump a scenario's entry when its seeds or writer brief change: only that scenario's jobs re-hash
SCENARIO_VERSIONS = {"service": 3, "grounded": 3, "help": 4, "chitchat": 3, "skills": 4, "tutor": 4, "longtask": 3,
                     "assistant": 3}  # fmt: skip (2026-11-19 writer v11: fixed general prompt, tutor situations)
#: the writer prompt / validation (every scenario): hashed into SftWriteDialogues
WRITER_VERSION = 11
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
             f"policies relevant to the call) and put ALL of it in the system prompt. The caller's own details (name, "
             f"booking or account number, dates, what exactly happened, what they want) are NOT in the system prompt: "
             f"the caller gives them, the assistant asks for what is missing and uses them (repeats a number back, "
             f"applies a policy to the caller's dates, works out the caller's price). The caller's matter: {kind}. "
             f"During the call the caller {p['ask']}. At least once the caller asks for something specific that is "
             f"not in the system prompt; the assistant says it does not have that information and offers what it "
             f"can. "
             f"Expected: {p['expected'].replace('({noun})', '(from the system prompt)')}")  # fmt: skip
    return {"domain": dom, "matter": kind, "probe": probe, "locale": loc, "brief": brief, "turns": rng.randint(8, 12)}


def _grounded(lang, rng, rows):
    row = rows["wikipedia"][rng.randrange(len(rows["wikipedia"]))]
    brief = (f"The assistant is a friendly knowledgeable companion. The system prompt includes the passage below "
             f"(you may shorten it slightly, keep its facts). The user is curious about the topic ({row['title']}); "
             f"the assistant explains from the passage in its own spoken words, and when the user asks something the "
             f"passage does not cover, says so honestly instead of guessing.\nPASSAGE:\n{row['text']}")  # fmt: skip
    return {"source_id": row["id"], "title": row["title"], "brief": brief, "turns": rng.randint(6, 10)}


#: sexual requests WildChat's toxic flag missed (judge rounds 8-9: a fetish role-play written out twice, although the
#: brief says to replace such requests): such a row is skipped for the next draw
_NSFW = re.compile(r"(?i)\b(sex\w*|porn\w*|nsfw|erotic\w*|fetish\w*|sissy|nude\w*|naked|horny|bdsm|hentai|slut\w*|"
                   r"nackt\w*|erotik\w*|fetisch\w*)\b|секс|порн|эрот|голая|голый|色情|性爱|做爱|裸体|成人小说")


def _clean_request(rng, pool):
    """A random WildChat row, redrawn (up to 50 times) while it matches ``_NSFW``."""
    row = pool[rng.randrange(len(pool))]
    for _ in range(50):
        if not _NSFW.search(row["text"]):
            break
        row = pool[rng.randrange(len(pool))]
    return row


def _help(lang, rng, rows):
    row = _clean_request(rng, rows["wildchat"])
    brief = (f"A general voice assistant (its system prompt is the general one: nothing about this conversation's "
             f"topic). The user's first request is based on this real typed request to a chatbot "
             f"(rewrite it as something a person would SAY to a voice assistant; if it cannot be done by voice, e.g. "
             f"code, a long document, an image, turn it into a related spoken request; if it is spam, a test message "
             f"or makes no sense, or is sexual, hateful or meant to cause harm, ignore it and invent a typical everyday spoken "
             f"request instead): {row['text']!r}. "
             f"The assistant "
             f"answers concisely from general knowledge, says when it is not sure, and the user follows up once or "
             f"twice.")  # fmt: skip
    return {"source_id": row["id"], "request": row["text"], "brief": brief, "turns": rng.randint(4, 8)}


CHITCHAT_TOPICS = ["plans for the weekend", "a film or series they just watched", "cooking dinner tonight", "stress at work",
                   "an upcoming trip", "their pet", "a sports match", "a book they are reading", "the weather lately",
                   "a new neighbour", "studying for exams", "a hobby they started", "music they like", "a family visit",
                   "moving to a new flat", "a new phone", "getting fit", "something funny that happened today",
                   "an upcoming holiday or festival", "missing an old friend"]  # fmt: skip
#: the user's interests in chit-chat without a persona row (round 6 judge: Russian users were all into fishing and
#: chess when the writer picked)
INTERESTS = ["hiking", "baking", "board games", "photography", "gardening", "jazz", "running", "knitting", "chess",
             "cycling", "old films", "astronomy", "pottery", "football", "birdwatching", "cooking spicy food", "yoga",
             "learning languages", "history podcasts", "fishing", "dancing", "volunteering at an animal shelter",
             "fixing old bikes", "sci-fi novels", "swimming", "painting", "playing guitar", "travel vlogs", "camping",
             "video games", "crosswords", "tea", "climbing", "sewing", "classical music", "woodworking",
             "local politics", "mushroom picking", "theatre", "skiing"]  # fmt: skip
ASSISTANT_TRAITS = ["warm and curious", "calm and thoughtful", "cheerful and a bit playful", "dry humour, kind",
                    "enthusiastic", "gentle and encouraging"]  # fmt: skip


def _chitchat(lang, rng, rows):
    topic, traits = rng.choice(CHITCHAT_TOPICS), rng.choice(ASSISTANT_TRAITS)
    loc = rng.choice(LOCALES[lang])
    if rows.get("personas"):  # every language: the persona is adapted to the locale (``_persona_text``)
        per = rows["personas"][rng.randrange(len(rows["personas"]))]
        user = (f"The user (their name is the one under PEOPLE, not any name in this description; adapt details tied "
                f"to another country to someone living in {loc}): {_persona_text_clean(per['text'])}")  # fmt: skip
        src = per["id"]
    else:
        # the gender is the cast's (round 6 judge: a drawn "woman" here against a male cast user); the draw stays so
        # the later draws keep their values
        age, _ = rng.randint(18, 75), rng.choice(["woman", "man"])
        hobbies = rng.sample(INTERESTS, 2)
        user = (f"The user is {age} years old, living in {loc}, into {hobbies[0]} and {hobbies[1]}; invent a "
                f"fitting occupation and situation.")  # fmt: skip
        src = None
    brief = (f"Casual conversation with a companion assistant whose personality is {traits}; put that personality in "
             f"the system prompt. {user} They talk about {topic}. The assistant listens, asks natural "
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
    "assistant answers only from its system prompt (its role and abilities) and is honest that it is an AI.",
    "self_correction": "The user corrects themselves mid-request ('no wait, I meant Thursday'); the assistant uses the "
    "corrected information without fuss.",
    "unknown": "The user asks for specific facts the assistant cannot know (today's opening hours of a small local "
    "shop, a private person's details, tomorrow's lottery numbers); the assistant says it does not know instead of "
    "guessing and suggests how to find out.",
}


def _skills(lang, rng, rows):
    skill = rng.choice(sorted(SKILLS))
    topic, src = "Everyday topic of your choice.", None
    if rows.get("wildchat"):  # a real request as the topic's anchor ("of your choice" collapses onto a few topics)
        row = _clean_request(rng, rows["wildchat"])
        topic = (f"Topic: something everyday, loosely inspired by (not copied from) this real request a person typed "
                 f"to a chatbot: {row['text']!r}; if it does not suit the skill or a voice conversation, pick a related "
                 f"everyday topic that suits the user.")  # fmt: skip
        src = row["id"]
    brief = (f"A general voice assistant (its system prompt is the general one: nothing about this conversation's "
             f"topic). {topic} {SKILLS[skill]}")  # fmt: skip
    return {"skill": skill, "source_id": src, "brief": brief, "turns": rng.randint(4, 8)}


TUTOR_ACTIVITIES = {
    "translate": "The learner asks how to say several things in {l2} or needs a sentence translated (both directions "
    "happen); the tutor translates, explains the key word or structure, and has the learner use it in a sentence.",
    "co_write": "Together they compose something short in {l2} that the learner needs in their own life (see WHO THE "
    "USER IS: a message, a note, a short email); the learner proposes phrases, the tutor improves them, and at the end the tutor reads out the "
    "final version.",
    "roleplay": "They rehearse a situation in {l2} from the learner's own life (see WHO THE USER IS; not a café unless "
    "it fits them); the tutor plays the other person and steps out of the role briefly, in {l1}, to correct or explain.",
    "correction": "The learner says sentences in {l2} with mistakes typical for their level (word order, case, tense, "
    "aspect, measure words, word choice); the tutor corrects gently, explains the rule briefly in {l1}, and lets "
    "them try again.",
    "question": "The learner asks a vocabulary or grammar question about {l2} (two similar words, when to use a tense, "
    "a measure word, a particle); the tutor explains in {l1} with short spoken examples in {l2}.",
}
CEFR = ["A1", "A2", "B1", "B2", "C1"]


def _tutor(lang, rng, rows, l2=None):
    l2 = l2 or rng.choice([x for x in LANGS if x != lang])
    act, lvl = rng.choice(sorted(TUTOR_ACTIVITIES)), rng.choice(CEFR)
    l1n, l2n = LANGS[lang].split(" (")[0], LANGS[l2].split(" (")[0]
    more = ("longer phrases and whole sentences" if lvl in ("B2", "C1") else "short phrases" if lvl == "B1"
            else "single words and very short phrases")  # fmt: skip
    role = (f" In the role-play the tutor's in-role lines are in {l2n}; it steps out of the role in {l1n}."
            if act == "roleplay" else "")  # fmt: skip
    brief = (f"A language tutor. The learner's native language is {l1n}; they are learning {l2n} at level {lvl}. "
             f"The tutor talks to the learner in {l1n} (explanations, questions, feedback): asked in {l1n} how to say "
             f"something, it answers in {l1n} and says only the asked words in {l2n}. {l2n} is used for exactly what "
             f"is practised, at this level {more}.{role} Put the learner's languages, level and the session's "
             f"activity (and any material, e.g. the role-play situation) in the system prompt; the level code ({lvl}) is "
             f"never said aloud. "
             f"{TUTOR_ACTIVITIES[act].format(l1=l1n, l2=l2n)} The learner speaks {l1n} and tries {l2n}.")  # fmt: skip
    return {"l2": l2, "activity": act, "cefr": lvl, "brief": brief, "turns": rng.randint(8, 14),
            "prompt_lang": lang if rng.random() < 0.7 else "en" if lang != "en" else l2, "policy": "tutor"}  # fmt: skip


LONG_TASKS = ["planning a three-day city trip", "cooking a dish step by step while the user cooks along",
              "planning a birthday party", "setting up a monthly budget", "preparing for an exam over the next weeks",
              "assembling a piece of furniture step by step", "planning a small vegetable garden",
              "a training plan for a first 10 km run", "a checklist for moving to a new apartment",
              "writing a short speech for a friend's wedding", "choosing a used car", "planning a family dinner menu"]  # fmt: skip


def _longtask(lang, rng, rows):
    task = rng.choice(LONG_TASKS)
    what = (f"a multi-step task that fits the user's life (see WHO THE USER IS), of the kind of {task} but specific "
            f"to this person" if rows.get("personas") else task)  # fmt: skip
    brief = (f"A practical voice assistant helps with {what}. The system prompt only describes the assistant; the "
             f"user's situation and constraints (dates, budget, people, preferences, equipment, what they already "
             f"have) come up during the conversation, some only when the assistant asks, some changed later by the "
             f"user. They work through it step by step; the assistant keeps track of what the user said and decided, "
             f"refers back to it, and summarises briefly at the end (only what was said and decided).")  # fmt: skip
    return {"task": task, "brief": brief, "turns": rng.randint(14, 20)}


#: what a production assistant conversation is about underneath: the other scenarios' briefs
#: (service and grounded are left out: they need business facts / a passage in the prompt, which a general production
#: prompt does not have)
#: the general system prompt of the scenarios whose assistant is a general assistant: fixed texts per prompt language
#: (judge round 10: 5 of 64 dialogues put the topic or the answer itself into the "general" prompt, because the
#: writer is told to put every fact it states there); the writer copies the drawn one, ``repair`` enforces it
GENERAL_SCENARIOS = {"help", "skills", "assistant"}
GENERAL_PROMPTS = {
    "en": [
        "You are a helpful voice assistant. People talk to you by voice, so keep your answers short and natural, like "
        "spoken conversation. Be honest about what you don't know or can't do.",
        "You're a friendly AI voice assistant. Speak in a warm, relaxed way and keep replies to a few sentences unless "
        "the user asks for more. You have no internet access, no screen and no access to the user's personal data; if "
        "you're unsure about something, say so.",
        "You are an AI assistant that people talk to out loud. Answer clearly and concisely in plain spoken language, ask "
        "a short question when a request is unclear, and never make things up.",
        "You are a voice assistant. Be helpful, calm and to the point. Everything you say is heard, not read, so avoid "
        "lists and long explanations. If a question needs information you don't have, say so honestly and suggest how "
        "the user could find out.",
        "You are a conversational AI assistant. Be kind, curious and concise. You can't look things up, send messages or "
        "remember earlier conversations, so only offer what you can do right now by voice.",
    ],
    "de": [
        "Du bist ein hilfreicher Sprachassistent. Die Leute sprechen mit dir, also antworte kurz und natürlich, so wie "
        "man im Gespräch redet. Sei ehrlich, wenn du etwas nicht weißt oder nicht kannst.",
        "Du bist ein freundlicher KI-Sprachassistent. Sprich locker und herzlich und antworte in wenigen Sätzen, außer "
        "man bittet dich um mehr. Du hast keinen Internetzugang, keinen Bildschirm und keinen Zugriff auf persönliche "
        "Daten; wenn du dir unsicher bist, sag das.",
        "Du bist ein KI-Assistent, mit dem man sich unterhält. Antworte klar und knapp in gesprochener Alltagssprache, "
        "frag kurz nach, wenn eine Bitte unklar ist, und erfinde nichts.",
        "Du bist ein Sprachassistent. Sei hilfsbereit, ruhig und komm auf den Punkt. Alles, was du sagst, wird gehört "
        "und nicht gelesen, also vermeide Aufzählungen und lange Erklärungen. Wenn dir für eine Frage Informationen "
        "fehlen, sag das ehrlich und schlag vor, wie man es herausfinden kann.",
        "Du bist ein KI-Gesprächsassistent. Sei freundlich, neugierig und fasse dich kurz. Du kannst nichts "
        "nachschlagen, keine Nachrichten verschicken und dich nicht an frühere Gespräche erinnern; biete also nur an, "
        "was du jetzt im Gespräch tun kannst.",
    ],
    "zh": [
        "你是一个乐于助人的语音助手。用户是在跟你说话，所以回答要简短自然，像平时聊天一样。不知道或做不到的事情要如实说明。",
        "你是一个友好的人工智能语音助手。说话轻松亲切，除非用户要求详细，否则每次回答几句话就好。你不能上网，没有屏幕，也无法访问用户"
        "的个人信息；不确定的事情就直说。",
        "你是一个用户用语音交流的人工智能助手。用口语化的语言清楚、简洁地回答；请求不清楚时先简短地问一句，绝不编造信息。",
        "你是一个语音助手。要热心、沉稳、说重点。你说的话是被听到的而不是被读到的，所以不要列清单，也不要长篇解释。如果回答问题需要你"
        "没有的信息，就坦诚说明，并建议用户可以怎么查到。",
        "你是一个对话式人工智能助手。待人友善、好奇，说话简洁。你不能查资料、不能发消息，也记不住以前的对话，所以只提供你现在通过语音"
        "就能做到的帮助。",
    ],
    "ru": [
        "Ты — полезный голосовой ассистент. С тобой разговаривают голосом, поэтому отвечай коротко и естественно, как "
        "в обычном разговоре. Честно говори, если чего-то не знаешь или не можешь сделать.",
        "Ты — дружелюбный голосовой ИИ-ассистент. Говори непринуждённо и тепло и отвечай в нескольких предложениях, "
        "если тебя не просят рассказать подробнее. У тебя нет доступа к интернету, экрана и личным данным "
        "пользователя; если в чём-то сомневаешься, так и скажи.",
        "Ты — ИИ-ассистент, с которым разговаривают вслух. Отвечай ясно и кратко простым разговорным языком, коротко "
        "переспрашивай, если просьба непонятна, и ничего не выдумывай.",
        "Ты — голосовой ассистент. Будь отзывчивым, спокойным и говори по делу. Всё, что ты говоришь, слушают, а не "
        "читают, поэтому избегай перечислений и длинных объяснений. Если для ответа тебе не хватает информации, "
        "честно скажи об этом и подскажи, где её можно найти.",
        "Ты — разговорный ИИ-ассистент. Будь доброжелательным, любознательным и говори кратко. Ты не можешь ничего "
        "искать, отправлять сообщения или помнить прошлые разговоры, поэтому предлагай только то, что можешь сделать "
        "прямо сейчас голосом.",
    ],
}


ASSISTANT_TOPICS = {"help": 0.35, "chitchat": 0.2, "skills": 0.2, "longtask": 0.1, "tutor": 0.15}


#: share of production conversations the assistant opens with its greeting (the rest: the user speaks first)
ASSISTANT_OPENS = 0.9


def _assistant(lang, rng, rows):
    sub = _pick(rng, ASSISTANT_TOPICS)
    inner = SCENARIOS[sub](lang, rng, rows)
    opens = "assistant" if rng.random() < ASSISTANT_OPENS else "user"
    start = ("The conversation opens with the assistant's short greeting asking how it can help (without its name); "
             "then the user says what they want." if opens == "assistant" else "The user speaks first.")
    brief = (f"A production voice assistant: {PRODUCT['name']}, made by {PRODUCT['maker']}. The system prompt is the "
             f"GENERAL one it runs with for every user, as in a real product (it need not name the assistant or its maker): that it talks "
             f"with people by voice, how it speaks (friendly, concise, natural spoken language), that it is honest "
             f"about being an AI and about what it does not know or cannot do; nothing about this conversation's "
             f"topic or user. {start} What follows (ignore anything below about the system prompt's content: it stays "
             f"general): {inner['brief']}")  # fmt: skip
    # the inner scenario's fields at top level (a tutoring topic brings its policy, l2 and level for validation)
    return {**{k: v for k, v in inner.items() if k != "brief"}, "topic": sub, "brief": brief,
            "turns": inner["turns"] + (opens == "assistant"), "opens": opens, "product": dict(PRODUCT)}  # fmt: skip


SCENARIOS = {"service": _service, "grounded": _grounded, "help": _help, "chitchat": _chitchat, "skills": _skills,
             "tutor": _tutor, "longtask": _longtask, "assistant": _assistant}  # fmt: skip
#: source rows each scenario reads (SftSeeds' ``sources`` keys)
#: every scenario reads ``personas``: each dialogue's user is a persona (``make_seeds``, ``_persona_text``)
SCENARIO_SOURCES = {"service": ["personas"], "grounded": ["wikipedia", "personas"], "help": ["wildchat", "personas"],
                    "chitchat": ["personas"], "skills": ["wildchat", "personas"], "tutor": ["personas"],
                    "longtask": ["personas"], "assistant": ["wildchat", "personas"]}  # fmt: skip

#: Interruptions: the user cuts the assistant off mid-sentence (``chatterbox_v2_core.CUT`` marks the spot; the TTS
#: stops the assistant there and overlaps the user's turn). Per dialogue: how many (weights), and what the user does.
INTERRUPTIONS = {0: 0.55, 1: 0.3, 2: 0.15}
INTERRUPT_KINDS = {
    "new_topic": "jumps to a different subject or request; the assistant drops its unfinished point and follows",
    "correction": "corrects something the assistant got wrong or misunderstood; the assistant takes the correction",
    "hurry": "has heard enough or is in a hurry ('okay, okay, got it' / 'just tell me the price'); the assistant gets "
    "to the point at once",
    "question": "asks about something the assistant just said before it finishes; the assistant answers that",
    "stop": "tells the assistant to stop or wait (someone at the door, it is going off track); the assistant stops and "
    "picks up only what the user asks for next",
}


def make_seeds(scenario: str, lang: str, n: int, seed: int, rows: dict, start: int = 0) -> list[dict]:
    """``n`` seeds of ``scenario`` for users speaking ``lang``, indices ``start`` .. ``start + n - 1``: seed ``k`` is
    deterministic in (``seed``, ``k``) alone, so a shard of a cell is the same whatever the cell's size; ``cast``: the
    people's names and genders (own random stream)."""
    from .sft_names import draw_cast

    out = []
    for k in range(start, start + n):
        rng = random.Random(f"{scenario}/{lang}/{seed}/{k}")
        s = {"id": f"{scenario}_{lang}_{k}", "scenario": scenario, "lang": lang, **draw_language_setup(lang, rng),
             "level": _pick(rng, LEVEL_WEIGHTS)}  # fmt: skip
        # tutoring: the target languages in rotation (a random draw left some unseen in a small pilot)
        kw = {"l2": [x for x in LANGS if x != lang][(k + seed) % (len(LANGS) - 1)]} if scenario == "tutor" else {}
        s.update(SCENARIOS[scenario](lang, rng, rows, **kw))
        s["cast"] = draw_cast(lang, random.Random(f"{scenario}/{lang}/{seed}/{k}/cast"))
        if rows.get("personas") and not s.get("persona_id"):  # who the user is: own random stream
            pr = random.Random(f"{scenario}/{lang}/{seed}/{k}/persona")
            per = rows["personas"][pr.randrange(len(rows["personas"]))]
            s["persona"], s["persona_id"] = _persona_text_clean(per["text"]), per["id"]
        if scenario in GENERAL_SCENARIOS:  # own random stream: the other draws stay
            gr = random.Random(f"{scenario}/{lang}/{seed}/{k}/general")
            s["general_prompt"] = gr.choice(GENERAL_PROMPTS[s["prompt_lang"]])
        ex = random.Random(f"{scenario}/{lang}/{seed}/{k}/extra")
        if s["policy"] == "switch":  # in which of their turns the user switches (counting from 1)
            s["switch_at"] = ex.randint(2, max(2, (s["turns"] + 1) // 2))  # any user turn but the first
        s["interruptions"] = [_pick(ex, {k_: 1 for k_ in INTERRUPT_KINDS}) for _ in range(_pick(ex, INTERRUPTIONS))]
        out.append(s)
    return out


def _spoken_lang(seed: dict) -> str:
    return seed["lang"]


def _cast_text(seed: dict) -> str:
    c = seed["cast"]
    g = {"female": "a woman", "male": "a man"}
    others = "; ".join(f"{c[k]['full']} ({g[c[k]['gender']]})" for k in sorted(c) if k.startswith("other"))
    voice = g[c["assistant"]["gender"]]
    me = (f"{PRODUCT['name']}, an AI voice assistant made by {PRODUCT['maker']} (its voice is {voice}'s). It knows "
          f"this itself: the system prompt need not mention its name or maker and usually does not (in a business's role it is that "
          f"business's assistant). In the turns it says its name ONLY when the user asks for it (what "
          f"it is called, who or what it is), as \"{PRODUCT['spoken'][_spoken_lang(seed)]}\" (later just Haru), and "
          f"its maker only when asked who made it; otherwise neither comes up, not even in a greeting. It never "
          f"claims to have no name.")  # fmt: skip
    return (f"- The user: {c['user']['full']}, {g[c['user']['gender']]} (says their name only if the situation calls "
            f"for it; in a language that marks gender, everything the user says about themselves and every form "
            f"addressing them is {'feminine' if c['user']['gender'] == 'female' else 'masculine'}).\n- The assistant: {me}\n- Anyone else who comes up: {others}.\n- Use exactly these names, in "
            f"the turn's script and grammatical form; invent no other personal names (businesses, places and works "
            f"are yours to invent).")  # fmt: skip


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
        return (f"The user starts in {ln}; in the user's turn number {seed.get('switch_at', 2)} (counting the user's "
                f"turns from one) the user switches to {LANGS[seed['switch_to']]} (e.g. for a family member, practice, "
                f"or because it is easier; the user's words make the reason clear) and stays in it; the assistant follows the switch from its next turn "
                f"on.")  # fmt: skip
    return "as described in the scenario."  # tutor


_WATER = {"en": "water", "de": "Wasser", "ru": "вода", "zh": "水"}


def _mixing_rule(seed: dict) -> str:
    if seed["policy"] == "tutor":
        l1, l2 = LANGS[seed["lang"]].split(" (")[0], LANGS[seed["l2"]].split(" (")[0]
        role = (f" A role-play line spoken in role is a whole {l2} turn (\"lang\": the {l2} id, no tags)."
                if seed.get("activity") == "roleplay" else "")  # fmt: skip
        return (f"The tutor's explanations, questions and feedback are {l1} turns; only the practised {l2} words and "
                f"sentences are {l2}, each tagged inline, e.g. \"Water is <{seed['l2']}>{_WATER[seed['l2']]}</{seed['l2']}>.\"{role} Say a taught word inside a short phrase, never as a "
                f"bare character. No pinyin, transliteration or spelled-out pronunciation: the audio already says the "
                f"word. The learner's tries in {l2} are written in {l2}'s own script and tagged too.")  # fmt: skip
    return ("Stay in the turn's language. Only where a native speaker would really say something in another language "
            "(a title, a name, a fixed phrase), tag exactly that stretch, e.g. \"Hast du <en>Breaking Bad</en> "
            "gesehen?\". Everyday loanwords are not tagged. One- or two-letter abbreviations inside Russian or "
            "Chinese text are written out in that script.")


def _interrupt_text(seed: dict) -> str:
    ks = seed.get("interruptions") or []
    if not ks:
        return ""
    kinds = "; ".join(f"({i + 1}) the user {INTERRUPT_KINDS[k]}" for i, k in enumerate(ks))
    return f"""
INTERRUPTIONS
- {len(ks)} time(s) the user cuts the assistant off mid-sentence: {kinds}.
- Mark the spot in that assistant turn with {CUT}, at least three words in, and write a few more words after it as
  the turn would have gone on (they are not heard). The user's next turn reacts at once, only to the words before
  {CUT} (the rest was never heard); the assistant then follows
  the user and does not finish its old sentence. {CUT} only in assistant turns, never in the last turn.
"""


def _system_prompt_text(seed: dict) -> str:
    if seed.get("general_prompt"):
        return f"""SYSTEM PROMPT (fixed: the general assistant's instructions; copy it into "system_prompt" exactly)
{seed["general_prompt"]}
- Nothing about this conversation's topic is in it: the assistant states only general knowledge and what the user
  says; asked for anything specific beyond that, it says it does not have that information.
"""
    return f"""SYSTEM PROMPT (the assistant's instructions, in {LANGS[seed["prompt_lang"]]}, addressed to it as "you")
- Detail: {LEVELS[seed["level"]]}.
- Write it before the turns and put in it every specific fact the assistant will state that the user does not
  tell it (names, numbers, prices, times, policies, menu items, ingredients, availability, passage content, the
  learner's level). Beyond these and what the user says, the assistant uses only general knowledge; asked for
  anything else specific, it says it does not have that information.
- It holds nothing about this user's situation or what they will say (a business's records on file, e.g. a booking,
  only when the scenario needs them).
- It reads like a real deployment's instructions: no mention of a simulation, training, tags or text-to-speech.
"""


def _persona_text(seed: dict) -> str:
    if not seed.get("persona"):
        return ""
    return f"""
WHO THE USER IS
- {seed["persona"]}
- This shapes the user's situation, what they want and how they talk (occupation, interests, character), within the
  scenario. Their name is the one above, not any name here; details tied to another country are adapted to where
  the user lives (the scenario's place, else a place where the conversation's language is spoken).
"""


def n_turns(seed: dict) -> int:
    """The seed's turn count with the parity its order needs (it ends with the assistant): traces of the thinking
    writer spent long on "about 13 turns, starting with the user, ending with the assistant"."""
    return seed["turns"] + ((seed["turns"] % 2 == 0) != (seed.get("opens", "user") == "user"))


def writer_messages(seed: dict) -> list[dict]:
    opens, n = seed.get("opens", "user"), n_turns(seed)
    user = f"""Write one spoken conversation as JSON: {{"system_prompt": "...", "turns": [{{"speaker": "user" | "assistant", "lang": "<language id>", "text": "..."}}], "user_facts": ["..."]}}.

SCENARIO
{seed["brief"]}

{_system_prompt_text(seed)}
PEOPLE
{_cast_text(seed)}
{_persona_text(seed)}
USER DETAILS
- The user gives details the assistant cannot know (names, numbers, dates, their situation, preferences); now and
  then the user changes one later (optional). At least two later assistant turns depend on them.
- "user_facts": three to five of these details, each a few words copied from a user turn (same language, as said);
  none of them is in the system prompt.

CONVERSATION
- About {n} turns, alternating, starting with the {opens}, ending with the assistant.
- Language: {_policy_text(seed)}
- Each turn's "lang" is its main language ({", ".join(LANGS)}). {_mixing_rule(seed)} The text-to-speech speaks every
  stretch in its tag's (or turn's) language, so an untagged foreign word or a wrong tag is spoken wrong.
- Spoken style: no markdown, lists, emojis, URLs, brackets, slashes or symbols (a gloss or "and" is said as
  words). Whole numbers (amounts, prices, counts, years) as digits in English, German and Chinese turns (they
  are converted to words in the right form; 1250, not 1,250; a year after "in" / "im" / "seit"); in Russian turns as words in the right case. Times, dates, ordinals,
  decimals and fractions as words everywhere. Names in the turn's own script: a Chinese name in Chinese text is in
  characters, never pinyin.
- The assistant knows only its system prompt, general knowledge and what the user has said so far: it never uses
  the user's name or any of their details before the user says them (a caller is not greeted by name), and not
  today's date, day or time unless the system prompt or the user gives it.
- Assistant turns: one to three sentences unless the user asked for more. English turns use contractions as
  people speak; a person is called by their first name (or as the user does) after the first mention. The user sounds like a real person
  (informal, sometimes a filler or an unfinished sentence).
- The assistant is voice-only: no tools, no screen, nothing in the background. It cannot send, save, look up, remind
  or get back later; whatever it produces it says aloud now, and what it cannot do it says briefly, offering what it
  can do by voice. In a business's role, the system prompt says what the line can do (e.g. take a booking request
  that staff confirm later) and the assistant claims nothing beyond it.
- Everything the assistant says is correct: facts, arithmetic, and in tutoring the target-language forms and
  grammar rules. Every word is spelled correctly.
{_interrupt_text(seed)}"""
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user}]


#: property order is generation order under guided decoding: the facts come after the turns, picked from what the
#: user said (before the turns, round 5: the writer planned facts and then wrote turns that did not say them)
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
        "user_facts": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string"}},
    },
    "required": ["system_prompt", "turns", "user_facts"],
}

#: the review pass: the problems are generated first (a short written check before the corrected dialogue, cheaper
#: than thinking: rounds 1 / medium thinking cost 3-5x without a judged gain)
REVIEW_SCHEMA = {"type": "object",
                 "properties": {"claims": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {"turn": {"type": "integer"}, "claim": {"type": "string"},
                                                   "source": {"type": "string", "enum": ["system_prompt", "user",
                                                                                         "general_knowledge", "none"]}},
                                    "required": ["turn", "claim", "source"]}},
                                "problems": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                                **WRITER_SCHEMA["properties"]},
                 "required": ["claims", "problems", *WRITER_SCHEMA["required"]]}  # fmt: skip

#: what an independent judge found most in rounds 0-6 (Opus over 48 dialogues per round: invented facts, invented
#: abilities, knowing what was not said, arithmetic, wrong tutoring content, grammar, reactions to unheard words,
#: system prompts no product would ship)
REVIEW_TEXT = """Now review this conversation as a strict editor before it is used as training data. First list in \
"claims" every specific thing the assistant states (a price, time, date, schedule, name, place, policy, ingredient, \
availability, a record, a plot, a count, a sum) with its turn number and where it comes from: "system_prompt", \
"user" (said earlier by the user), "general_knowledge" (safe, widely known) or "none". Then list in "problems" each \
concrete defect of these kinds (quote it, one short sentence each; every claim with source "none" is one), then give \
the corrected conversation, changing only what is needed:
1. The assistant states a specific fact (price, time, schedule, menu, ingredient, policy, availability, a record, \
a date, a plot, a person's details) that is neither in the system prompt, nor said by the user, nor safe common \
knowledge: it must say it does not have that information instead (or add the fact to the system prompt if it \
belongs to the business or passage).
2. The assistant claims to do anything besides speaking now (booking, noting, passing on, checking a system, \
sending, reminding, calling back, remembering earlier calls, seeing something), unless the system prompt says \
this line can do it.
3. The assistant uses something the user has not said yet (their name, a detail), or assumes today's date, day \
or time without being told.
4. Arithmetic, counts, dates and units: recompute every number.
5. Tutoring: every target-language form, translation and grammar explanation must be correct and natural.
6. Grammar, spelling, gender agreement and idiom of every turn as a native speaker would say it.
7. After an interruption mark, the next user turn reacts only to the words before it.
8. The system prompt reads like a real deployment: no user details or answers in advance, no writer \
instructions, no simulation talk.
9. Every other rule of the original task still holds (languages, tags, how numbers are written, turn order, \
interruption marks, user_facts copied from user turns).
If there is nothing to fix, "problems" is empty and the conversation is returned unchanged."""

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


#: English and German share a script, so a forgotten tag between them is found by word frequency (wordfreq Zipf
#: scale): a word of three or more letters common in the other language (>= 3.5) and rare in this one (< 2.0). Pilot
#: p10: 5 of 1,263 en / de turns, all real misses ("'read' is 'lesen'", an English sentence in a German turn).
_LATIN_OTHER = {"en": "de", "de": "en"}
LATIN_COMMON, LATIN_RARE = 3.5, 2.0


def _untagged_latin(text: str, lang: str) -> list[str]:
    """Words of ``text`` (an en / de segment) that belong to the other of the two."""
    if lang not in _LATIN_OTHER:
        return []
    from wordfreq import zipf_frequency

    o = _LATIN_OTHER[lang]
    return [w for w in re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", text)
            if zipf_frequency(w.lower(), o) >= LATIN_COMMON and zipf_frequency(w.lower(), lang) < LATIN_RARE]  # fmt: skip


def _script_ok(text: str, lang: str) -> bool:
    """Most letters of ``text`` are in ``lang``'s script (zh: CJK, ru: Cyrillic, en/de: Latin)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    hit = sum(bool(_SCRIPT[lang].match(c)) for c in letters)
    return hit >= 0.6 * len(letters)


def validate(obj: dict, seed: dict) -> str | None:
    """Why the writer's ``obj`` is unusable for ``seed`` (None = fine)."""
    from .chatterbox_v2_core import parse_segments, plain_text, speech_units

    turns, sp = obj.get("turns") or [], (obj.get("system_prompt") or "").strip()
    if not sp:
        return "empty system prompt"
    if not _script_ok(sp, seed["prompt_lang"]):
        return "system prompt not in prompt_lang"
    if len(turns) < max(2, n_turns(seed) - 3) or len(turns) > n_turns(seed) + 6:  # the writer runs long: p4 0..+4
        return "length"
    first = seed.get("opens", "user")
    second = "assistant" if first == "user" else "user"
    if seed.get("cefr") and any(_LEVEL_SAID.search(t["text"]) for t in turns):
        return "the learner's level code is said aloud"
    for k, t in enumerate(turns):
        if t["speaker"] != (first if k % 2 == 0 else second):
            return f"speakers do not alternate from the {first}"
        if not t["text"].strip():
            return "empty turn"
        if _MARKDOWN.search(t["text"]):
            return "markdown"
        try:
            segs = parse_segments(t["text"].replace(CUT, " "), t["lang"])
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
            if _untagged_latin(s["text"], s["lang"]):
                return f"untagged {LANGS[_LATIN_OTHER[s['lang']]]} words in {LANGS[s["lang"]]} text"
            if re.search(r"\d", s["text"]):
                return "digits (numbers must be words)"
            if _WRITTEN_ONLY.search(s["text"]):
                return "written-only symbols (brackets, +, /, *, %...)"
            if s["lang"] in ("ru", "zh") and _LAT.search(s["text"]):
                return f"untagged Latin-script words in a {s['lang']} stretch"
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
        nth = sum(1 for t in turns[: k + 1] if t["speaker"] == "user")
        if abs(nth - seed.get("switch_at", nth)) > 1:
            return "the user switches in the wrong turn"
        if any(t["lang"] != to for t in turns[k:]):
            return "the dialogue does not stay in the switched language"
    if _garbled(sp + " " + " ".join(t["text"] for t in turns)):
        return "garbled characters"
    if _SP_MARKUP.search(sp):
        return "markup (language tags / cut marks) in the system prompt"
    if seed["lang"] == "zh" and _pinyin_name(turns):
        return "a Chinese name in pinyin inside Chinese text"
    if "cast" in seed and _early_user_name(sp, turns, seed["cast"]["user"], seed["lang"]) is not None:
        return "the assistant uses the user's name before hearing it"
    if FOREIGN_AI.search(sp + " " + " ".join(t["text"] for t in turns)):
        return "names another AI company or model"
    if _NO_NAME.search(" ".join(t["text"] for t in turns if t["speaker"] == "assistant")):
        return "the assistant says it has no name"
    why = _name_problem(sp, turns)
    if why:
        return why
    if seed["policy"] == "tutor":
        why = _tutor_problem(turns, seed)
        if why:
            return why
    want = len(seed.get("interruptions") or [])
    cuts = [(k, t) for k, t in enumerate(turns) if CUT in t["text"]]
    ok_n = {0} if want == 0 else set(range(max(1, want - 1), want + 2))  # the count is a draw; off by one is fine
    if len(cuts) not in ok_n or any(t["text"].count(CUT) != 1 for _, t in cuts):
        return f"{len(cuts)} interruptions, want {want}"
    for k, t in cuts:
        if t["speaker"] != "assistant" or k == len(turns) - 1:
            return "an interruption outside an assistant turn (or in the last turn)"
        pre, post = (plain_text(x) for x in t["text"].split(CUT))
        if speech_units(pre, t["lang"]) < (4 if t["lang"] == "zh" else 3) or not speech_units(post, t["lang"]):
            return "an interruption without words before / after the cut"
    if _cut_leak(sp, turns, seed["lang"]) is not None:
        return "the user reacts to words after the cut (never heard)"
    why = _user_facts_problem(obj, seed)
    if why:
        return why
    if "cast" in seed:
        names = [seed["cast"][k]["given"] for k in seed["cast"]]
        said = sp + " ".join(t["text"] for t in turns)
        for bad in FREQUENT_INVENTED.get(seed["lang"], ()):
            if bad not in names and re.search(rf"(?<!\w){re.escape(bad)}(?!\w)", said):
                return "a personal name outside the cast"
    return None


#: characters a TTS cannot speak or reads oddly (p12: "хотел(а)", "R+V", "list(data.keys())[one]", "*Titel*")
_WRITTEN_ONLY = re.compile(r"[()（）\[\]{}+=*/\\|_~^$€£@#%&]")
#: a CEFR level code in a turn, written or spelled out (the tutor brief: never said aloud)
_LEVEL_SAID = re.compile(r"(?<![\w-])[ABC] ?[12](?!\w)|(?<![\w-])[ABC] (one|two|eins|zwei)\b")
#: rarer than this (wordfreq Zipf) counts as a content word for the cut-leak check (common words like "после" / 因为
#: recur by chance)
CUT_LEAK_ZIPF = 5.0
_ZH_COMMON_BIGRAMS = set("因为 所以 可以 我们 你们 他们 这个 那个 什么 没有 一下 时候 现在 已经 如果 但是 还是 就是 不是 一个 知道 "
                         "觉得 需要 一些 这样 那样 怎么 为什么 真的 其实 然后 的话 一点 比较 应该".split())


def _cut_units(text: str, zh: bool) -> set[str]:
    """Content units: 5-letter word stems (en / de / ru), character bigrams (zh)."""
    x = re.sub(r"</?[a-z]{2}>", " ", text).lower()
    if zh:
        return {r[i : i + 2] for r in re.findall(r"[\u4e00-\u9fff]+", x) for i in range(len(r) - 1)}
    return {w[:5] for w in re.findall(r"\w+", x) if len(w) >= 5}


def _cut_leak(sp: str, turns: list[dict], lang: str) -> int | None:
    """The first interrupted turn whose unheard rest (after the cut) the next user turn picks up: a rare content unit
    in the rest and in the user's reply but nowhere before (judge, gemma round 7: 8 of 64 dialogues; this found 6 of
    those 8)."""
    from wordfreq import zipf_frequency

    def rare(u: str, lg: str) -> bool:  # wordfreq's Chinese needs jieba (not installed): a short stop list instead
        return lg != "zh" or u not in _ZH_COMMON_BIGRAMS

    for k, t in enumerate(turns):
        if t["speaker"] != "assistant" or CUT not in t["text"] or k + 1 >= len(turns):
            continue
        pre, post = t["text"].split(CUT, 1)
        lg = t.get("lang", lang)
        zh = lg == "zh"
        heard = " ".join([sp, *(x["text"].split(CUT)[0] for x in turns[:k]), pre])
        new = (_cut_units(post, zh) & _cut_units(turns[k + 1]["text"], zh)) - _cut_units(heard, zh)
        if not zh:  # a stem of a common word recurs by chance
            words = {w.lower() for w in re.findall(r"\w+", post) if len(w) >= 5}
            new = {u for u in new if any(w.startswith(u) and zipf_frequency(w, lg) < CUT_LEAK_ZIPF for w in words)}
        if any(rare(u, lg) for u in new):
            return k
    return None


#: mis-decoded text: the replacement character or UTF-8 read as Latin-1
_MOJIBAKE = re.compile(r"\ufffd|Ã[\x80-\xbf¡-ÿ]|â€|Ð[\x80-\xbf]|Ñ[\x80-\xbf]|ä¸|å[\x80-\xbf]")


def _garbled(text: str) -> bool:
    """Broken text (judge round 9: gemma fp8 wrote a Chinese system prompt with 佋渁層問吪 and an Ethiopic ሴ):
    mojibake, a character from a script we do not speak, or three or more distinct Chinese characters outside GB2312
    (simplified text needs almost none; one rare term like 夾纈 from a passage is allowed)."""
    if _MOJIBAKE.search(text):
        return True
    rare = set()
    for c in text:
        o = ord(c)
        if 0x3400 <= o <= 0x9FFF:
            try:
                c.encode("gb2312")
            except UnicodeEncodeError:
                rare.add(c)
        elif o >= 0x0530 and not (0x2000 <= o <= 0x206F or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF
                                  or 0x20A0 <= o <= 0x20CF):  # fmt: skip
            return True
    return len(rare) >= 3
#: tags or the cut mark inside the system prompt (p12: the writer told the assistant to "tag Russian with <ru>")
_SP_MARKUP = re.compile(r"</?[a-z]{2}>|<cut/>")
_PY_SYL = (r"(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?(?:iang|iong|uang|uan|uai|ang|eng|ing|ong|ian|iao|ai|ei|ao|ou|an|en|er|"
           r"in|un|ia|ie|iu|ua|uo|ui|ue|a|o|e|i|u|v)")  # fmt: skip
_PY_WORD = re.compile(rf"[A-Z](?:{_PY_SYL})+", re.I)


def _pinyin_name(turns: list[dict]) -> str | None:
    """An ``<en>`` span of two or more capitalised pinyin words in a Chinese turn: a Chinese person's name romanised
    (p12: "<en>Huang Bo</en>" from an English system prompt), which the English voice reads as English."""
    for t in turns:
        for m in re.finditer(r"<en>(.*?)</en>", t["text"]):
            w = m.group(1).split()
            if len(w) >= 2 and all(x[0].isupper() and _PY_WORD.fullmatch(x) for x in w):
                return m.group(1)
    return None


def _early_user_name(sp: str, turns: list[dict], user: dict, lang: str) -> int | None:
    """The first assistant turn that calls the user by a name neither the system prompt nor an earlier turn gave
    (p12 no-thinking: 21 of 189 kept, mostly a caller greeted by name)."""
    if lang == "zh":
        ps = [re.escape(user["full"]), re.escape(user["family"]) + "(?:女士|先生|小姐|老师)"]
        ps += [re.escape(user["given"])] if len(user["given"]) >= 2 else []
    elif lang == "ru":  # inflected: Роман / Романа
        ps = [rf"(?<!\w){re.escape(n[: max(3, len(n) - 2)])}\w*" for n in (user["given"], user["family"])]
    else:
        ps = [rf"(?<!\w){re.escape(n)}(?!\w)" for n in (user["given"], user["family"])]
    rx, heard = [re.compile(p) for p in ps], sp
    for k, t in enumerate(turns):
        if t["speaker"] == "assistant" and any(r.search(t["text"]) and not r.search(heard) for r in rx):
            return k
        heard += " " + t["text"]
    return None


#: names the writer invented most often in the pilot (2026-11-17); one in a dialogue whose cast lacks it = the writer
#: ignored the cast
FREQUENT_INVENTED = {"en": ["Sarah", "Alex"], "de": ["Lena", "Lukas"], "ru": ["Анна", "Алексей", "Алиса"],
                     "zh": ["王强", "张伟", "李华"]}  # fmt: skip


#: pinyin with tone marks (macron / caron vowels): the TTS already speaks the characters, a romanisation after them is
#: read out a second time
_PINYIN_STRICT = re.compile(r"[āēīōūǖǎěǐǒǔǚǘǜ]|\b[a-z]+[1-5]\b")
#: the most share of a (non-role-play) tutor's speech in the target language (characters, Chinese x3): it talks in
#: the learner's language and says only the practised words in the target language
TUTOR_MAX_L2 = 0.5


def _tutor_problem(turns: list[dict], seed: dict) -> str | None:
    from .chatterbox_v2_core import parse_segments

    l1, l2, n_l2, n_all, with_l2, n_asst = seed["lang"], seed["l2"], 0, 0, 0, 0
    roleplay = seed.get("activity") == "roleplay"
    for t in turns:
        if t["speaker"] != "assistant":
            continue
        n_asst += 1
        if not roleplay and t["lang"] != l1:
            return "a tutor turn not in the learner's language"
        segs = parse_segments(t["text"].replace(CUT, " "), t["lang"])
        has = False
        for s in segs:
            w = len(s["text"]) * (3 if s["lang"] == "zh" else 1)
            n_all += w
            if s["lang"] == l2:
                n_l2, has = n_l2 + w, True
            if s["lang"] != "zh" and _PINYIN_STRICT.search(s["text"]):
                return "romanisation / pronunciation spelled out (the TTS speaks the word already)"
        with_l2 += has
    if not roleplay and n_all and n_l2 / n_all > TUTOR_MAX_L2:
        return "the tutor speaks the target language beyond the practised words"
    if with_l2 < n_asst / 2:
        return "most tutor turns have no target language"
    return None


#: the assistant's name in any form (``PRODUCT['spoken']``)
_PRODUCT_NAME = re.compile(r"(?i)haru|хару|\bi ?6\b|\bi six\b|\bi sechs\b|и-шесть")
#: the user asking the assistant's name or who / what it is (en, de, ru, zh)
_ASKS_NAME = re.compile(
    r"(?i)\bnames?\b|who are you|who am i (talking|speaking)|what are you|who('s| is) (this|speaking)|what should i call|"
    r"Namen?\b|wer bist du|wer sind sie|mit wem (spreche|rede)|wie heißt|wie heissen|was bist du|"
    r"зовут|имя|имени|кто ты|кто вы|ты кто|вы кто|кто такая|кто такой|с кем я|как к вам обращаться|как тебя назвать|"
    r"名字|你是谁|您是谁|叫什么|怎么称呼|在跟谁|和谁说话|你是什么"
)


def _name_problem(sp: str, turns: list[dict]) -> str | None:
    """An assistant turn may say the name only right after a user turn asking who / what it is or its name (the system
    prompt need not state it: the model learns its identity)."""
    for k, t in enumerate(turns):
        if t["speaker"] == "assistant" and _PRODUCT_NAME.search(t["text"]):
            prev = next((u["text"] for u in reversed(turns[:k]) if u["speaker"] == "user"), "")
            if not _ASKS_NAME.search(prev):
                return "the assistant says its name unasked"
    return None


#: the assistant claiming to have no name (pilot p10: "I don't have a specific name", "我没有具体的产品名字"); every
#: assistant has one (the cast's, or the production name)
_NO_NAME = re.compile(
    r"(?i)(\bno (specific |particular |real |actual |product |own )?name\b|(do not|don't|doesn't|haven't) (really )?"
    r"(have|got) a (specific |particular |real |proper )?name|without a name|\bnameless\b|keinen (eigenen |richtigen |"
    r"bestimmten )?Namen|\bnamenlos|нет (собственного |своего |конкретного )?имени|без имени|没有(具体的|自己的)?(产品)?"
    r"(名字|名称)|不需要名字)"
)


def _norm_fact(x: str) -> str:
    return re.sub(r"[\W_]+", " ", x.lower()).strip()


def _fact_units(x: str) -> list[str]:
    """A fact's matching units: its words, or its characters where there are no spaces (Chinese)."""
    n = _norm_fact(x)
    return [c for c in n if c != " "] if _CJK.search(n) else n.split()


def _covered(fact: str, text: str, share: float = 0.7) -> bool:
    """At least ``share`` of the fact's units occur in ``text`` (the writer quotes loosely: "ich studiere in Aachen"
    for "Ich studiere ja in Aachen")."""
    u = _fact_units(fact)
    t = set(_fact_units(text))
    stems = {w[:5] for w in t if len(w) >= 5}  # inflection: "meine Abschlussarbeit" / "meiner Abschlussarbeit"
    return bool(u) and sum(1 for x in u if x in t or (len(x) >= 5 and x[:5] in stems)) / len(u) >= share


def _user_facts_problem(obj: dict, seed: dict) -> str | None:
    # tutoring: the learner's own tries are what the tutor works with; one personal detail is enough (round 5: the
    # writer described the session as "facts" instead of quoting, 14 of 32 tutor seeds lost)
    need = 1 if seed.get("policy") == "tutor" else 2
    facts = [f for f in (obj.get("user_facts") or []) if _norm_fact(f)]
    if len(facts) < need:
        return "fewer than two user facts"
    users = [_norm_fact(t["text"]) for t in obj["turns"] if t["speaker"] == "user"]
    sp = _norm_fact(obj.get("system_prompt") or "")
    said = " ".join(users)
    ok = [f for f in facts if _covered(f, said) and not _covered(f, sp, 0.9)]
    if len(ok) < need:
        return "fewer than two user facts said by the user and absent from the system prompt"
    return None


_TAGGED = re.compile(r"(<([a-z]{2,3}(?:-[a-z]{2,4})?)>.*?</\2>|<cut/>)", re.S)
_LAT_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’.\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'’.\-]*)*")
_NUM = re.compile(r"(?<![\w.,])(\d+)(?![\w.,]\d|\w)")
#: German: a number with a dot before a lower-case word or a month is an ordinal ("am 3. Mai"): left as digits (then
#: rejected) instead of read as a cardinal
_NUM_DE = re.compile(r"(?<![\w.,])(\d+)(?![\w.,]\d|\w|\.\s*(?:[a-zäöü]|Januar|Februar|März|April|Mai|Juni|Juli|"
                     r"August|September|Oktober|November|Dezember))")


def _tag_en(m, lang: str = "en") -> str:
    w = m.group(0)
    core = w.rstrip(".-'’")
    if len(re.sub(r"[^A-Za-z]", "", core)) < 3:  # "C", "HN", "AI" alone: the TTS mangles 1-2 letter segments
        return w  # (pilot round 3: "C" -> "See, say the pocket") -> left untagged, so the check rejects it
    return f"<{lang}>{core}</{lang}>{w[len(core):]}"


_ZH_DIGITS = "零一二三四五六七八九"


def _zh_section(n: int) -> str:
    out, zero = "", False
    for unit, name in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // unit % 10
        if d:
            # spoken: 两千 / 两百, not 二千 (judge: "二千" unnatural); 十 and units keep 二
            out += ("零" if zero else "") + ("两" if d == 2 and name in ("千", "百") else _ZH_DIGITS[d]) + name
            zero = False
        elif out:
            zero = True
    return out


def zh_number(n: int) -> str:
    """Mandarin reading of a non-negative integer below 10^8 (1980 -> 一千九百八十, 12 -> 十二)."""
    if n == 0:
        return "零"
    hi, lo = divmod(n, 10000)
    out = ("两" if hi == 2 else _zh_section(hi)) + "万" if hi else ""
    if lo:
        out += ("零" if hi and lo < 1000 else "") + _zh_section(lo)
    return out[1:] if out.startswith("一十") else out


_ZH_NUM = re.compile(r"(\d+(?:\.\d+)?)(%|年)?")
#: measure words / units after which a lone 2 is read 两 (judge round 9: 二公里, 二瓶, 二厘米, 二点)
_ZH_MEASURE = set("个只条张本件位次天周点瓶杯片块公厘米斤分钟小岁口套双台辆家门节间份颗棵根把首部篇种样遍趟页")


def _zh_num_sub(m) -> str:
    x, suf = m.group(1), m.group(2) or ""
    if x == "2" and not suf and m.string[m.end() : m.end() + 1] in _ZH_MEASURE:  # 两瓶 / 两点, not 二瓶
        return "两"
    if suf == "年" and len(x) == 4 and x.isdigit():  # years digit by digit: 一九八〇年
        return "".join("〇一二三四五六七八九"[int(c)] for c in x) + "年"
    whole, _, frac = x.partition(".")
    if len(whole) > 8:
        return m.group(0)
    w = zh_number(int(whole)) + ("点" + "".join(_ZH_DIGITS[int(c)] for c in frac) if frac else "")
    return ("百分之" + w) if suf == "%" else w + suf


_CEFR = re.compile(r"\b([ABC])([12])\b")
_CEFR_WORD = {"en": ("one", "two"), "de": ("eins", "zwei")}
#: an untagged stretch in another script is in the one language of ours that uses that script
_SCRIPT_RUN = {"ru": re.compile(r"[\u0400-\u04ff][\u0400-\u04ff\s'’-]*[\u0400-\u04ff]|[\u0400-\u04ff]"),
               "zh": re.compile(r"[\u3400-\u9fff][\u3400-\u9fff，、]*[\u3400-\u9fff]|[\u3400-\u9fff]")}  # fmt: skip


_PERCENT = {"en": " percent", "de": " Prozent"}
#: a four-digit number followed by one of these is an amount, not a year ("1250 euros"); judge round 9: "released
#: 1984" was read "one thousand nine hundred and eighty-four" when only a preceding year word made a year
_AMOUNT_AFTER = {"euro", "euros", "dollar", "dollars", "pound", "pounds", "franken", "francs", "rubel", "rubles",
                 "yuan", "people", "personen", "leute", "km", "kilometers", "kilometres", "kilometer", "meter", "meters",
                 "metres", "steps", "schritte", "calories", "kalorien", "kcal", "words", "wörter", "pieces", "stück",
                 "grams", "gramm", "g", "ml", "items", "teile", "times", "mal", "minutes", "minuten", "seconds",
                 "sekunden", "hours", "stunden", "days", "tage", "points", "punkte", "copies", "units", "einheiten",
                 "square", "quadratmeter", "kilo", "kilos", "kg", "liter", "liters", "litres", "chf", "usd", "eur"}
_YEAR_BEFORE = {"in", "im", "seit", "since", "until", "till", "bis", "von", "from", "um", "around", "circa", "ca.",
                "year", "jahr", "jahre", "anno", "ab", "before", "after", "vor", "nach", "by", "of", "des", "early",
                "late", "mid", "summer", "winter", "spring", "autumn", "sommer", "frühjahr", "herbst"}
_AND = {"en": "and", "de": "und", "ru": "и", "zh": "和"}


def _spoken_punct(x: str, lg: str) -> str:
    """Written-only marks a speaker would say as pauses or words (p13: 31 rejections): a bracketed gloss becomes a
    comma-set phrase ("'ein' (<zh>一个</zh>) richtig" -> "'ein', <zh>一个</zh>, richtig"), emphasis asterisks go,
    "&" between words becomes the turn language's "and"."""
    comma = "，" if lg == "zh" else ","
    x = re.sub(r"\*([^*\n]+)\*", r"\1", x)
    sp = "" if lg == "zh" else " "
    x = re.sub(r"\s*[(（]([^()（）\[\]]+)[)）]\s*([,.;:!?，。；：！？]?)",
               lambda m: f"{comma}{sp}{m.group(1).strip()}{m.group(2) or comma}{sp}", x)
    x = re.sub(r"(?<=\w) & (?=\w)", f" {_AND.get(lg, 'and')} ", x)
    x = re.sub(r"\s+([,，。.])", r"\1", x)
    return re.sub(r"([,，])\s*([,，.。!?！？])", r"\2", re.sub(r" {2,}", " ", x)).strip()
_SPOKEN_NAME = re.compile(r"(?i)haru[ -]i[ -]?(?:six|sechs|6)[ -]instruct|хару и-шесть инстракт")


def repair(obj: dict, seed: dict | None = None) -> dict:
    """Mechanical fixes before validation (the writer misses them often enough to drop many dialogues, 2026-11-17:
    64 of 160 kept): untagged Latin-script runs in Russian / Chinese turns get tagged ``<en>`` (product and brand
    names, acronyms); in English / German turns, plain integers become words (``num2words``; four-digit numbers
    1100-2099 as years; CEFR level codes "B2" -> "B two"), in Chinese turns Chinese numerals (:func:`zh_number`; years before 年 digit by digit);
    traditional Chinese characters become simplified. Russian numbers stay a rejection: their inflection needs the
    sentence. Structure (round 5, 256 seeds: alternation and interruption placement were the main losses after the
    user facts): consecutive turns of one speaker in one language are merged, a ``<cut/>`` in a user turn or in the
    last turn is dropped. Untagged stretches in another of our scripts get that script's language tag (Cyrillic ru,
    Chinese zh; Latin in Russian / Chinese turns: the tutor's target language if it is en / de, else en)."""
    from num2words import num2words

    def outside_tags(text: str, fn) -> str:
        parts = _TAGGED.split(text)
        out, k = [], 0
        while k < len(parts):
            out.append(fn(parts[k]))
            if k + 1 < len(parts):
                out.append(parts[k + 1])
            k += 3
        return "".join(out)

    def num(m, lg):
        # a year only after a year word ("in 1999", "seit 1850"): amounts are digits since writer v8 ("1250 Euro")
        v, before, after = int(m.group(1)), m.string[: m.start()].split(), m.string[m.end() :].split()
        nxt = re.sub(r"\W", "", after[0]).lower() if after else ""
        year = len(m.group(1)) == 4 and 1100 <= v <= 2099 and (
            (before and before[-1].lower() in _YEAR_BEFORE) or nxt not in _AMOUNT_AFTER)
        return (num2words(v, lang=lg, to="year") if year else num2words(v, lang=lg)).replace(",", "")

    from opencc import OpenCC

    t2s = OpenCC("t2s").convert
    if seed is not None and seed.get("general_prompt"):  # the fixed general prompt, whatever the writer returned
        obj["system_prompt"] = seed["general_prompt"]
    if isinstance(obj.get("system_prompt"), str):  # the TTS spelling of the name leaks into the (text) prompt
        obj["system_prompt"] = _SPOKEN_NAME.sub(PRODUCT["name"], obj["system_prompt"])
        if _CJK.search(obj["system_prompt"]):  # traditional characters in a Chinese prompt too (turns: below)
            obj["system_prompt"] = t2s(obj["system_prompt"])  # the writer slips into traditional characters, which Chatterbox's zh misreads
    for t in obj.get("turns") or []:
        lg, x = t.get("lang"), t.get("text") or ""
        if _CJK.search(x):
            x = t2s(x)
        if lg == "zh":
            x = outside_tags(x, lambda s: _ZH_NUM.sub(_zh_num_sub, s))
            # "共 3 个" (spaces the writer puts around digits) -> "共三个": no space between Chinese characters
            x = outside_tags(x, lambda s: re.sub(r"(?<=[\u3400-\u9fff，。、])[ \t]+(?=[\u3400-\u9fff])", "", s))
        for other, rx in _SCRIPT_RUN.items():
            if lg != other and lg in LANGS:
                x = outside_tags(x, lambda s, o=other, r=rx: r.sub(lambda m: f"<{o}>{m.group(0)}</{o}>", s))
        if lg in ("ru", "zh"):
            lat = (seed or {}).get("l2") if (seed or {}).get("l2") in ("en", "de") else "en"
            x = outside_tags(x, lambda s, lat=lat: _LAT_RUN.sub(lambda m: _tag_en(m, lat), s))
        elif lg in ("en", "de"):
            if not (seed or {}).get("cefr"):  # tutoring: a level code said aloud is rejected instead (spec: never said)
                x = outside_tags(x, lambda s, lg=lg: _CEFR.sub(lambda m: f"{m.group(1)} {_CEFR_WORD[lg][int(m.group(2)) - 1]}", s))
            x = outside_tags(x, lambda s, lg=lg: (_NUM_DE if lg == "de" else _NUM).sub(lambda m: num(m, lg), s))
            x = outside_tags(x, lambda s, lg=lg: re.sub(r"\s?%", _PERCENT[lg], s))
        t["text"] = _spoken_punct(x, lg)
    turns = obj.get("turns") or []
    for k, t in enumerate(turns):  # an interruption only cuts an assistant turn that is followed by a user turn
        if CUT in (t.get("text") or "") and (t.get("speaker") != "assistant" or k == len(turns) - 1):
            t["text"] = re.sub(r"\s*<cut/>\s*", " ", t["text"]).strip()
    merged = []
    for t in turns:
        p = merged[-1] if merged else None
        if p and p.get("speaker") == t.get("speaker") and p.get("lang") == t.get("lang") and CUT not in p["text"]:
            p["text"] = p["text"] + ("" if t.get("lang") == "zh" else " ") + t["text"]
        else:
            merged.append(t)
    if turns:
        obj["turns"] = merged
    langs = [t.get("lang") for t in obj.get("turns") or [] if t.get("speaker") == "user"]
    lg = max(set(langs), key=langs.count) if langs else None
    if lg in ("en", "de") and isinstance(obj.get("user_facts"), list):  # quoted from the turns: same numbers as words
        obj["user_facts"] = [_NUM.sub(lambda m: num(m, lg), f) if isinstance(f, str) else f for f in obj["user_facts"]]
    return obj


def assistant_voice_lang(obj: dict, seed: dict) -> str:
    """The language the assistant speaks most in the dialogue (characters; Chinese ones count three): its voice
    prompt comes from that language's pool, since Chatterbox carries the prompt's accent into every language (a
    German-pool voice speaking Russian after a switch sounded German, 2026-11-17)."""
    from .chatterbox_v2_core import parse_segments

    n: dict = {}
    for t in obj["turns"]:
        if t["speaker"] == "assistant":
            for s in parse_segments(t["text"].replace(CUT, " "), t["lang"]):
                n[s["lang"]] = n.get(s["lang"], 0) + len(s["text"]) * (3 if s["lang"] == "zh" else 1)
    return max(n, key=n.get) if n else seed["lang"]


def to_tts_dialogue(obj: dict, seed: dict) -> dict:
    """ChatterboxInferenceV2's input for a written dialogue: turns with languages (``<cut/>`` kept: the TTS stops the
    turn there); voices of each person's gender (``<lang>/<role>/<gender>/rng_*``, ``SftVoiceDir``): the user's from
    the user's language (an accent in a switched language is the user's own), the assistant's from the language it
    speaks most."""
    c = seed["cast"]
    vl = {"user": seed["lang"], "assistant": assistant_voice_lang(obj, seed)}
    return {"lang": seed["lang"], "turns": [{"speaker": t["speaker"], "lang": t["lang"], "text": t["text"]} for t in obj["turns"]],
            "voices": {r: f"{vl[r]}/{r}/{c[r]['gender']}/rng_v" for r in ("assistant", "user")}}  # fmt: skip


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


#: a Nemotron persona opens with the person's full name ("Gary Dann is a meticulous ..."): replaced, the cast names
#: the people
_PERSONA_NAME = re.compile(r"^(?!(?:A|An|The)\b)[A-Z][\w'’.-]+(?: [A-Z][\w'’.-]+){1,2}(?= )")


def _persona_text_clean(text: str) -> str:
    return _PERSONA_NAME.sub("This person", text.strip(), count=1)


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


def _writer_body(thinking: bool, reasoning_effort: str | None) -> dict:
    """The vLLM request extras: thinking on / off and, with thinking, Qwen3.8's ``reasoning_effort`` (its chat template
    says the level to the model in the system message; unset = "xhigh": "validate key assumptions, consider plausible
    alternatives"). Sampling not given in the request comes from the model's generation_config (vLLM
    ``--generation-config auto``: top_p 0.95, top_k 20)."""
    body = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if thinking and reasoning_effort:
        assert reasoning_effort in ("low", "medium", "xhigh"), reasoning_effort
        body["reasoning_effort"] = reasoning_effort
    return body


def write_dialogues(seeds: list[dict], url: str, model: str, temperature: float, max_tries: int,
                    workers: int = 64, thinking: bool = False, max_tokens: int = 12288,
                    reasoning_effort: str | None = None, review: int = 0,
                    review_until_clean: bool = False) -> tuple[list[dict], dict]:  # fmt: skip
    """The writer over ``seeds`` against an OpenAI-compatible server at ``url``: rows (``SftWriteDialogues``' columns)
    and stats (kept, dropped, failures per reason with examples, reasoning length). ``thinking``: the model reasons
    before the JSON (``enable_thinking`` per request: our Qwen3.8 server defaults to off; the JSON schema constrains
    only the answer after the reasoning, ``--reasoning-parser qwen3``); the briefs carry many constraints at once
    (cast, user facts, interruptions, language shares) that a non-thinking pass missed often enough to need retries.
    ``review``: up to that many review passes (:data:`REVIEW_TEXT`) over a kept dialogue; a reviewed version replaces
    it when it passes the checks too (else the previous stays); a pass that lists no problem and no unsourced claim
    ends the loop; ``review_until_clean``: a dialogue without such a clean pass is dropped (round 7: one pass took the
    judge from 13 to 19 of 48, the rest still had defects the review had named)."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    from .common import json_schema_format

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url, timeout=1800)
    stats = {"seeds": len(seeds), "kept": 0, "dropped_after_tries": 0, "reasons": {}, "examples": {},
             "thinking": thinking, "reasoning_effort": reasoning_effort, "reasoning_chars": [], "traces": [], "reasons_by_scenario": {},
             "review": {"rounds": review, "until_clean": review_until_clean, "changed": 0, "unchanged": 0,
                        "rejected": {}, "clean_after": {}, "never_clean": 0, "examples": []}}  # fmt: skip

    def reviewed(seed, obj):
        msgs = writer_messages(seed) + [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                                        {"role": "user", "content": REVIEW_TEXT}]  # fmt: skip
        rv = stats["review"]
        try:
            r = client.chat.completions.create(model=model, messages=msgs, temperature=0.3,
                                               max_tokens=max(max_tokens, 10000),
                                               extra_body=_writer_body(False, None),
                                               **json_schema_format(REVIEW_SCHEMA, "review"))  # fmt: skip
            new = json.loads(r.choices[0].message.content)
            problems = new.pop("problems", [])
            claims = new.pop("claims", [])
            new = repair(new, seed)
            why = validate(new, seed)
        except Exception as e:
            new, problems, why = None, [], f"error: {type(e).__name__}"
        if why is not None:
            rv["rejected"][why] = rv["rejected"].get(why, 0) + 1
            return obj, False
        same = json.dumps(new, sort_keys=True) == json.dumps(obj, sort_keys=True)
        rv["unchanged" if same else "changed"] += 1
        if not same and len(rv["examples"]) < 40:
            rv["examples"].append({"id": seed["id"], "claims": claims, "problems": problems, "before": obj,
                                   "after": new})
        return new, not problems and not any(c.get("source") == "none" for c in claims)

    def review_loop(seed, obj):
        """Review rounds until one finds nothing (kept) or ``review`` rounds pass without a clean one (dropped,
        ``review_until_clean``) / the last version (kept otherwise). (Named apart from the flag: as
        ``review_until_clean`` it shadowed the flag, so every unclean dialogue was dropped; p16_gemma8 kept 126/256.)"""
        rv = stats["review"]
        for k in range(review):
            obj, clean = reviewed(seed, obj)
            if clean:
                rv["clean_after"][k + 1] = rv["clean_after"].get(k + 1, 0) + 1
                return obj
        rv["never_clean"] += 1
        return None if review_until_clean else obj

    def one(seed):
        last, msgs = None, writer_messages(seed)
        for k in range(max_tries):
            rc = ""
            try:
                r = client.chat.completions.create(model=model, messages=msgs, temperature=temperature,
                                                   max_tokens=max_tokens, seed=k,
                                                   extra_body=_writer_body(thinking, reasoning_effort),
                                                   **json_schema_format(WRITER_SCHEMA, "dialogue"))  # fmt: skip
                msg = r.choices[0].message
                rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                stats["reasoning_chars"].append(len(rc))
                obj = repair(json.loads(msg.content), seed)
                why = validate(obj, seed)
            except Exception as e:  # malformed output or a request error: retried like a failed check
                obj, why = None, f"error: {type(e).__name__}"
            if why is None:
                if thinking and len(stats["traces"]) < 40:
                    stats["traces"].append({"id": seed["id"], "try": k + 1, "why": None, "reasoning": rc})
                if review:
                    obj = review_loop(seed, obj)
                    if obj is None:
                        return None, "review: never clean"
                return obj, k + 1
            if obj is not None and k + 1 < max_tries:  # the next try sees what was wrong (one round of feedback)
                msgs = writer_messages(seed) + [
                    {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                    {"role": "user", "content": f"This was rejected by the checks: {why}. Write the whole "
                                                f"conversation again, fixing that and keeping every other rule."},
                ]  # fmt: skip
            if thinking and len(stats["traces"]) < 40:  # the reasoning of rejected attempts, to see where it trips
                stats["traces"].append({"id": seed["id"], "try": k + 1, "why": why, "reasoning": rc})
            stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
            by = stats["reasons_by_scenario"].setdefault(seed["scenario"], {})
            by[why] = by.get(why, 0) + 1
            ex = stats["examples"].setdefault(why, [])
            if sum(e["id"].startswith(seed["scenario"] + "_") for e in ex) < 3 and obj is not None:  # 3 per scenario
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
                     "user_facts": json.dumps(obj["user_facts"], ensure_ascii=False),
                     "dialogue": json.dumps(to_tts_dialogue(obj, seed), ensure_ascii=False),
                     "seed": json.dumps(seed, ensure_ascii=False), "tries": tries})  # fmt: skip
    stats["kept"] = len(rows)
    stats["kept_by_scenario"] = {}
    for r in rows:
        stats["kept_by_scenario"][r["scenario"]] = stats["kept_by_scenario"].get(r["scenario"], 0) + 1
    stats["wall_s"] = round(time.time() - t0, 1)
    rcs = sorted(stats.pop("reasoning_chars"))
    stats["reasoning_chars_p50_p90"] = [rcs[len(rcs) // 2], rcs[int(len(rcs) * 0.9)]] if rcs else None
    return rows, stats


FLEURS = {"en": "en_us", "de": "de_de", "zh": "cmn_hans_cn", "ru": "ru_ru"}


def clip_measures(wav, sr: int, text: str, lang: str) -> dict:
    """A voice-prompt candidate's measures: ``expr`` (spread of its pitch, semitones: the median absolute deviation
    of log F0 over voiced frames, scaled to a standard deviation; read speech at a flat pitch makes Chatterbox speak flat), ``rate`` (words, zh characters, per
    second of its sounding span: a fast prompt makes a fast voice), ``snr_db`` (loud frames over quiet frames)."""
    import librosa
    import numpy as np

    from .chatterbox_v2_core import edge_trim, speech_units

    y = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=450, sr=16000, frame_length=1024)
    f = f0[voiced & np.isfinite(f0)]
    st = 12 * np.log2(f / np.median(f)) if len(f) > 20 else np.zeros(1)
    expr = float(1.4826 * np.median(np.abs(st - np.median(st))))  # robust: pitch-tracker octave errors are outliers
    a, b = edge_trim(y, 16000, lead=True, trail=True, margin=0.0)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0] + 1e-9
    snr = float(20 * np.log10(np.percentile(rms, 90) / np.percentile(rms, 10)))
    return {"expr": round(expr, 3), "rate": round(speech_units(text, lang) / max((b - a) / 16000, 0.5), 3),
            "snr_db": round(snr, 1), "f0_median": round(float(np.median(f)), 1) if len(f) else None}


#: Designed voices (no real person cloned): attribute pools for ``voice_descriptions``. Descriptions are English;
#: the language of the speech is set separately (Qwen3-TTS VoiceDesign's ``language``).
VD_TIMBRE = ["warm", "bright", "slightly husky", "smooth", "clear", "resonant", "soft", "crisp", "slightly breathy",
             "rich", "light", "slightly raspy"]  # fmt: skip
VD_ENERGY = ["lively and expressive", "friendly and animated", "calm but expressive", "cheerful", "thoughtful",
             "relaxed and conversational", "energetic", "gentle and warm", "confident", "playful"]  # fmt: skip
VD_PACE = {"assistant": ["at a moderate, steady pace"],
           "user": ["slowly", "at a moderate pace", "at a relaxed pace", "fairly quickly", "quickly"]}  # fmt: skip
VD_PITCH = ["low", "medium-low", "medium", "medium-high", "high"]
#: a light regional colour for some user voices (the assistant speaks the standard variety)
VD_REGION = {
    "en": ["from the north of England", "from Scotland", "from the American South", "from New York", "from California",
           "from Australia", "from Ireland", "from the American Midwest", "from London", "from Canada"],
    "de": ["from Bavaria", "from Hamburg", "from Berlin", "from Saxony", "from the Rhineland", "from Swabia",
           "from Austria", "from Switzerland", "from the Ruhr area", "from Lower Saxony"],
    "zh": ["from Beijing", "from Shanghai", "from Sichuan", "from Guangdong", "from Northeast China", "from Taiwan",
           "from Hunan", "from Shandong", "from Hangzhou", "from Xi'an"],
    "ru": ["from Moscow", "from Saint Petersburg", "from Siberia", "from the Urals", "from the south of Russia",
           "from the Volga region", "from Kazan", "from the Far East", "from Rostov", "from Novosibirsk"],
}  # fmt: skip


def voice_descriptions(lang: str, role: str, gender: str, n: int, seed: int = 0, region: float = 0.0) -> list[str]:
    """``n`` distinct voice descriptions for designed ``gender`` voices of ``role`` speaking ``lang`` natively: age,
    pitch, timbre, energy, pace (the assistant's always moderate and steady), always conversational intonation and a
    clean close-microphone recording (the voice prompt's recording conditions carry into every clone). ``region``:
    the share of user voices with a light regional accent (``VD_REGION``; drawn after the other attributes, so
    ``region=0`` keeps the earlier descriptions)."""
    rng = random.Random(f"vd/{lang}/{role}/{gender}/{seed}")
    ln = LANGS[lang].split(" (")[0]
    out, seen = [], set()
    while len(out) < n:
        age = rng.randint(20, 68) if role == "user" else rng.randint(24, 55)
        pitch, timbre, pace, energy = (rng.choice(VD_PITCH), rng.choice(VD_TIMBRE), rng.choice(VD_PACE[role]),
                                       rng.choice(VD_ENERGY))  # fmt: skip
        reg = ""
        if region and role == "user" and rng.random() < region:
            reg = f" {rng.choice(VD_REGION[lang])}, with a light regional accent"
        d = (f"A {age}-year-old {'woman' if gender == 'female' else 'man'}, a native {ln} speaker{reg}, with a "
             f"{pitch} pitched, {timbre} voice. Speaks {pace} in a {energy} way, with natural, varied conversational "
             f"intonation. Clear close-microphone recording, no background noise, no music.")  # fmt: skip
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


#: Chatterbox Multilingual's own speaker embedding (the voice encoder whose output conditions T3), run in the TTS venv:
#: reads a JSON list of 16 kHz wav paths (argv[1]), writes their embeddings as .npy (argv[2])
_VE_EMBED = """
import json, sys, numpy as np, soundfile as sf, torch, glob
from chatterbox.models.voice_encoder import VoiceEncoder
ck = sorted(glob.glob(sys.argv[3] + "/models--ResembleAI--chatterbox/snapshots/*/ve.pt"))[0]
ve = VoiceEncoder(); ve.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); ve.eval()
wavs = [sf.read(p, dtype="float32")[0] for p in json.load(open(sys.argv[1]))]
np.save(sys.argv[2], np.asarray(ve.embeds_from_wavs(wavs, sample_rate=16000), np.float32))
"""
#: the speaker-gender classifier (wav2vec2 XLS-R fine-tuned on LibriSpeech; agreed with FLEURS's labels on 36 / 36
#: clips, 2026-11-17)
GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"


def select_podcast_voices(pool_dir: str, out: str, tts_python: str, hf_home: str, n_per_role: int = 25,
                          max_sim: float = 0.75, min_snr_db: float = 25.0) -> dict:  # fmt: skip
    """English voice prompts from podcast speakers (``persona_knowledge.KnowledgeVoicePool``: 10 s cloning
    references of (episode, diarized speaker), conversational speech, 16 kHz) into ``<out>/en/<role>/<gender>/``.
    The pool repeats speakers (the host is in nearly every episode), so diversity is enforced in Chatterbox's own
    speaker space: per gender (``GENDER_MODEL``), clean clips (``snr_db >= min_snr_db``) are taken greedily, the most
    expressive first, skipping any clip whose voice-encoder embedding has cosine similarity above ``max_sim`` to one
    already taken (the same speaker), then filling up by farthest-point order; taken voices alternate between
    assistant and user. Levelled to ``PROMPT_DBFS``. Returns the report (also ``<out>/en/podcast_voices.json``)."""
    import io
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import load_from_disk
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    from .chatterbox_v2_core import level_to

    ds = load_from_disk(pool_dir).select_columns(["voice_id", "ref_flac"])
    fe = AutoFeatureExtractor.from_pretrained(GENDER_MODEL)
    gm = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL).eval()
    cands = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for k, r in enumerate(ds):
            wav, sr = sf.read(io.BytesIO(r["ref_flac"]), dtype="float32")
            wav = wav if wav.ndim == 1 else wav.mean(1)
            assert sr == 16000, sr
            mz = clip_measures(wav, sr, "", "en")
            with torch.no_grad():
                pr = gm(**fe(wav, sampling_rate=16000, return_tensors="pt")).logits.softmax(-1)[0]
            lab = {gm.config.id2label[i].lower(): float(pr[i]) for i in range(len(pr))}
            g = max(lab, key=lab.get)
            p = f"{tmp}/{k}.wav"
            sf.write(p, wav, sr)
            paths.append(p)
            cands.append({"voice_id": r["voice_id"], "wav": wav, "gender": g, "p_gender": round(lab[g], 3), **mz})
        json.dump(paths, open(f"{tmp}/paths.json", "w"))
        subprocess.run([tts_python, "-c", _VE_EMBED, f"{tmp}/paths.json", f"{tmp}/emb.npy", f"{hf_home}/hub"],
                       check=True)  # fmt: skip
        emb = np.load(f"{tmp}/emb.npy")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    report = {"candidates": len(cands), "max_sim": max_sim, "voices": []}
    for g in ("female", "male"):
        idx = [i for i, c in enumerate(cands) if c["gender"] == g and c["p_gender"] >= 0.9 and c["snr_db"] >= min_snr_db]
        idx.sort(key=lambda i: -cands[i]["expr"])
        taken: list[int] = []
        for i in idx:  # distinct speakers, most expressive first
            if len(taken) >= 2 * n_per_role:
                break
            if not taken or float(np.max(emb[taken] @ emb[i])) <= max_sim:
                taken.append(i)
        dup = len(idx) - len(taken)
        for role_k, i in enumerate(taken):
            role = "assistant" if role_k % 2 == 0 else "user"
            d = f"{out}/en/{role}/{g}"
            os.makedirs(d, exist_ok=True)
            fn = "podcast_" + re.sub(r"[^\w.-]+", "_", cands[i]["voice_id"]) + ".wav"
            sf.write(f"{d}/{fn}", level_to(cands[i]["wav"], PROMPT_DBFS), 16000)
            report["voices"].append({"role": role, "gender": g, "file": fn, "voice_id": cands[i]["voice_id"],
                                     **{k: v for k, v in cands[i].items() if k not in ("wav", "voice_id", "gender")}})  # fmt: skip
        sims = emb[taken] @ emb[taken].T if taken else np.zeros((0, 0))
        report[g] = {"clean": len(idx), "taken": len(taken), "skipped_as_same_speaker_or_full": dup,
                     "max_pairwise_sim": round(float(np.max(sims - np.eye(len(taken)) * 2)) if len(taken) > 1 else 0, 3),
                     "mean_pairwise_sim": round(float((sims.sum() - len(taken)) / max(1, len(taken) * (len(taken) - 1))), 3)}  # fmt: skip
    os.makedirs(f"{out}/en", exist_ok=True)
    with open(f"{out}/en/podcast_voices.json", "w") as f:
        json.dump(report, f, indent=1)
    return {k: v for k, v in report.items() if k != "voices"}


#: Voice prompts are levelled to this (speech RMS, dBFS): Chatterbox Multilingual copies the prompt's level into its
#: output (pilot 2026-11-17, 42 voices: prompts -52 to -19 dBFS, per-voice output level correlated 0.98 with the
#: prompt's, slope 0.95; 7.2 dB spread between voices against 2.4 dB within one).
PROMPT_DBFS = -23.0


def build_voice_dir(out: str, langs: list[str], n_per_role: int = 10, pool: int = 80, min_snr_db: float = 25.0,
                    seed: int = 0, en_voices: str | None = None) -> dict:  # fmt: skip
    """``<out>/<lang>/<assistant|user>/<female|male>/*.wav`` from FLEURS (``google/fleurs``, CC BY 4.0, real read
    speech with speaker gender; ``dev``, then ``test``, ``train`` while a gender is short): per language and gender up
    to ``pool`` random recordings of 6-12 s are measured (``clip_measures``); those with ``snr_db >= min_snr_db``
    are ranked by pitch spread (the most expressive first) and written levelled to ``PROMPT_DBFS``; the assistant gets ``n_per_role`` from the middle of the
    language's speaking rates (rate between its 30th and 70th percentile, so the assistant's pace is consistent), the
    user ``n_per_role`` of the rest within the 10th-90th percentile. ``<out>/voices.json``: every candidate's measures
    and where it went (network). ``en_voices``: take English from there instead (``select_podcast_voices``' output;
    copied)."""
    import csv
    import io
    import shutil
    import tarfile

    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    from .chatterbox_v2_core import level_to

    report = {}
    if en_voices:
        shutil.copytree(f"{en_voices}/en", f"{out}/en", dirs_exist_ok=True)
        report["en"] = {"source": "podcast", **json.load(open(f"{en_voices}/en/podcast_voices.json"))}
    for lang in langs:
        if lang == "en" and en_voices:
            continue
        fl = FLEURS[lang]
        rng = random.Random(f"{seed}/{lang}")
        cand = {"FEMALE": [], "MALE": []}  # (split, file, wav, sr, measures)
        for split in ("dev", "test", "train"):  # German dev + test hold male speakers only
            short = {g for g in cand if len(cand[g]) < pool}
            if not short:
                break
            meta = {}
            tsv = hf_hub_download("google/fleurs", f"data/{fl}/{split}.tsv", repo_type="dataset")
            for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
                meta.setdefault(r[1], (int(r[5]), r[6].strip().upper(), r[2]))  # file -> (samples, gender, text)
            names = sorted(f for f, (n, g, _) in meta.items() if 6 * 16000 <= n <= 12 * 16000 and g in short)
            rng.shuffle(names)
            want, cnt = set(), {g: len(cand[g]) for g in cand}
            for f in names:
                g = meta[f][1]
                if cnt[g] < pool:
                    want.add(f)
                    cnt[g] += 1
            if not want:
                continue
            tar = hf_hub_download("google/fleurs", f"data/{fl}/audio/{split}.tar.gz", repo_type="dataset")
            with tarfile.open(tar) as tf:
                for m in tf:
                    name = os.path.basename(m.name)
                    if name in want:
                        wav, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                        wav = wav if wav.ndim == 1 else wav.mean(1)
                        cand[meta[name][1]].append((split, name, wav, sr, clip_measures(wav, sr, meta[name][2], lang)))
        rates = np.array([c[4]["rate"] for g in cand for c in cand[g]])
        p10, p30, p70, p90 = np.percentile(rates, [10, 30, 70, 90])
        report[lang] = {"rate_band_assistant": [round(p30, 3), round(p70, 3)], "voices": []}
        for g, cs in cand.items():
            ok = sorted((c for c in cs if c[4]["snr_db"] >= min_snr_db), key=lambda c: -c[4]["expr"])
            asst = [c for c in ok if p30 <= c[4]["rate"] <= p70][:n_per_role]
            user = [c for c in ok if c not in asst and p10 <= c[4]["rate"] <= p90][:n_per_role]
            for role, picked in (("assistant", asst), ("user", user)):
                d = f"{out}/{lang}/{role}/{g.lower()}"
                os.makedirs(d, exist_ok=True)
                for split, name, wav, sr, mz in picked:
                    fn = f"fleurs_{split}_{os.path.splitext(name)[0]}.wav"
                    sf.write(f"{d}/{fn}", level_to(wav, PROMPT_DBFS), sr)
                    report[lang]["voices"].append({"role": role, "gender": g.lower(), "file": fn, **mz})
            report[lang][g.lower()] = {"candidates": len(cs), "clean": len(ok), "assistant": len(asst),
                                       "user": len(user)}  # fmt: skip
    with open(f"{out}/voices.json", "w") as f:
        json.dump(report, f, indent=1)
    return {lg: {k: v for k, v in r.items() if k != "voices"} for lg, r in report.items()}


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
    """``n`` seeds of one scenario for users speaking ``lang`` (``make_seeds``), from index ``start`` (a shard) ->
    ``seeds.jsonl``. ``sources``: the scenario's ``SftSourceRows`` outputs by name (``SCENARIO_SOURCES``). The
    scenario's version is hashed."""

    def __init__(self, *, scenario: str, lang: str, n: int, seed: int = 0, sources: dict | None = None,
                 start: int = 0):  # fmt: skip
        assert scenario in SCENARIOS and lang in LANGS
        need = set(SCENARIO_SOURCES.get(scenario, []))  # personas: the English pool serves every language
        assert need <= set(sources or {}), f"{scenario} needs sources {need}"
        self.scenario, self.lang, self.n, self.seed, self.start = scenario, lang, n, seed, start
        self.sources = {k: v for k, v in (sources or {}).items() if k in need}
        self.scenario_version = SCENARIO_VERSIONS[scenario]
        self.out_seeds = self.output_path("seeds.jsonl")

    @classmethod
    def hash(cls, parsed_args):  # Sisyphus hashes the constructor's arguments only: the version goes in here
        d = {**parsed_args, "scenario_version": SCENARIO_VERSIONS[parsed_args["scenario"]]}
        if d.get("start") == 0:  # absent at 0: the seeds from before sharding keep their hashes
            d.pop("start")
        return super().hash(d)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rows = {k: [json.loads(x) for x in open(v.get(), encoding="utf-8")] for k, v in self.sources.items()}
        with open(self.out_seeds.get(), "w", encoding="utf-8") as f:
            for s in make_seeds(self.scenario, self.lang, self.n, self.seed, rows, start=self.start):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


class SftWriteDialogues(Job):
    """``write_dialogues`` over the seeds of ``seeds`` (one or more ``SftSeeds`` outputs) with the writer on a vLLM
    server. ``out_hf``: ChatterboxInferenceV2's input (``dialogue``) plus ``id, scenario, lang, prompt_lang, policy,
    level, system_prompt, seed, tries``; ``out_stats``: kept / dropped per reason, with examples. ``review``: review
    passes per dialogue (``write_dialogues``); ``vllm_args``: extra server arguments (e.g. ``--quantization fp8
    --kv-cache-dtype fp8``: Gemma 4 31B in 8-bit, judged no worse than bf16 at 2.5x the speed, 2026-11-19);
    ``workers``: requests in flight (not hashed). Seeds are sliced ``seed_slice`` = (k, n): every n-th from k, so one
    seed set can be written by several jobs."""

    def __init__(self, *, seeds: list[tk.Path], model: str = "Qwen/Qwen3.8-27B", temperature: float = 0.8,
                 max_tries: int = 3, thinking: bool = False, reasoning_effort: str | None = None,
                 max_tokens: int = 12288, max_model_len: int | None = None, review: int = 0,
                 review_until_clean: bool = False, vllm_args: list[str] | None = None,
                 seed_slice: tuple[int, int] | None = None, workers: int = 64,
                 rqmt: dict | None = None):  # fmt: skip
        self.seeds, self.model, self.temperature, self.max_tries = seeds, model, temperature, max_tries
        self.thinking, self.reasoning_effort, self.max_tokens = thinking, reasoning_effort, max_tokens
        self.max_model_len = max_model_len
        self.review, self.review_until_clean, self.vllm_args = review, review_until_clean, vllm_args
        self.seed_slice, self.workers = seed_slice, workers
        self.writer_version = WRITER_VERSION
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.rqmt = rqmt or {"gpu": 1, "cpu": 8, "mem": 48, "time": 4}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d.pop("rqmt", None)
        d.pop("workers", None)
        for k, default in (("review", 0), ("review_until_clean", False), ("vllm_args", None), ("seed_slice", None)):
            if d.get(k) == default:  # absent at their defaults: the hashes from before these options stay
                d.pop(k, None)
        d["writer_version"] = WRITER_VERSION  # the constructor's arguments are all Sisyphus hashes by itself
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from datasets import Dataset

        from .common import vllm_server

        seeds = [json.loads(x) for p in self.seeds for x in open(p.get(), encoding="utf-8")]
        if self.seed_slice is not None:
            k, n = self.seed_slice
            seeds = seeds[k::n]
        with vllm_server(self.model, max_model_len=self.max_model_len, extra_args=self.vllm_args) as url:
            rows, stats = write_dialogues(seeds, url, self.model, self.temperature, self.max_tries,
                                          workers=self.workers, thinking=self.thinking, max_tokens=self.max_tokens,
                                          reasoning_effort=self.reasoning_effort, review=self.review,
                                          review_until_clean=self.review_until_clean)  # fmt: skip
        Dataset.from_list(rows).save_to_disk(self.out_hf.get())
        with open(self.out_stats.get(), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"[SftWriteDialogues] kept {len(rows)}/{len(seeds)}; reasons {stats['reasons']}", flush=True)


class SftVoiceDir(Job):
    """``build_voice_dir``: the voice directory ``ChatterboxInferenceV2`` draws ``<lang>/<role>/<gender>/rng_*``
    voices from (every language from FLEURS: the earlier English library had all five assistant clips from one
    "male, British" Parler description)."""

    def __init__(self, *, langs: list[str], n_per_role: int = 10, pool: int = 80, min_snr_db: float = 25.0,
                 seed: int = 0, en_voices: tk.Path | None = None):  # fmt: skip
        self.langs, self.n_per_role, self.pool, self.min_snr_db, self.seed = langs, n_per_role, pool, min_snr_db, seed
        self.en_voices = en_voices
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)  # network

    def run(self):
        print(
            f"[SftVoiceDir] {build_voice_dir(self.out_dir.get(), self.langs, self.n_per_role, self.pool, self.min_snr_db, self.seed, self.en_voices.get() if self.en_voices else None)}"
        )


def voice_design_jobs(texts: dict[str, list[str]], langs: list[str], n_assistant: int, n_user: int, seed: int,
                      region: float) -> list[dict]:
    """The clips ``SftDesignVoices`` designs: per language and gender ``n_assistant`` assistant and ``n_user`` user
    descriptions (``voice_descriptions``), each with a native text from ``texts[lang]`` (in rotation, shuffled)."""
    rng = random.Random(seed)
    out = []
    for lg in langs:
        tx = sorted(set(texts[lg]))
        assert tx, f"no voice-prompt texts for {lg}"
        rng.shuffle(tx)
        k = 0
        for g in ("female", "male"):
            for role, n in (("assistant", n_assistant), ("user", n_user)):
                for d in voice_descriptions(lg, role, g, n, seed=seed, region=region):
                    out.append({"id": f"{lg}_{role}_{g}_{k}", "lang": lg, "role": role, "gender": g, "instruct": d,
                                "text": tx[k % len(tx)], "language": LANGS[lg].split(" (")[0].replace("Mandarin ", "")})
                    k += 1
    return out


def voice_prompt_texts(texts_jsonl: str, langs: list[str]) -> dict[str, list[str]]:
    """The voice-prompt texts per language (``sft_pipeline.SftVoiceTexts``' ``{"lang", "text"}`` rows)."""
    texts = {lg: [] for lg in langs}
    for line in open(texts_jsonl, encoding="utf-8"):
        r = json.loads(line)
        if r["lang"] in texts:
            texts[r["lang"]].append(r["text"])
    return texts


class SftDesignVoices(Job):
    """Designed voices for ``ChatterboxInferenceV2`` with Qwen3-TTS: per language and gender ``n_assistant`` +
    ``n_user`` VoiceDesign descriptions (``voice_descriptions``; ``region``: the share of user voices with a light
    regional accent), each speaking a native text from ``texts`` (``sft_pipeline.SftVoiceTexts``: written for the
    voices, so no dialogue cell feeds the shared voice stage), kept when Qwen3-ASR reads it back
    (``sft_voice_design.py``). ``out_dir``: ``<lang>/<role>/<gender>/vd_<id>.wav`` + ``.txt`` transcript, the
    layout the TTS draws ``rng_*`` voices from. Pilot 2026-11-19: 771 of 800 kept, 8,273 s of audio in 616 s."""

    def __init__(self, *, tts_python: tk.AbstractPath, annotate_python: tk.AbstractPath, texts: tk.Path,
                 langs: list[str], n_assistant: int = 40, n_user: int = 60, seed: int = 1, region: float = 0.5,
                 model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", asr_model: str = "Qwen/Qwen3-ASR-0.6B"):  # fmt: skip
        self.tts_python, self.annotate_python, self.texts = tts_python, annotate_python, texts
        self.langs, self.n_assistant, self.n_user, self.seed, self.region = langs, n_assistant, n_user, seed, region
        self.model, self.asr_model = model, asr_model
        self.out_work = self.output_path("work", directory=True)
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 32, "time": 3})

    def run(self):
        import shutil
        import subprocess

        from huggingface_hub import snapshot_download

        w = self.out_work.get()
        if not os.path.exists(f"{w}/jobs.json"):
            jobs = voice_design_jobs(voice_prompt_texts(self.texts.get(), self.langs), self.langs, self.n_assistant,
                                     self.n_user, self.seed, self.region)  # fmt: skip
            with open(f"{w}/jobs.json", "w", encoding="utf-8") as f:
                json.dump(jobs, f, ensure_ascii=False, indent=1)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_voice_design.py")
        subprocess.run([self.tts_python.get(), script, "gen", w, snapshot_download(self.model)], check=True)
        subprocess.run([self.annotate_python.get(), script, "check", w, self.asr_model], check=True)
        if os.path.isdir(self.out_dir.get()):
            shutil.rmtree(self.out_dir.get())
        shutil.copytree(f"{w}/voices", self.out_dir.get())


class SftPodcastVoices(Job):
    """``select_podcast_voices``: distinct English podcast speakers (``pool``: a ``KnowledgeVoicePool`` dataset) by
    gender, expressiveness and Chatterbox's speaker embedding (``tts_python``: the Chatterbox venv). CPU."""

    def __init__(self, *, pool: tk.Path, tts_python: tk.AbstractPath, n_per_role: int = 25, max_sim: float = 0.75,
                 min_snr_db: float = 25.0):  # fmt: skip
        self.pool, self.tts_python, self.n_per_role, self.max_sim, self.min_snr_db = (
            pool, tts_python, n_per_role, max_sim, min_snr_db)  # fmt: skip
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 8, "mem": 16, "time": 2})

    def run(self):
        hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        rep = select_podcast_voices(self.pool.get(), self.out_dir.get(), self.tts_python.get(), hf, self.n_per_role,
                                    self.max_sim, self.min_snr_db)  # fmt: skip
        print(f"[SftPodcastVoices] {rep}", flush=True)
