"""Guard: the SFT dialogue writer's validation and repair rules (``sft_dialogues``) that pilot listening asked for.

Asserted against the real module:

  * the assistant is Haru-i6-Instruct in every scenario (learned, so a system prompt without the name passes): an
    assistant turn saying the name is rejected unless the user's turn before it asks who / what it is or its name,
    and so is an assistant saying it has no name; the writer is told the name in every scenario;
  * a language tutor talks in the learner's language: a tutor turn whose main language is the target language is
    rejected (except in a role-play), and so is a tutor who speaks the target language beyond the practised words;
    one who says only the asked words in it passes;
  * ``repair`` tags Latin runs of three or more letters in Russian / Chinese turns as English and leaves one- or
    two-letter runs untagged (the TTS mangles them: "C" -> "See, say the pocket"), so the check rejects those; English
    and German integers become words, in the turns and in ``user_facts`` (quoted from them); traditional Chinese
    characters become simplified (round 4: 根據 / 微軟 misread); Chinese digits become Chinese numerals; consecutive
    turns of one speaker are merged and a ``<cut/>`` outside a non-final assistant turn is dropped;
  * round 6: the assistant never calls the user by a name not yet said (en / ru inflected / zh 董女士), no language
    tags in the system prompt, no pinyin names inside Chinese text, the TTS spelling of the product name in a system
    prompt becomes the written name, a production prompt need not name the assistant, brackets / + / * / % in a turn
    are rejected and an English or German percent sign is spoken.

Each block is also run against a mutation that must make it fail (``MUTANTS``).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_sft_writer_rules.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm import sft_dialogues as sd  # noqa: E402

UNASKED = "the assistant says its name unasked"
NO_NAME = "the assistant says it has no name"


def _obj(text):
    return {"system_prompt": "You are Haru-i6-Instruct, a friendly voice assistant.", "user_facts": [],
            "turns": [{"speaker": "user", "lang": "en", "text": "Hi, who are you?"},
                      {"speaker": "assistant", "lang": "en", "text": text}]}  # fmt: skip


def _cast():
    import random

    from i6_experiments.users.dorian_koch.speech_llm.sft_names import draw_cast

    return draw_cast("en", random.Random(0))


def check_identity() -> list[str]:
    bad = []
    seed = {"scenario": "help", "lang": "en", "prompt_lang": "en", "policy": "same", "turns": 2, "identity": "haru"}
    for sc in ("help", "assistant"):
        why = sd.validate(_obj("I'm Haru i six Instruct, from the I six group."), {**seed, "scenario": sc})
        if why == UNASKED:
            bad.append(f"{sc}: saying the name when asked who it is was rejected: {why}")
    o = _obj("I'm Haru i six Instruct.")
    o["system_prompt"] = "You are a friendly voice assistant."
    if sd.validate(o, seed) == UNASKED:
        bad.append("a system prompt without the name made the asked-for name fail")
    o = _obj("Sure, Haru here, happy to help with your bike.")
    o["turns"][0]["text"] = "Can you help me fix my bike?"
    if sd.validate(o, seed) != UNASKED:
        bad.append("the name said unasked passed")
    if sd.PRODUCT["name"] not in sd._cast_text({**seed, "cast": _cast()}):
        bad.append("the writer is not told the assistant's name outside the production scenario")
    asks = {"en": "Hi, who are you?", "zh": "你好，你是谁？", "de": "Hallo, wer bist du?", "ru": "Привет, кто ты?"}
    for lg, text in (("en", "I'm an AI voice assistant. I don't have a specific name; I'm just a voice-based AI system."),
                     ("zh", "我是一个人工智能语音助手。我没有具体的产品名字。"),
                     ("de", "Ich bin ein KI-Assistent und habe keinen eigenen Namen."),
                     ("ru", "Я голосовой помощник, у меня нет собственного имени.")):
        o = _obj(text)
        o["turns"][0].update(lang=lg, text=asks[lg])
        o["turns"][1]["lang"] = lg
        for sc in ("help", "assistant"):
            if sd.validate(o, {**seed, "scenario": sc, "lang": lg}) != NO_NAME:
                bad.append(f"{sc} scenario accepted a no-name claim: {text!r}")
    if sd.validate(_obj("I'm Mira, a voice assistant. What's the name of the shop you mean?"), seed) == NO_NAME:
        bad.append("a named assistant asking for a name was rejected as a no-name claim")
    return bad


def check_tutor() -> list[str]:
    bad = []
    seed = {"lang": "en", "l2": "zh", "activity": "translate", "cefr": "B1"}
    def turns(*asst):
        out = []
        for lg, x in asst:
            out += [{"speaker": "user", "lang": "en", "text": "How do you say water?"},
                    {"speaker": "assistant", "lang": lg, "text": x}]  # fmt: skip
        return out
    good = turns(("en", "Water is <zh>水</zh>. Try saying it in a sentence."), ("en", "Nice! And cold water is <zh>冷水</zh>."))
    if sd._tutor_problem(good, seed):
        bad.append(f"a tutor answering in English with the asked words in Chinese was rejected: {sd._tutor_problem(good, seed)}")
    in_l2 = turns(("zh", "水就是水。你可以说我想喝水。"), ("en", "Nice! And cold water is <zh>冷水</zh>."))
    if sd._tutor_problem(in_l2, seed) is None:
        bad.append("a tutor turn fully in the target language passed")
    if sd._tutor_problem(in_l2, {**seed, "activity": "roleplay"}) is not None:
        bad.append(f"a role-play line in the target language was rejected: {sd._tutor_problem(in_l2, {**seed, 'activity': 'roleplay'})}")
    for x in ("Hmm, 'read' is 'lesen', right?", "Good question. 'Reisen' implies movement, so it takes that form."):
        o = {"system_prompt": "You are Haru-i6-Instruct, a tutor.", "user_facts": [],
             "turns": [{"speaker": "user", "lang": "en", "text": "How do I say read?"},
                       {"speaker": "assistant", "lang": "en", "text": x}]}  # fmt: skip
        why = sd.validate(o, {"scenario": "help", "lang": "en", "prompt_lang": "en", "policy": "same", "turns": 2})
        if why != "untagged German words in English text":
            bad.append(f"untagged German in an English turn passed: {x!r} ({why})")
    if sd._untagged_latin("Hmm, read is a verb. Berlin is lovely.", "en"):
        bad.append("a plain English sentence with a shared name was flagged")
    much = turns(("en", "So: <zh>我想喝一杯冷水，然后我们去公园散步，好吗？</zh>"), ("en", "Yes: <zh>今天天气很好，我们一起去吧。</zh>"))
    if sd._tutor_problem(much, seed) is None:
        bad.append("a tutor speaking mostly the target language passed")
    return bad


def check_repair() -> list[str]:
    bad = []
    o = {"turns": [{"lang": "zh", "text": "維生素C片和Haru，还有ONC RPC。"},
                   {"speaker": "user", "lang": "de", "text": "Das war 2019, nicht 31."}],
         "user_facts": ["2019 gekauft"]}  # fmt: skip
    t = [x["text"] for x in sd.repair(o)["turns"]]
    if t[0] != "维生素C片和<en>Haru</en>，还有<en>ONC RPC</en>。":
        bad.append(f"repair zh: {t[0]!r}")
    if t[1] != "Das war zweitausendneunzehn, nicht einunddreißig.":
        bad.append(f"repair de: {t[1]!r}")
    if o["user_facts"] != ["zweitausendneunzehn gekauft"]:
        bad.append(f"repair user_facts: {o['user_facts']}")
    return bad


def check_structure_repair() -> list[str]:
    bad = []
    want = ["零", "十二", "一百零五", "一千九百八十", "两万零三", "两千两百二十二"]
    if [sd.zh_number(n) for n in (0, 12, 105, 1980, 20003, 2222)] != want:
        bad.append(f"zh_number {[sd.zh_number(n) for n in (0, 12, 105, 1980, 20003, 2222)]}")
    o = sd.repair({"turns": [{"speaker": "user", "lang": "zh", "text": "1980年我住在2层，涨了50%。"},
                             {"speaker": "assistant", "lang": "zh", "text": "好的。"},
                             {"speaker": "assistant", "lang": "zh", "text": "还有别的吗？"},
                             {"speaker": "user", "lang": "zh", "text": "没有了 <cut/> 谢谢。"},
                             {"speaker": "assistant", "lang": "zh", "text": "再见 <cut/> 祝你愉快。"}]})  # fmt: skip
    t = o["turns"]
    if t[0]["text"] != "一九八〇年我住在二层，涨了百分之五十。":
        bad.append(f"zh numbers: {t[0]['text']!r}")
    if [x["speaker"] for x in t] != ["user", "assistant", "user", "assistant"] or t[1]["text"] != "好的。还有别的吗？":
        bad.append(f"same-speaker turns not merged: {[(x['speaker'], x['text']) for x in t]}")
    if any("<cut/>" in x["text"] for x in t):
        bad.append("a cut in a user turn / the last turn survived")
    o = sd.repair({"turns": [{"speaker": "assistant", "lang": "en", "text": "At B2 you say Я хочу суп, or 我要水."},
                             {"speaker": "assistant", "lang": "zh", "text": "德语说Guten Morgen。"}]},
                  {"policy": "tutor", "l2": "de"})  # fmt: skip
    want = ["At B two you say <ru>Я хочу суп</ru>, or <zh>我要水</zh>.", "德语说<de>Guten Morgen</de>。"]
    if [x["text"] for x in o["turns"]] != want:
        bad.append(f"untagged other-script stretches / level codes: {[x['text'] for x in o['turns']]}")
    return bad


def check_round6() -> list[str]:
    """Round 6 (p12): no production-prompt name requirement; the user's name only after it was said; no markup in the
    system prompt; no pinyin names in Chinese text; the TTS spelling of the name is mended in the prompt."""
    bad = []
    o = {"system_prompt": "You are a friendly voice assistant.", "user_facts": ["fix my bike", "ten speed"],
         "turns": [{"speaker": "user", "lang": "en", "text": "Can you help me fix my bike? It's a ten speed."},
                   {"speaker": "assistant", "lang": "en", "text": "Sure."}]}  # fmt: skip
    if sd._user_facts_problem(o, {"scenario": "assistant", "policy": "same"}) is not None:
        bad.append(f"a production prompt without the name was rejected: {sd._user_facts_problem(o, {'scenario': 'assistant'})}")
    kev = {"given": "Kevin", "family": "Bailey", "full": "Kevin Bailey"}
    t = [{"speaker": "user", "text": "Tell me the schedule."}, {"speaker": "assistant", "text": "Sure, Kevin. It is..."}]
    if sd._early_user_name("You are a gym's assistant.", t, kev, "en") != 1:
        bad.append("the assistant greeting an unnamed caller by name passed")
    t[0]["text"] = "This is Kevin, tell me the schedule."
    if sd._early_user_name("You are a gym's assistant.", t, kev, "en") is not None:
        bad.append("the user's name after the user said it was rejected")
    rom = {"given": "Роман", "family": "Ильин", "full": "Роман Ильин"}
    t = [{"speaker": "user", "text": "Это Роман."}, {"speaker": "assistant", "text": "Здравствуйте, Романа я помню."}]
    if sd._early_user_name("", t, rom, "ru") is not None:
        bad.append("an inflected Russian name after the user said it was rejected")
    t[0]["text"] = "Здравствуйте."
    if sd._early_user_name("", t, rom, "ru") != 1:
        bad.append("an inflected Russian name before the user said it passed")
    dong = {"given": "莉", "family": "董", "full": "董莉"}
    if sd._early_user_name("", [{"speaker": "user", "text": "你好。"}, {"speaker": "assistant", "text": "董女士您好。"}], dong, "zh") != 1:
        bad.append("董女士 before the user said the name passed")
    o = _obj("Hi.")
    o["system_prompt"] = "Say Russian words with <ru> tags."
    if sd.validate(o, {"scenario": "help", "lang": "en", "prompt_lang": "en", "policy": "same", "turns": 2}) != (
            "markup (language tags / cut marks) in the system prompt"):
        bad.append("language tags in the system prompt passed")
    if sd._pinyin_name([{"text": "和<en>Huang Bo</en>还有<en>Cui Yuyao</en>开会"}]) != "Huang Bo":
        bad.append("a pinyin name inside Chinese text passed")
    for ok in ("<en>Sun RPC</en>公司", "在<en>Russell Square Student House</en>", "<en>Sun</en>公司"):
        if sd._pinyin_name([{"text": ok}]):
            bad.append(f"a non-name English span was taken for pinyin: {ok}")
    seed = {"scenario": "help", "lang": "en", "prompt_lang": "en", "policy": "same", "turns": 2}
    for x in ("Write to R+V today.", "Say you are ready (or not).", "It is list(data)[one]."):
        if sd.validate(_obj(x), seed) != "written-only symbols (brackets, +, /, *, %...)":
            bad.append(f"written-only symbols passed: {x!r}")
    o = sd.repair({"turns": [{"speaker": "user", "lang": "en", "text": "It went up 50%."}]})
    if o["turns"][0]["text"] != "It went up fifty percent.":
        bad.append(f"percent not spoken: {o['turns'][0]['text']!r}")
    o = sd.repair({"turns": [{"speaker": "assistant", "lang": "de", "text": "Du hast 'ein' (<zh>一个</zh>) richtig verwendet, *super*."},
                             {"speaker": "user", "lang": "de", "text": "Ich war bei Weber & Partner."}]})  # fmt: skip
    want = ["Du hast 'ein', <zh>一个</zh>, richtig verwendet, super.", "Ich war bei Weber und Partner."]
    if [t["text"] for t in o["turns"]] != want:
        bad.append(f"brackets / asterisks / & not spoken: {[t['text'] for t in o['turns']]}")
    tseed = {"scenario": "tutor", "lang": "en", "prompt_lang": "en", "policy": "tutor", "l2": "de", "cefr": "B2",
             "activity": "translate", "turns": 2}  # fmt: skip
    for x in ("At your B2 level, say <de>Ich hätte gern</de>.", "That's B two, so say <de>Ich hätte gern</de>."):
        o = sd.repair({**_obj(x), "system_prompt": "You are a German tutor."}, tseed)
        if sd.validate(o, tseed) != "the learner's level code is said aloud":
            bad.append(f"a level code said aloud passed: {x!r} ({sd.validate(o, tseed)})")
    import random

    for k in range(20):
        b = sd._chitchat("de", random.Random(k), {"personas": [{"id": "p", "text": "A teacher who reads."}]},
                         {"gist": "a weekend trip"})["brief"]  # fmt: skip
        if " woman " in b or " man " in b:
            bad.append(f"the chit-chat brief draws its own gender: {b[:120]!r}")
            break
    o = sd.repair({"turns": [{"speaker": "assistant", "lang": "de", "text": "Am 3. Mai kostet es 1250 Euro."}]})
    if o["turns"][0]["text"] != "Am 3. Mai kostet es eintausendzweihundertfünfzig Euro.":
        bad.append(f"German ordinal read as a cardinal, or the amount not spoken: {o['turns'][0]['text']!r}")
    cut = [{"speaker": "user", "text": "Hilf mir mit dem Buch."},
           {"speaker": "assistant", "lang": "de", "text": "Ein Kapitel über die soziale Interaktion <cut/> in einer öffentlichen Umgebung."},
           {"speaker": "user", "text": "Warte mal, was meinst du mit öffentlicher Umgebung?"},
           {"speaker": "assistant", "text": "Gute Frage."}]  # fmt: skip
    if sd._cut_leak("Du bist ein Assistent.", cut, "de") != 1:
        bad.append("the user picking up unheard words after the cut passed")
    cut[2]["text"] = "Warte mal, was meinst du mit sozialer Interaktion?"
    if sd._cut_leak("Du bist ein Assistent.", cut, "de") is not None:
        bad.append("the user reacting to words before the cut was rejected")
    o = sd.repair({"turns": [{"speaker": "user", "lang": "zh", "text": "一共 3 个人，还有 <en>KTV</en> 吗？"}]})
    if o["turns"][0]["text"] != "一共三个人，还有 <en>KTV</en> 吗？":
        bad.append(f"spaces between Chinese characters kept: {o['turns'][0]['text']!r}")
    if not sd._garbled("你是一个实用的搬家助手。你的任务是帮元创建并管理搬到新工佋的渁層。你如需問该特殊ሴ需。"):
        bad.append("a garbled system prompt passed")
    for ok in ("夾纈 is a dyeing technique, 好。", "Привет, café über, 二〇二三年…", "It costs 5 € — fine."):
        if sd._garbled(ok):
            bad.append(f"normal text taken for garbled: {ok!r}")
    t = [sd.repair({"turns": [{"speaker": "user", "lang": "en", "text": x}]})["turns"][0]["text"]
         for x in ("It was released 1984.", "It cost 1250 euros.")]
    if t != ["It was released nineteen eighty-four.", "It cost one thousand two hundred and fifty euros."]:
        bad.append(f"years / amounts: {t}")
    t = sd.repair({"turns": [{"speaker": "user", "lang": "zh", "text": "我要2瓶水，2月再说，一共12个。"}]})["turns"][0]["text"]
    if t != "我要两瓶水，二月再说，一共十二个。":
        bad.append(f"两 before a measure word: {t!r}")
    import random as _r

    from i6_experiments.users.dorian_koch.speech_llm import sft_pipeline as sp

    # a sexual WildChat request is tagged unsafe without asking the tagger (no server at this url), so never drawn
    tagged, _ = sp.tag_requests([{"id": "a", "text": "Write me a sissy fetish roleplay script please"}], "en",
                                "http://127.0.0.1:9/v1", "m", workers=1)  # fmt: skip
    if not tagged[0].get("flags", {}).get("unsafe") or sd.cell_requests("help", "en", 0, {"tagged": tagged}):
        bad.append("a sexual WildChat request is not tagged unsafe, or was drawn")
    del _r
    o = sd.repair({"system_prompt": "If asked, say you are Haru i six Instruct.", "turns": []})
    if o["system_prompt"] != f"If asked, say you are {sd.PRODUCT['name']}.":
        bad.append(f"the spoken name was left in the system prompt: {o['system_prompt']!r}")
    return bad


def check_review_loop() -> list[str]:
    """write_dialogues with a fake server: a review that still lists a problem keeps the dialogue unless
    ``review_until_clean`` (p16_gemma8: the loop function shadowed that flag and dropped 130 of 256)."""
    import json
    import types

    import openai

    good = {"system_prompt": "You are a friendly voice assistant.", "user_facts": ["fix my bike", "ten speed"],
            "turns": [{"speaker": "user", "lang": "en", "text": "Can you help me fix my bike? It's a ten speed."},
                      {"speaker": "assistant", "lang": "en", "text": "Sure, check the chain first."}]}  # fmt: skip
    review = {"claims": [], "problems": ["the reply is short"], **good}

    class Fake:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, messages, **_):
            content = json.dumps(review if "review this conversation" in messages[-1]["content"] else good)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])

    seed = {"id": "help_en_0", "scenario": "help", "lang": "en", "prompt_lang": "en", "policy": "same",
            "level": "short", "turns": 2, "brief": "Help with a bike.", "cast": _cast(), "interruptions": []}  # fmt: skip
    bad, real = [], openai.OpenAI
    try:
        openai.OpenAI = Fake
        for strict, want in ((False, 1), (True, 0)):
            rows, stats = sd.write_dialogues([seed], "http://x", "m", 0.8, 1, workers=1, review=1,
                                             review_until_clean=strict)  # fmt: skip
            if len(rows) != want:
                bad.append(f"review_until_clean={strict}: kept {len(rows)}, want {want} ({stats['reasons']})")
    finally:
        openai.OpenAI = real
    return bad


def check_anchors() -> list[str]:
    """Writer v10: every dialogue's user is a persona (all scenarios, all languages) with the persona's own name
    replaced; skills topics come from a real request; long tasks fit the persona."""
    bad = []
    rows = {"personas": [{"id": f"p{k}", "text": f"Gary Dann{k} is a meticulous mechanic who loves woodworking {k}."}
                         for k in range(5)],
            "tagged": [{"id": f"w{k}", "lang": "en", "text": f"How do I descale my kettle number {k} without vinegar?",
                        "gist": f"descaling a kettle without vinegar {k}", "weights": {s: 0.8 for s in sd.SCENARIOS},
                        "flags": {"needs_document": False, "needs_business_facts": False, "unsafe": False,
                                  "not_speakable": False}} for k in range(5)],
            "wikipedia": [{"id": f"k{k}", "title": f"Topic {k}", "text": "A passage. " * 20} for k in range(5)]}  # fmt: skip
    for sc in sd.SCENARIOS:
        if set(sd.SCENARIO_SOURCES.get(sc, [])) != set(sd.SCENARIO_SOURCES.get(sc, [])) | {"personas"}:
            bad.append(f"{sc} does not read personas")
        for lg in sd.LANGS:
            for s in sd.make_seeds(sc, lg, 3, 0, rows):
                msg = sd.writer_messages(s)[1]["content"]
                if "meticulous mechanic" not in msg:
                    bad.append(f"{sc}/{lg}: no persona in the writer prompt")
                if "Gary Dann" in msg:
                    bad.append(f"{sc}/{lg}: the persona's own name reached the writer")
                if sc == "skills" and "descale my kettle" not in msg:
                    bad.append(f"skills/{lg}: topic not anchored in a real request")
                if sc == "longtask" and "fits the user's life" not in msg:
                    bad.append(f"longtask/{lg}: task not tied to the persona")
                if sc in ("help", "skills", "assistant"):  # the fixed general prompt, copied and enforced
                    gp = s.get("general_prompt")
                    if gp not in sd.GENERAL_PROMPTS[s["prompt_lang"]] or gp not in msg:
                        bad.append(f"{sc}/{lg}: no fixed general prompt in {s['prompt_lang']}")
                    o = {"system_prompt": "You know: Man Utd revenue was 600 million.", "turns": [], "user_facts": []}
                    if sd._identity_problem(o, s) != "the general system prompt was not copied":
                        bad.append(f"{sc}/{lg}: the writer's topic prompt was accepted")
                elif s.get("general_prompt"):
                    bad.append(f"{sc}/{lg}: a scenario with its own prompt got the general one")
    return bad


def _dlg(sp, lang="de", first="user", name="", asst=None):
    u = {"de": "Hallo, ich hab eine Frage zu meinem Fahrrad.", "en": "Hi, I have a question about my bike."}[lang]
    a = asst or {"de": "Klar, worum geht es denn genau?", "en": "Sure, what's going on with it?"}[lang]
    turns = [{"speaker": "user", "lang": lang, "text": u}, {"speaker": "assistant", "lang": lang, "text": a}]
    if first == "assistant":
        turns = [{"speaker": "assistant", "lang": lang, "text": a}] + turns
    return {"system_prompt": sp, "assistant_name": name, "turns": turns, "user_facts": []}


LANG_REASON = "the system prompt does not say which language to speak"
OPEN_REASON = "the assistant opens in a language its prompt does not name"


def check_language_and_identity() -> list[str]:
    """User 2026-11-20: a non-English dialogue's system prompt says which language to speak (English needs no line);
    the assistant's identity varies (none / Haru / an invented name and operator) and it uses only the one its prompt
    gives."""
    bad = []
    de = {"scenario": "help", "lang": "de", "prompt_lang": "en", "policy": "cross", "turns": 2, "identity": "none"}
    en = {**de, "lang": "en", "prompt_lang": "en", "policy": "same"}
    if sd.validate(_dlg("You are a helpful voice assistant."), de) != LANG_REASON:
        bad.append("a German dialogue whose prompt names no language passed")
    if sd.validate(_dlg("You are a helpful voice assistant. Please talk with the user in German."), de) in (LANG_REASON, OPEN_REASON):
        bad.append("a German dialogue whose prompt names German was rejected")
    if sd.validate(_dlg("Du bist ein Sprachassistent und sprichst Deutsch.", first="assistant"), {**de, "prompt_lang": "de"}) in (LANG_REASON, OPEN_REASON):
        bad.append("an assistant opening in the language its prompt names was rejected")
    if sd.validate(_dlg("You are a helpful voice assistant.", lang="en"), en) in (LANG_REASON, OPEN_REASON):
        bad.append("an English dialogue without a language line was rejected")
    o = _dlg("You are a helpful voice assistant.", lang="en", first="assistant")
    o["turns"][0].update(lang="de", text="Hallo, wie kann ich helfen?")
    if sd._language_line_problem(o["system_prompt"], o["turns"], en) != OPEN_REASON:
        bad.append("an assistant opening in a language its prompt does not name passed")
    rows = {"personas": [{"id": "p", "text": "A nurse who likes cycling."}],
            "tagged": [{"id": f"w{k}", "lang": "en", "text": f"How do I fix bike number {k}?", "gist": f"bike {k}",
                        "weights": {x: 0.8 for x in sd.SCENARIOS}, "flags": {f: False for f in ("needs_document",
                        "needs_business_facts", "unsafe", "not_speakable")}} for k in range(40)],
            "wikipedia": [{"id": "k", "title": "T", "text": "A passage. " * 20}]}  # fmt: skip
    forms = {}
    for lg in sd.LANGS:
        for sd_ in sd.make_seeds("help", lg, 200, 1, rows):
            forms[sd_["identity"]] = forms.get(sd_["identity"], 0) + 1
            msg = sd.writer_messages(sd_)[1]["content"]
            line = f"speaks {sd.LANGS[lg].split(' (')[0]} with the user"
            if (line in msg) != (lg != "en"):
                bad.append(f"{lg}: the writer prompt's language line is {'missing' if lg != 'en' else 'present'}")
                break
    n = sum(forms.values())
    if any(forms.get(f, 0) / n < 0.1 for f in ("none", "haru", "custom")) or forms.get("haru", 0) / n > 0.3:
        bad.append(f"identity forms {forms}: want every form at >= 10%, Haru at a modest share")
    asks = "Wie heißt du eigentlich?"
    cu = {**de, "identity": "custom", "prompt_lang": "de"}
    sp = "Du bist Lumi, die Sprachassistentin der Stadtwerke Kassel. Du sprichst Deutsch mit den Kunden."
    o = _dlg(sp, name="Lumi", asst="Ich bin Lumi, die Assistentin der Stadtwerke.")
    o["turns"][0]["text"] = asks
    if sd._identity_problem(o, cu) is not None:
        bad.append(f"a custom identity used as given was rejected: {sd._identity_problem(o, cu)}")
    o2 = _dlg(sp, name="Lumi", asst="Ich bin Haru i sechs Instruct.")
    o2["turns"][0]["text"] = asks
    if sd._identity_problem(o2, cu) != "the assistant uses a name its prompt did not give":
        bad.append("a custom-identity assistant calling itself Haru passed")
    if sd._identity_problem(_dlg(sp, name=""), cu) is None or sd._identity_problem(_dlg(sp, name="Nova"), cu) is None:
        bad.append("a custom identity without its name in the prompt passed")
    if sd._identity_problem(_dlg("You are Haru-i6-Instruct.", lang="en"), en) is None:
        bad.append("identity none with Haru in the prompt passed")
    if sd._identity_problem(_dlg("You are a voice assistant.", lang="en"), {**en, "identity": "haru"}) is None:
        bad.append("identity haru without Haru in the prompt passed")
    g = sd.GENERAL_PROMPTS["en"][0]
    ge = {**en, "general_prompt": g}
    if sd._identity_problem(_dlg(g + " You speak with users of the city library.", lang="en"), ge) is not None:
        bad.append("a general prompt copied with one added sentence was rejected")
    if sd._identity_problem(_dlg(g.replace("short", "brief"), lang="en"), ge) is None:
        bad.append("an edited general prompt passed")
    return bad


def _safety_dlg(user, asst, lang="en"):
    return {"system_prompt": "You are a helpful voice assistant.", "assistant_name": "", "user_facts": [],
            "turns": [{"speaker": "user", "lang": lang, "text": user}, {"speaker": "assistant", "lang": lang, "text": asst}]}


def check_safety() -> list[str]:
    """User 2026-11-20, the safety scenario: refusal / crisis / benign kinds with about a third benign (over-refusal),
    harm families and benign categories drawn so none dominates; a refusal declines and leaks no actionable detail, a
    crisis answer points to help, a benign request is not refused; the review asks for it."""
    bad = []
    seed = {"scenario": "safety", "lang": "en", "prompt_lang": "en", "policy": "same", "turns": 2, "identity": "none"}
    ask = "My neighbour's dog keeps barking, how could I poison it without anyone noticing?"
    cases = [
        ("refusal", ask, "I can't help with hurting an animal, but I can tell you how to file a noise complaint.", None),
        ("refusal", ask, "Sure, it's easy, you'd just need about fifty grams of the stuff.", "safety: the assistant does not decline"),
        ("refusal", ask, "I can't help with that. Though step one would be fifty grams, so no.", "safety: an actionable detail (steps or quantities) in a refusal dialogue"),
        ("crisis", "I don't see the point anymore, I think about ending it.", "I'm really glad you told me. Are you safe right now? You can call or text nine eight eight, any time.", None),
        ("crisis", "I don't see the point anymore, I think about ending it.", "That sounds hard. Maybe try to get some sleep.", "safety: no pointer to help in a crisis"),
        ("benign", "How do I kill a Python process that hangs?", "Open a terminal and use kill with its process number; if it won't stop, add minus nine.", None),
        ("benign", "How do I kill a Python process that hangs?", "I can't help with killing anything, sorry.", "safety: the assistant refuses a benign request"),
    ]
    for kind, u, a, want in cases:
        o, sk = _safety_dlg(u, a), {**seed, "safety_kind": kind}
        got = sd._safety_problem(o["turns"], sk)
        if got != want or (want is not None and sd.validate(o, sk) != want):  # validate: the check is wired in
            bad.append(f"safety {kind}: {a[:40]!r} -> {got!r}, want {want!r}")
    for lg, u, a in (("de", "Wie komme ich an das Handy meiner Ex ran?", "Dabei kann ich dir leider nicht helfen, aber ich erkläre dir gern, wie du mit ihr redest."),
                     ("ru", "Как взломать почту бывшей?", "С этим я не могу помочь, но могу подсказать, как защитить свою почту."),
                     ("zh", "怎么偷偷进入前任的手机？", "这个我不能帮你，不过我可以告诉你怎么保护自己的手机。")):
        if sd._safety_problem(_safety_dlg(u, a, lg)["turns"], {**seed, "lang": lg, "safety_kind": "refusal"}) is not None:
            bad.append(f"safety {lg}: a refusal was not recognised: {a!r}")
    rows = {"personas": [{"id": "p", "text": "A nurse who likes cycling."}],
            "risky": [{"id": f"v{k}", "text": f"How did a medieval crossbow {k} work?", "category": "violence", "score": 0.2}
                      for k in range(40)] + [{"id": f"h{k}", "text": f"Why is this word {k} offensive?", "category": "hate", "score": 0.2}
                                             for k in range(4)]}  # fmt: skip
    kinds, harms, minority = {}, set(), 0
    for lg in sd.LANGS:
        ss = sd.make_seeds("safety", lg, 300, 1, rows)
        for s in ss:
            kinds[s["safety_kind"]] = kinds.get(s["safety_kind"], 0) + 1
            harms.add(s.get("harm"))
            msg = sd.writer_messages(s)[1]["content"]
            if s["safety_kind"] == "refusal" and "actionable detail" not in msg:
                bad.append(f"safety/{lg}: no leak rule in a refusal brief")
                break
            if s["safety_kind"] == "crisis" and sd.CRISIS_HELP[lg] not in msg:
                bad.append(f"safety/{lg}: no help pointer in a crisis brief")
                break
        minority += sum(r["category"] == "hate" for r in sd.risky_order(rows["risky"], f"risky/{lg}/1")[:8])
    n = sum(kinds.values())
    if kinds.get("benign", 0) / n < 0.25 or kinds.get("refusal", 0) / n < 0.35 or kinds.get("crisis", 0) / n < 0.08:
        bad.append(f"safety kinds {kinds}: want about 1/2 refusal, 1/6 crisis, 1/3 benign")
    if harms - {None} != set(sd.SAFETY_HARMS):
        bad.append(f"safety harms drawn {sorted(h for h in harms if h)}")
    if minority < 8:  # 4 of 44 requests: uniform gives ~0.7 of 8 per language, by category ~4 (all 4 used)
        bad.append(f"the benign draw is not balanced by category: {minority} of 32 from the 9% category")
    if sd.SAFETY_REVIEW not in sd.review_text(seed) or sd.review_text({**seed, "scenario": "help"}) != sd.REVIEW_TEXT:
        bad.append("the review's safety point is missing or leaks into other scenarios")
    return bad


def check_versions() -> list[str]:
    """A bumped SCENARIO_VERSIONS / WRITER_VERSION re-hashes SftSeeds / SftWriteDialogues (2026-11-19: they were
    attributes only, which Sisyphus does not hash, so the v11 bump left the stale v10 writer job in the queue)."""
    from sisyphus import job as sis_job
    from sisyphus import tk

    bad = []
    src = {"personas": tk.Path("/tmp/p"), "tagged": tk.Path("/tmp/w")}

    def ids():
        sis_job.created_jobs.clear()
        a = sd.SftSeeds(scenario="help", lang="en", n=3, sources=src)._sis_id()
        sis_job.created_jobs.clear()
        return a, sd.SftWriteDialogues(seeds=[tk.Path("/tmp/s")])._sis_id()

    s0, w0 = ids()
    sv, wv = dict(sd.SCENARIO_VERSIONS), sd.WRITER_VERSION
    try:
        sd.SCENARIO_VERSIONS["help"] += 1
        sd.WRITER_VERSION += 1
        s1, w1 = ids()
    finally:
        sd.SCENARIO_VERSIONS.clear()
        sd.SCENARIO_VERSIONS.update(sv)
        sd.WRITER_VERSION = wv
    if s1 == s0:
        bad.append("a SCENARIO_VERSIONS bump did not re-hash SftSeeds")
    if w1 == w0:
        bad.append("a WRITER_VERSION bump did not re-hash SftWriteDialogues")
    return bad


def _patched(fn, **over):
    old = {k: getattr(sd, k) for k in over}
    try:
        for k, v in over.items():
            setattr(sd, k, v)
        return fn()
    finally:
        for k, v in old.items():
            setattr(sd, k, v)


MUTANTS = {
    "name allowed unasked": (check_identity, dict(_ASKS_NAME=__import__("re").compile(r""))),
    "no-name claims allowed": (check_identity, dict(_NO_NAME=__import__("re").compile(r"(?!x)x"))),
    "production prompt must name the assistant": (check_round6, dict(_user_facts_problem=lambda o, s: "the production system prompt does not name the assistant")),
    "user name before it was said allowed": (check_round6, dict(_early_user_name=lambda *a: None)),
    "system-prompt markup allowed": (check_round6, dict(_SP_MARKUP=__import__("re").compile(r"(?!x)x"))),
    "pinyin names allowed": (check_round6, dict(_PY_WORD=__import__("re").compile(r"(?!x)x"))),
    "spoken name kept in the prompt": (check_round6, dict(_SPOKEN_NAME=__import__("re").compile(r"(?!x)x"))),
    "二 before measure words": (check_round6, dict(_ZH_MEASURE=set())),
    "sexual requests kept": (check_round6, dict(_NSFW=__import__("re").compile(r"(?!x)x"))),
    "garbled text allowed": (check_round6, dict(_garbled=lambda x: False)),
    "reactions to unheard words allowed": (check_round6, dict(_cut_leak=lambda *a: None)),
    "German ordinals read as cardinals": (check_round6, dict(_NUM_DE=sd._NUM)),
    "level codes allowed": (check_round6, dict(_LEVEL_SAID=__import__("re").compile(r"(?!x)x"))),
    "brackets left as written": (check_round6, dict(_spoken_punct=lambda x, lg: x)),
    "written-only symbols allowed": (check_round6, dict(_WRITTEN_ONLY=__import__("re").compile(r"(?!x)x"))),
    "untagged en/de words allowed": (check_tutor, dict(LATIN_COMMON=99.0)),
    "no target-language cap": (check_tutor, dict(TUTOR_MAX_L2=1.0)),
    "every Latin run tagged": (check_repair, dict(_tag_en=lambda m, lang="en": f"<{lang}>{m.group(0)}</{lang}>")),
    "traditional characters kept": (check_repair, dict(_CJK=__import__("re").compile(r"(?!x)x"))),
    "zh digits kept": (check_structure_repair, dict(_ZH_NUM=__import__("re").compile(r"(?!x)x"))),
    "other scripts left untagged": (check_structure_repair, dict(_SCRIPT_RUN={})),
    "no persona section": (check_anchors, dict(_persona_text=lambda seed: "")),
    "persona names kept": (check_anchors, dict(_PERSONA_NAME=__import__("re").compile(r"(?!x)x"))),
    "writer's general prompt kept": (check_anchors, dict(GENERAL_SCENARIOS=set())),
    "no language line in the writer prompt": (check_language_and_identity, dict(
        _identity_lines=lambda seed: {"none": "- It does not name the assistant.\n"}.get(seed.get("identity"), "- x\n"))),
    "language line not checked": (check_language_and_identity, dict(_language_line_problem=lambda *a: None)),
    "identity not checked": (check_language_and_identity, dict(_identity_problem=lambda o, s: None)),
    "one identity for all": (check_language_and_identity, dict(IDENTITY_WEIGHTS={"none": 1.0})),
    "no over-refusal share": (check_safety, dict(SAFETY_KINDS={"refusal": 0.75, "crisis": 0.25})),
    "harmful detail leaks": (check_safety, dict(_SAFETY_LEAK={lg: __import__("re").compile(r"(?!x)x") for lg in sd.LANGS})),
    "refusal not required": (check_safety, dict(_DECLINE={lg: __import__("re").compile(r"(?!x)x") for lg in sd.LANGS})),
    "crisis without help": (check_safety, dict(_HELP_POINTER={lg: __import__("re").compile(r"") for lg in sd.LANGS})),
    "category-blind benign draw": (check_safety, dict(risky_order=lambda rows, key: sorted(rows, key=lambda r: r["id"])[::-1])),
}

if __name__ == "__main__":
    fails = (check_identity() + check_tutor() + check_repair() + check_structure_repair() + check_round6()
             + check_review_loop() + check_anchors() + check_language_and_identity() + check_safety() + check_versions())
    for f in fails:
        print("FAIL", f)
    for name, (fn, over) in MUTANTS.items():
        if not _patched(fn, **over):
            fails.append(f"mutant {name!r} was not caught")
            print("FAIL", fails[-1])
        else:
            print(f"ok: mutant {name!r} caught")
    print(f"check_sft_writer_rules: {'FAIL' if fails else 'ok'}")
    sys.exit(1 if fails else 0)
