"""Guard: self-play training scenarios v2 -- category pools with assigned first names, mixed by weight (2026-11-04).

v1 (``SelfPlayScenarioGen``) let gpt-oss pick the names: 321 first names over 4,080 sides, the top 5 on 26% (Maya alone
8.8%), and one job held every category, so changing a proportion meant regenerating everything. v2 generates one
``SelfPlayScenarioPool`` per category, gives the LLM each scenario's two names (``FirstNamePool``: a pinned
name-census list, uniform country then uniform name) and mixes pools by weight (``SelfPlayTrainScenarioMix``).

  [1] name folding: accents dropped (Jose, Omer), anything not plain Latin rejected (Cyrillic, 'ł', spaces), and
      initials without a vowel (Ch, Mj)
  [2] census CSV -> pool: one entry per (country, name), a name listed for both genders is ``u``, non-Latin dropped
  [3] name pairs: seeded, the two sides differ, no name twice in a call, countries spread (uniform country first)
  [4] pool requests: each call states its pairs in order (``SP_POOL_NAMES_RULE``); two categories draw different
      names; per-level call counts; same inputs -> same requests
  [5] names: no scenario is dropped for its names; one that does not use its assigned pair records the names it uses
      (``prompt_name``), else None. CAN FAIL: a mutant that records the unused assignment
  [5b] Tulu 3 persona IF requests: ``persona_if_candidate`` keeps a row only if every constraint label is audible
      and the text has no written-only wording, code, links, dates or non-English; ``select`` gives B the request
      verbatim within ``SP_IF_MAX_WORDS``, the judge the request and its labels as hints, other-language answers dropped, drops near-duplicates, is
      seeded. CAN FAIL: a mutant label set with "punctuation:use no comma" keeps a written-only row
  [5c] generation resumes: a requeued pool job requests only the calls missing from its cache (a cut line ignored);
      an edited request is generated again
  [5d] persona IF requests written into scenarios (``SelfPlayPersonaIfScenes``, the v2 generator): the real run on a
      fake server -- one scenario per request in order with the assigned names, a lost quoted phrase or number
      truncation or a copied example dropped by rule, a lost requirement or a B told to answer by the LLM check,
      judge notes from B's instructions. CAN FAIL: a mutant
      without the rule check keeps a scenario that lost a number
  [6] mix: rows split by weight (largest remainder), a pool too small raises, a weight without a pool raises;
      [6b] the real mix job on tiny pools: rows per category, ``grader_info`` carried, unknown names blank
  [7] v1 untouched: the declared v1 jobs keep their hashes (``SelfPlayScenarioGen.99kKTwOy4fJc``,
      ``SelfPlayTrainScenarioSet.3HPo2by9EG3X``); v2 is declared (pools per category, one mix) but registers nothing

Run from the setup root (~1 min, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_selfplay_scenario_pools.py
"""

import collections
import json
import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
SETUP_DIR = os.getcwd()

import numpy as np  # noqa: E402
from sisyphus import tk  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.selfplay_bench as sb  # noqa: E402

# [1] ---------------------------------------------------------------------------------------------------------------
cases = {"José": "Jose", "Ömer": "Omer", "Anne-Marie": "Anne-Marie", "O'Neil": "O'Neil", "Kofi": "Kofi",
         "Michał": None, "Сергей": None, "Mary Ann": None, "maya": None, "R2": None, "": None,
         "Mj": None, "Ch": None, "Li": "Li", "Ty": "Ty"}  # fmt: skip
for raw, want in cases.items():
    assert sb.ascii_first_name(raw) == want, (raw, sb.ascii_first_name(raw), want)
print(f"[1] name folding: {len(cases)} cases")

# [2] ---------------------------------------------------------------------------------------------------------------
csv_text = (
    "Name;Name ASCII;Country code;Country name;Continent;Official;Gender;Unisex;Country Rank;Frequency\n"
    + "\n".join(
        [
            "Andrea;andrea;IT;Italy;Europe;y;m;y;1;10",
            "Andrea;andrea;IT;Italy;Europe;y;f;y;2;9",
            "José;jose;ES;Spain;Europe;y;m;n;1;8",
            "Сергей;sergey;RU;Russia;Europe;n;m;n;1;7",
            "Kofi;kofi;GH;Ghana;Africa;y;m;n;1;6",
            "Ama;ama;GH;Ghana;Africa;y;f;n;2;5",
        ]
    )
)
pool = sb.name_pool_from_csv(csv_text)
assert pool == {"ES": [["Jose", "m"]], "GH": [["Ama", "f"], ["Kofi", "m"]], "IT": [["Andrea", "u"]]}, pool
assert sb.name_pool_from_csv(csv_text, min_frequency=7) == {"ES": [["Jose", "m"]], "IT": [["Andrea", "u"]]}
print("[2] census CSV -> pool: unisex marked, non-Latin dropped, accents folded")

# [3] ---------------------------------------------------------------------------------------------------------------
big = {f"C{i:02d}": [[f"N{i:02d}x{j}", "m" if j % 2 else "f"] for j in range(i % 7 + 3)] for i in range(60)}
p1 = sb.draw_name_pairs(big, 10, np.random.default_rng(0))
p2 = sb.draw_name_pairs(big, 10, np.random.default_rng(0))
assert p1 == p2, "name pairs are not seeded"
flat = [s[0] for pair in p1 for s in pair]
assert len(set(flat)) == len(flat) == 20, "a name repeats within one call"
assert all(a[0] != b[0] for a, b in p1)
countries = collections.Counter(
    s[2] for seed in range(40) for pair in sb.draw_name_pairs(big, 10, np.random.default_rng(seed)) for s in pair
)
assert len(countries) >= 55 and max(countries.values()) < 3 * (800 / 60), countries.most_common(3)
print(f"[3] name pairs: seeded, distinct within a call, {len(countries)}/60 countries over 800 sides")

# [4] ---------------------------------------------------------------------------------------------------------------
names_path = tk.Path("/nonexistent/names.json")


def pool_job(cat, **kw):
    return sb.SelfPlayScenarioPool(category=cat, definition=sb.SP_TRAIN_CATEGORY_TEXT.get(cat, "a test category"),
                                   example=sb.SP_TRAIN_EXAMPLES.get(cat, ("x", "Your name is A. a b c d e f", "Your name is B. a b c d e f")),
                                   names=names_path, **kw)  # fmt: skip


jt = pool_job("topic", per_level=20, per_call=5)
calls = jt.calls(big)
assert len(calls) == 3 * jt.calls_per_level and jt.calls_per_level == 7, (len(calls), jt.calls_per_level)
for c in calls:
    assert len(c["names"]) == 5
    assert sb.SP_POOL_NAMES_RULE in c["user"]
    assert sb.SP_POOL_STYLE_RULE in c["user"]
    for k, (a, b) in enumerate(c["names"]):
        assert f"{k + 1}. A: {a[0]}, B: {b[0]}" in c["user"], c["user"][-300:]
assert calls == pool_job("topic", per_level=20, per_call=5).calls(big), "pool requests are not deterministic"
other = pool_job("task", per_level=20, per_call=5).calls(big)
assert [c["names"] for c in other] != [c["names"] for c in calls], "two categories drew the same names"
assert pool_job("topic")._sis_hash() != pool_job("task")._sis_hash()
assert pool_job("topic")._sis_hash() == pool_job("topic", levels=dict(sb.SP_TRAIN_LEVELS))._sis_hash(), "default levels"
assert pool_job("my_new_category").category == "my_new_category"  # a new category needs no table entry
print(f"[4] pool requests: {len(calls)} calls, names stated in order, deterministic, per-category draws")

# [5] ---------------------------------------------------------------------------------------------------------------
c0 = {"names": [[["Kofi", "m", "GH"], ["Ama", "f", "GH"]], [["Jose", "m", "ES"], ["Andrea", "u", "IT"]]]}
good = ("topic", "s1", "Your name is Kofi. You fix bikes.", "Your name is Ama. Your bike is broken.", "minimal")
wrong = ("topic", "s2", "Your name is Maya. You sell bread.", "Your name is Andrea. You buy bread.", "minimal")
extra = ("topic", "s3", "You paint landscapes.", "Your name is Zoe. You watch.", "minimal")


def run5(follows):
    orig = sb.follows_names
    sb.follows_names = follows
    try:
        return sb.assign_names([[good, wrong, extra], None], [c0, c0])
    finally:
        sb.follows_names = orig


cands, nm, followed = run5(sb.follows_names)
assert [c[1] for c in cands] == ["s1", "s2", "s3"] and followed == 1, (cands, followed)  # nothing dropped
assert nm[(good[2], good[3])] == (("Kofi", "m", "GH"), ("Ama", "f", "GH"))
assert nm[(wrong[2], wrong[3])] == (("Maya", "", ""), ("Andrea", "", "")), nm[(wrong[2], wrong[3])]
assert nm[(extra[2], extra[3])] == (None, ("Zoe", "", "")), nm[(extra[2], extra[3])]
assert not sb.follows_names("Your name is Kofia. x", "Your name is Ama. y", "Kofi", "Ama"), "prefix match"
_, mnm, _ = run5(lambda *a: True)
assert mnm[(wrong[2], wrong[3])][0] == ("Jose", "m", "ES"), "mutant should record the unused assignment"
print("[5] names: every scenario kept; a scenario using other names records the names it uses (mutant: the assignment)")

fp_c = [("topic", "fp1", "Your name is Dita. My toilet won't flush; ask how to fix it now.", "Your name is Josh. You fix toilets and explain plungers.", "minimal"),
        ("topic", "fp2", "Your name is Mira. Your bike chain keeps slipping off; ask what causes it.", "Your name is Tom. You repair bicycles and explain gears.", "minimal")]  # fmt: skip
fp_k, fp_r = sb.filter_train_scenarios(fp_c, [], per_cell=5, require_you=True)
assert [k[1] for k in fp_k] == ["fp2"] and fp_r == {"not_second_person": 1}, (fp_k, fp_r)
assert len(sb.filter_train_scenarios(fp_c, [], per_cell=5)[0]) == 2, "v1 default must keep first-person sides"
sc_c = [
    (
        "instruction",
        "sc1",
        "Your name is Ana. You run a bakery and answer questions about bread.",
        "Your name is Leo. You ask about rye bread and reply in two short sentences.",
        "general",
    )
]
assert sb.filter_train_scenarios(sc_c, [], per_cell=5, reject_sentence_counts=True)[1] == {"sentence_count": 1}
assert len(sb.filter_train_scenarios(sc_c, [], per_cell=5)[0]) == 1
ar_c = [("teach_check", "ar1", "Your name is Carla. You teach Khalid the phrase 'هل تقبلون' and ask him to repeat it.", "Your name is Khalid. You listen and repeat the phrase when asked.", "general"),
        ("teach_check", "fr1", "Your name is Jose. You teach Mia the French word café and ask her to repeat it.", "Your name is Mia. You listen to Jose and repeat the word when asked.", "general")]  # fmt: skip
assert sb.filter_train_scenarios(ar_c, [], per_cell=5, latin_only=True)[1] == {"non_latin_script": 1}  # café is Latin
import re as _re  # noqa: E402

assert not _re.search(sb._SENTENCE_COUNT, sb.SP_V2_INSTRUCTION[1][2], _re.I), sb.SP_V2_INSTRUCTION[1][2]
assert _re.search(sb._SENTENCE_COUNT, sb.SP_TRAIN_EXAMPLES["instruction"][2], _re.I), "v1 example changed"
print("[5] style: v2 pools drop a side not going on as You/Your (here: My); v1's default keeps it")


# [5b] Tulu 3 persona IF requests ------------------------------------------------------------------------------------
def prow(i, prompt, *labels):
    return {"id": f"p{i}", "prompt": prompt, "constraints": list(labels) or ["include keywords"]}


ok_p = "Explain why bees matter for vineyards in no more than 40 words, and include the word grapes."
assert sb.persona_if_candidate(prow(0, ok_p, "length constraints:number of words", "include keywords")) == (
    ok_p,
    "",
)
sent_p = "Explain why bees matter for vineyards in exactly 4 sentences."
assert sb.persona_if_candidate(prow(12, sent_p, "length constraints:number of sentences"))[1] == "written_label"
assert sb.persona_if_candidate(prow(13, sent_p, "include keywords"))[1] == "written_text"  # count in text, unlabelled
cases5b = {
    "written_label": prow(1, ok_p, "include keywords", "punctuation:use no comma"),
    "long": prow(2, "word " * 60),
    "text_only": prow(3, "Fix this:\n```py\nx=1\n```"),
    "written_text": prow(4, "List three tips for sleep. Format: Bullet points."),
    "time_sensitive": prow(5, "Summarize the latest news about solar power."),
    "non_english": prow(6, "Расскажи мне, почему пчёлы важны для виноградников."),
    "other_language": prow(9, "Explain in Japanese why bees matter, using the word honey twice.", "response language"),
}
for want, r in cases5b.items():
    assert sb.persona_if_candidate(r)[1] == want, (want, sb.persona_if_candidate(r))
assert (
    sb.persona_if_candidate(prow(10, "List three upcoming events in Kelmscott within the next 30 days."))[1]
    == "time_sensitive"
)
role_row = prow(11, "What would a lighthouse say about storms? Imagine you are a lighthouse and answer that question.")
assert sb.persona_if_candidate(role_row)[1] == "role_play"
for txt in ("Write exactly 150 words about tea.", "Two paragraphs separated by ^^^^ about tea.", "Write a 300-word story.",
            "Give a detailed plan for a garden.", "Use no commas in a story about tea.",
            "Repeat this question changing ? to !: why is tea hot?", "Create a one-page document about tea.",
            "Repeat this without changing anything except changing . to ! Why is tea hot.", "Design a user interface mockup for a tea shop.", "Explain how to brew tea and include two screenshots of the timer app.", "Compose a short instrumental piece for a tea ceremony.",
            "Develop a user authentication feature for a tea shop website.", "Name brands that worked with this persona."):  # fmt: skip
    assert sb.persona_if_candidate(prow(7, txt))[1] == "written_text", txt
assert sb.persona_if_candidate(prow(8, "In a poem about tea, end with the sentence: Cheers to that."))[1] == ""

topics = ["bees", "tea", "trains", "maps", "owls", "chess", "bread", "rivers", "kites", "clocks", "boats", "mushrooms"]
rows5 = [prow(i, f"Tell me three facts about {t}, and do not use the word {t[:-1]} at all.", "keywords:exclude words")
         for i, t in enumerate(topics)]  # fmt: skip
rows5 += [prow(100, rows5[0]["prompt"]), prow(101, "Describe tea for a child.", "punctuation:use no comma")]
ifp = sb.SelfPlayPersonaIfPool(names=names_path, n_keep=10)
kept5, stats5 = ifp.select(rows5, big)
assert len(kept5) == 10 and stats5["reasons"]["written_label"] == 1, stats5
assert kept5 == ifp.select(rows5, big)[0], "selection not seeded"
for cat_, name_, pa_, pb_, level_, notes_, info_ in kept5:
    a_, b_ = info_["names"]
    assert pa_.startswith(f"Your name is {a_[0]}. ") and pa_.split(". ", 1)[1] in sb.SP_IF_ANSWERER
    assert pb_.startswith(f"Your name is {b_[0]}. ") and pb_.endswith(info_["prompt"]) and len(pb_.split()) <= 70
    assert info_["prompt"] in notes_ and sb.SP_IF_LABEL_CHECKS["keywords:exclude words"] in notes_
    assert cat_ == "ask_and_answer" and level_ == "persona_if" and name_.startswith("if_")
assert len({k[6]["prompt"] for k in kept5}) == 10, "a duplicate request was kept twice"
orig_labels = sb.SP_IF_SPOKEN_LABELS
sb.SP_IF_SPOKEN_LABELS = (*orig_labels, "punctuation:use no comma")
try:
    assert sb.persona_if_candidate(rows5[-1])[1] == "", "mutant should keep the written-only row"
finally:
    sb.SP_IF_SPOKEN_LABELS = orig_labels
print(
    "[5b] persona IF: rows kept by audible labels only (mutant keeps a written one), written wording dropped, "
    "B gets the request verbatim, the judge the request + label hints, seeded, no duplicates"
)

# [5c] generation resumes from its cache ----------------------------------------------------------------------------
import contextlib  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import openai  # noqa: E402

served = []


class _Client:
    def __init__(self, **_):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, *, messages, **_):
        served.append(messages[1]["content"])
        body = json.dumps({"scenarios": [{"name": "n_" + messages[1]["content"][:3], "prompt_a": "Your name is A. a b c",
                                          "prompt_b": "Your name is B. a b c"}]})  # fmt: skip
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=body))])


orig_server, orig_client = sb.vllm_server, openai.OpenAI
sb.vllm_server = contextlib.contextmanager(lambda *_a, **_k: iter(["http://fake"]))
openai.OpenAI = _Client
try:
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        cache = os.path.join(tmp, "calls.jsonl")
        calls5 = [{"user": f"u{i}", "seed": i, "category": "topic", "level": "minimal"} for i in range(6)]
        first = sb.generate_scenario_batches("m", 1.0, calls5, cache_path=cache)
        assert len(served) == 6 and all(first), served
        with open(cache) as f:
            lines = f.readlines()
        with open(cache, "w") as f:  # an interruption: 3 calls finished, the 4th line cut short
            f.writelines(lines[:3] + [lines[3][:10]])
        served.clear()
        again = sb.generate_scenario_batches("m", 1.0, calls5, cache_path=cache)
        assert len(served) == 3 and again == first, (served, again == first)
        served.clear()
        calls5[0] = {**calls5[0], "user": "changed"}
        sb.generate_scenario_batches("m", 1.0, calls5, cache_path=cache)
        assert served == ["changed"], served
finally:
    sb.vllm_server, openai.OpenAI = orig_server, orig_client
    os.chdir(SETUP_DIR)
print("[5c] generation cache: a requeued job requests only the calls it had not finished; an edited request is redone")

# [5d] persona IF requests written into scenarios -------------------------------------------------------------------
assert sb.spoken_numbers("twenty-five words, said twice, in 3 sentences") == {25, 2, 3}
assert sb.spoken_numbers("seventy\u2011five words") == {75}
assert sb.spoken_numbers("choose one of these", min_word=2) == set()
req_ok = 'Give me 3 tips for growing tomatoes and end with the phrase "happy gardening".'
nm_ok = [["Kofi", "m", "GH"], ["Ama", "f", "GH"]]
pa_ok = "Your name is Kofi. You run a small garden centre and help customers with their plants."
pb_ok = 'Your name is Ama. You just planted tomatoes. Ask for three tips and ask that it ends with "Happy gardening".'
cases5d = {
    "": (pa_ok, pb_ok),
    "names": (pa_ok.replace("Kofi", "Kojo"), pb_ok),
    "b_names_a": (pa_ok, pb_ok.replace("Ask for", "Ask Kofi for")),
    "length": (pa_ok, pb_ok + " word" * 60),
    "lost_quote": (pa_ok, pb_ok.replace('"Happy gardening"', "a friendly goodbye")),
    "lost_number": (pa_ok, pb_ok.replace("three tips", "some tips")),
    "truncated": (pa_ok, pb_ok[:-20]),
    "written_format": (pa_ok, pb_ok.replace("three tips", "three tips (short ones)")),
    "copied_example": (pa_ok, sb.SP_IF_SCENE_EXAMPLE[3].replace("Malik", "Ama")),
}
for want, (a_, b_) in cases5d.items():
    assert sb.persona_if_scene_reject(req_ok, a_, b_, nm_ok) == want, (
        want,
        sb.persona_if_scene_reject(req_ok, a_, b_, nm_ok),
    )
assert (
    sb.persona_if_scene_reject("Pick one of: red, blue. Don't use 'very'.", pa_ok, pb_ok + " Avoid 'very'.", nm_ok)
    == ""
)

seen5d = []


class _SceneClient:
    """Generation: B repeats each request (request 2 drops its number), answers in reverse order with the request
    numbers; check: "LOSTREQ" fails keeps_all, "NOASK" fails b_asks."""

    def __init__(self, **_):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, *, messages, **_):
        import re as _re

        user = messages[1]["content"]
        seen5d.append(messages[0]["content"] == sb.SP_IF_VERIFY_SYSTEM)
        if messages[0]["content"] == sb.SP_IF_VERIFY_SYSTEM:
            items = _re.split(r"(?m)^Item \d+\n", user)[1:]
            body = {"items": [{"i": j + 1, "keeps_all": "LOSTREQ" not in t, "b_asks": "NOASK" not in t,
                               "a_neutral": True, "problem": "x"} for j, t in enumerate(items)]}  # fmt: skip
        else:
            reqs = _re.findall(
                r"(?m)^\d+\. (.*)$", user.split("give each its request number.\n", 1)[1].split("\n\n")[0]
            )
            pairs = _re.findall(r"(?m)^\d+\. A: (\S+), B: (\S+)$", user)
            body = {"scenarios": [{"request": j + 1, "name": "x", "prompt_a": f"Your name is {a}. You answer questions about anything.",
                                   "prompt_b": f"Your name is {b}. Please ask the other person: {q.replace('7', 'many') if q.startswith('Two') else q}"}
                                  for j, (q, (a, b)) in enumerate(zip(reqs, pairs))][::-1]}  # fmt: skip
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(body)))]
        )


sb.vllm_server = contextlib.contextmanager(lambda *_a, **_k: iter(["http://fake"]))
openai.OpenAI = _SceneClient
try:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        texts = ["Name 7 birds.", "Two facts on owls, in 7 words.", "LOSTREQ: say why tea is hot.", "List 4 rivers.",
                 "NOASK: name 5 trees.", "Name 6 lakes &amp; ponds.", 'End with "the end".']  # fmt: skip
        reqs5d = [{"scenario": f"if_{i}", "prompt": t} for i, t in enumerate(texts)]
        with open(os.path.join(src, "provenance.jsonl"), "w") as f:
            for r in reqs5d:
                f.write(json.dumps({**r, "tulu_id": r["scenario"], "constraints": ["include keywords"]}) + "\n")
        json.dump({r["scenario"]: [[f"A{i}x", "f", "ES"], [f"B{i}x", "m", "GH"]] for i, r in enumerate(reqs5d)},
                  open(os.path.join(src, "names.json"), "w"))  # fmt: skip
        json.dump(
            {r["scenario"]: f"notes {r['scenario']}" for r in reqs5d}, open(os.path.join(src, "grader_info.json"), "w")
        )
        job5d = sb.SelfPlayPersonaIfScenes(requests=tk.Path(src), n_keep=3, per_call=4)
        os.makedirs(os.path.join(tmp, "pool"))
        job5d.out_dir = tk.Path(os.path.join(tmp, "pool"))
        os.chdir(tmp)
        job5d.run()
        kept_scen = json.load(open(os.path.join(tmp, "pool", "scenarios.json")))
        rej = [json.loads(x) for x in open(os.path.join(tmp, "pool", "rejected.jsonl"))]
        assert [s[1] for s in kept_scen] == ["if_0", "if_3", "if_5"], kept_scen
        assert {r["scenario"]: r["reason"] for r in rej} == {
            "if_1": "lost_number",
            "if_2": "lost_requirement",
            "if_4": "b_not_asking",
        }, rej
        assert all(s[3].endswith(texts[int(s[1][3:])].replace("&amp;", "&")) for s in kept_scen), kept_scen
        notes5d = json.load(open(os.path.join(tmp, "pool", "grader_info.json")))
        assert all(notes5d[s[1]].startswith(f'B was told: "{s[3]}"') for s in kept_scen), notes5d
        assert all(sb.SP_IF_LABEL_CHECKS["include keywords"] in v for v in notes5d.values())
        assert all(s[3].startswith(f"Your name is B{s[1][3:]}x.") for s in kept_scen)
        assert seen5d.count(True) == 2 and seen5d.count(False) == 2, seen5d  # 7 requests / 4 per call, 6 checked
        orig_rej = sb.persona_if_scene_reject
        sb.persona_if_scene_reject = lambda *a: ""  # mutant: no rule check
        try:
            k_m, _, _ = job5d.assemble(
                job5d.read_requests(src),
                {s: (None, s, "a", "b") for s in ["if_1"]},
                {"if_1": ["if_1", True, True, True, ""]},
            )
            assert [s[1] for _, s in k_m] == ["if_1"], "mutant should keep the scenario that lost a number"
        finally:
            sb.persona_if_scene_reject = orig_rej
finally:
    sb.vllm_server, openai.OpenAI = orig_server, orig_client
    os.chdir(SETUP_DIR)
pools5d = sb.SelfPlayPersonaIfScenes(requests=tk.Path("/x"))
assert pools5d._sis_hash() != sb.SelfPlayPersonaIfScenes(requests=tk.Path("/x"), per_call=6)._sis_hash()
print(
    "[5d] IF scenes: one scenario per request matched by number, assigned names, entities decoded; lost quote/number, "
    "truncated, copied example dropped by rule (mutant keeps one), lost requirement / B not asking by the check; judge "
    "notes quote B's instructions"
)

# [6] ---------------------------------------------------------------------------------------------------------------
assert sb.mix_counts({"a": 10, "b": 10, "c": 10}, {"a": 2, "b": 1, "c": 1}, 8) == {"a": 4, "b": 2, "c": 2}
assert sb.mix_counts({"a": 10, "b": 10, "c": 10}, {"a": 1, "b": 1, "c": 1}, 10) == {"a": 4, "b": 3, "c": 3}
assert sb.mix_counts({"a": 10, "b": 10}, {"a": 1, "b": 0}, 5) == {"a": 5}
for bad_args in (({"a": 3, "b": 10}, {"a": 1, "b": 1}, 10), ({"a": 10}, {"a": 1, "z": 1}, 5)):
    try:
        sb.mix_counts(*bad_args)
    except (RuntimeError, AssertionError):
        continue
    raise AssertionError(f"mix_counts accepted {bad_args}")
print("[6] mix: weights -> row counts (largest remainder); too-small pool and unknown weight raise")

# [6b] the mix job on two tiny pools: weights -> rows, grader notes carried, unknown names blank -----------------
import tempfile as _tf  # noqa: E402

from datasets import Dataset, load_from_disk  # noqa: E402

with _tf.TemporaryDirectory() as tmp:
    pools6 = {}
    for cat, notes in (("topic", False), ("ask_and_answer", True)):
        d = os.path.join(tmp, cat)
        os.makedirs(d)
        scen = [[cat, f"{cat}_{i}", f"Your name is A{i}. a b c d e f", f"Your name is B{i}. a b c d e f", "l"]
                for i in range(6)]  # fmt: skip
        json.dump(scen, open(os.path.join(d, "scenarios.json"), "w"))
        json.dump({s_[1]: [["Ana", "f", "ES"], None] for s_ in scen}, open(os.path.join(d, "names.json"), "w"))
        if notes:
            json.dump({s_[1]: f"facts {s_[1]}" for s_ in scen}, open(os.path.join(d, "grader_info.json"), "w"))
        pools6[cat] = tk.Path(d)
    vd = os.path.join(tmp, "voices")
    Dataset.from_list([{"label": f"v{i}", "codes": np.zeros(16, np.int16)} for i in range(5)]).save_to_disk(vd)
    mix6 = sb.SelfPlayTrainScenarioMix(pools=pools6, weights={"topic": 2, "ask_and_answer": 1}, voices=tk.Path(vd),
                                       n_rows=6)  # fmt: skip
    mix6.out_dir, mix6.out_stats = tk.Path(os.path.join(tmp, "out")), tk.Path(os.path.join(tmp, "stats.json"))
    mix6.run()
    rows6 = list(load_from_disk(os.path.join(tmp, "out")))
    per = collections.Counter(r["category"] for r in rows6)
    assert per == {"topic": 4, "ask_and_answer": 2}, per
    assert all(r["grader_info"] == f"facts {r['scenario']}" for r in rows6 if r["category"] == "ask_and_answer")
    assert all(r["grader_info"] == "" for r in rows6 if r["category"] == "topic")
    assert all(r["name"] == "Ana" and r["name_other"] == "" and r["gender_other"] == "" for r in rows6)
print("[6b] mix job: weights -> rows per category, grader notes carried (blank elsewhere), unknown names blank")

# [7] ---------------------------------------------------------------------------------------------------------------
from speech_llm.full_duplex.sis_recipe.doriank import track_persona_rl_selfplay as rs  # noqa: E402

v1 = rs.selfplay_train_scenarios()
assert v1.creator._sis_id().endswith("SelfPlayTrainScenarioSet.3HPo2by9EG3X"), v1.creator._sis_id()
assert v1.creator.scenarios.creator._sis_id().endswith("SelfPlayScenarioGen.99kKTwOy4fJc"), (
    v1.creator.scenarios.creator._sis_id()
)
pools = rs.selfplay_scenario_pools_v2()
assert (
    sorted(pools) == sorted([*sb.SPJ2_CATEGORIES, *sb.SP_V2_CATEGORIES, "ask_and_answer"])
    and len({p.creator._sis_id() for p in pools.values()}) == 8
)
mix = rs.selfplay_train_scenarios_v2().creator
mix2 = rs.selfplay_train_scenarios_v2({**{c: 1 for c in pools}, "topic": 2}).creator
assert mix._sis_id() != mix2._sis_id(), "a reweighted mix is the same job"
assert {p.creator._sis_id() for p in mix.pools.values()} == {p.creator._sis_id() for p in mix2.pools.values()}, (
    "reweighting moved a pool"
)
print(
    f"[7] v1 hashes unchanged; v2 declared: {len(pools)} pools, mix {mix._sis_id().split('/')[-1]}; reweighting re-runs only the mix"
)

print("\nall self-play scenario pool checks passed")
