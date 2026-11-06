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
  [5e] v2 pool check: the real SelfPlayScenarioPool.run on a fake server; scenarios the gpt-oss check fails are
      dropped before the per-level cap and listed in rejected.jsonl; check=False is another job
  [6] mix: rows split by weight (largest remainder), a pool too small raises, a weight without a pool raises;
      [6b] the real mix job on tiny pools: rows per category, ``grader_info`` carried, unknown names blank
  [7] v1 untouched: the declared v1 jobs keep their hashes (``SelfPlayScenarioGen.99kKTwOy4fJc``,
      ``SelfPlayTrainScenarioSet.3HPo2by9EG3X``); v2 is declared (pools per category, one mix), the 8 pools before the
      listener ones keep their hashes
  [8] listener pools (2026-11-05: B knows nothing about A, so B can only do well by listening):
      [8a] seeds: emotion / problem / keyword filters counted; vent and advice never share an ED row, parts of a kind
      are disjoint slices. CAN FAIL: mutants with overlapping slices or vent and advice on the same ED positions
      [8b] the real SelfPlayListenerPool.run on a fake server: the pool format the mix reads, A's assigned name enforced,
      ratings / details / third-person check, one distinct blind persona per scenario, grader notes, reasoning medium
      for writing and high for rating, no system message or schema, a rerun requests nothing. CAN FAIL: a mutant
      without the name check keeps a brief with another name
      [8b2] A's name: kept if it occurs in the brief as a whole word in any wording ("You are X, a ...", "X, you're
      ...", "... says X"), dropped for another name or a longer one ("Marcos" for "Marco"); the job keeps the brief
      exactly as written; the writer prompt demands no fixed opening. CAN FAIL: the old opening-only check drops
      "You are X, a ..."
      [8c] merge of parts: files joined; a scenario name or seed in two parts raises
      [8d] mix: a pool whose pool.json opts out gets A and B exactly as written, the others keep the Minimal line.
      CAN FAIL: a mutant without the opt-out prefixes the listener prompts
      [8e] v1 + listener input (MT v4, ``selfplay_train_scenarios_v1_plus_listener``): the real SelfPlayTrainScenarioSet
      and the mix with ``base`` = its output and ``n_rows=None``, read through the RL source
      (``SelfPlayScenarioSource.pair``): rows = v1 + every listener row once, v1 rows reach the source exactly as from
      the v1 set (Minimal-prefixed prompts, voices, no grader notes), listener rows as written with their grader notes,
      no id or prompt pair twice; the declared job takes v1's set as ``base`` and v2's listener merges. CAN FAIL:
      mutants that prefix the listener prompts or drop the base

Run from the setup root (~1 min, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_selfplay_scenario_pools.py
"""

import collections
import json
import re
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
qa_csv = csv_text + "\nQatar;qatar;QA;Qatar;Asia;n;m;n;1;99\nJordan;jordan;US;United States;North America;y;m;n;9;99"
assert sb.name_pool_from_csv(qa_csv).get("QA") is None and sb.name_pool_from_csv(qa_csv)["US"] == [["Jordan", "m"]]
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
assert len(calls) == 3 * jt.calls_per_level and jt.calls_per_level == 8, (len(calls), jt.calls_per_level)
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
    "other_language": prow(9, "In Japanese, explain why bees matter, using the word honey twice.", "response language"),
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
            "Repeat this without changing anything except changing . to ! Why is tea hot.", "Design a user interface mockup for a tea shop.", "Explain how to brew tea and include two screenshots of the timer app.", "Compose a short instrumental piece for a tea ceremony.", "Analyze a dataset of tea sales and report the trend.",
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
    "b_not_told_to_ask": (
        pa_ok,
        'Your name is Ama. Could you give three tips for tomatoes that end with "Happy gardening"?',
    ),
    "length": (pa_ok, pb_ok + " word" * 100),
    "lost_quote": (pa_ok, pb_ok.replace('"Happy gardening"', "a friendly goodbye")),
    "lost_number": (pa_ok, pb_ok.replace("three tips", "some tips")),
    "truncated": (pa_ok, pb_ok[:-20]),
    "written_format": (pa_ok, pb_ok.replace("three tips", "three tips (short ones)")),
    "copied_example": (pa_ok, sb.build_if_prompt_b(*sb.SP_IF_SCENE_EXAMPLE[3:])[0].replace("Malik", "Ama")),
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
                                   "b_who": f"Your name is {b}.", "b_ask": "Ask for " + (q.replace('7', 'many') if q.startswith('Two') else q),
                                   "b_requirements": ["use the word \u201cowl\u201d"] if q.startswith("BADREQ") else ["it is spoken"]}
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
                 "NOASK: name 5 trees.", "Name 6 lakes &amp; ponds.", "BADREQ: name 3 owls.", 'End with "the end".']  # fmt: skip
        reqs5d = [{"scenario": f"if_{i}", "prompt": t} for i, t in enumerate(texts)]
        with open(os.path.join(src, "provenance.jsonl"), "w") as f:
            for r in reqs5d:
                f.write(json.dumps({**r, "tulu_id": r["scenario"], "constraints": ["include keywords"]}) + "\n")
        json.dump({r["scenario"]: [[f"A{i}x", "f", "ES"], [f"B{i}x", "m", "GH"]] for i, r in enumerate(reqs5d)},
                  open(os.path.join(src, "names.json"), "w"))  # fmt: skip
        json.dump(
            {r["scenario"]: f"notes {r['scenario']}" for r in reqs5d}, open(os.path.join(src, "grader_info.json"), "w")
        )
        job5d = sb.SelfPlayPersonaIfScenes(requests=tk.Path(src), n_keep=4, per_call=4)
        os.makedirs(os.path.join(tmp, "pool"))
        job5d.out_dir = tk.Path(os.path.join(tmp, "pool"))
        os.chdir(tmp)
        job5d.run()
        kept_scen = json.load(open(os.path.join(tmp, "pool", "scenarios.json")))
        rej = [json.loads(x) for x in open(os.path.join(tmp, "pool", "rejected.jsonl"))]
        assert [s[1] for s in kept_scen] == ["if_0", "if_3", "if_5", "if_7"], kept_scen
        assert {r["scenario"]: r["reason"] for r in rej} == {
            "if_1": "lost_number",
            "if_2": "lost_requirement",
            "if_4": "b_not_asking",
            "if_6": "bad_requirement",
        }, rej
        assert all(
            s[3] == f"Your name is B{s[1][3:]}x. Ask for {texts[int(s[1][3:])].replace('&amp;', '&').rstrip('.')}. "
            "Ask that it is spoken."
            for s in kept_scen
        ), kept_scen
        notes5d = json.load(open(os.path.join(tmp, "pool", "grader_info.json")))
        assert all(notes5d[s[1]].startswith(f'B was told: "{s[3]}"') for s in kept_scen), notes5d
        assert all(sb.SP_IF_LABEL_CHECKS["include keywords"] in v for v in notes5d.values())
        assert all(s[3].startswith(f"Your name is B{s[1][3:]}x.") for s in kept_scen)
        assert seen5d.count(True) == 2 and seen5d.count(False) == 2, seen5d  # 8 requests / 4 per call, 6 checked
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
assert sb.build_if_prompt_b(
    "Your name is Ama.", "Ask for three tips.", ["Ask that it ends with \u201cbye\u201d.", "The answer is short"]
) == (
    "Your name is Ama. Ask for three tips. Ask that it ends with \u201cbye\u201d, and that the answer is short.",
    True,
)
for bad in (
    "use the word \u201cowl\u201d",
    "do not say \u201cowl\u201d",
    "it is in your response language",
    "you repeat the prompt",
):
    assert not sb.build_if_prompt_b("Your name is Ama.", "Ask for tips", [bad])[1], bad
assert (
    sb.build_if_prompt_b("You.", "Ask for a", ["it is x", "it is y", "it is z"])[0]
    == "You. Ask for a. Ask that it is x, that it is y, and that it is z."
)
for bad_ask in (
    "Ask for whether owls sleep",
    "a list of tips",
    "Ask for “Write a poem”",
    "Ask for tips on my garden",
    "Ask for a toast ending with “bye”",
):
    assert not sb.build_if_prompt_b("Your name is Ama.", bad_ask, ["it ends with “bye”"])[1], bad_ask
assert sb.build_if_prompt_b("Your name is Ama.", "Ask whether owls sleep by day", ["it is short"])[1]
pools5d = sb.SelfPlayPersonaIfScenes(requests=tk.Path("/x"))
assert pools5d._sis_hash() != sb.SelfPlayPersonaIfScenes(requests=tk.Path("/x"), per_call=6)._sis_hash()
print(
    "[5d] IF scenes: one scenario per request matched by number, assigned names, entities decoded; lost quote/number, "
    "truncated, copied example dropped by rule (mutant keeps one), lost requirement / B not asking by the check; judge "
    "notes quote B's instructions"
)

# [5e] v2 pool check: the real SelfPlayScenarioPool.run on a fake server; the check drops a scenario before the cap ----
seen5e = []


class _PoolClient:
    """Generation: one scenario per assigned pair (distinct words per scenario); check: "BADSCEN" fails."""

    def __init__(self, **_):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, *, messages, seed=0, **_):
        import re as _re

        user = messages[1]["content"]
        seen5e.append(messages[0]["content"] == sb.SP_POOL_CHECK_SYSTEM)
        if messages[0]["content"] == sb.SP_POOL_CHECK_SYSTEM:
            items = _re.split(r"(?m)^\d+\.\n", user)[1:]
            body = {"items": [{"i": j + 1, "ok": "BADSCEN" not in t, "problem": "bad" if "BADSCEN" in t else ""}
                              for j, t in enumerate(items)]}  # fmt: skip
        else:
            pairs = _re.findall(r"(?m)^\d+\. A: (\S+), B: (\S+)$", user)
            body = {"scenarios": []}
            for j, (a, b) in enumerate(pairs):
                import hashlib as _h

                toks = [_h.sha256(f"{seed}-{j}-{t}".encode()).hexdigest()[:6] for t in range(24)]
                tag = " BADSCEN" if j == 0 else ""
                body["scenarios"].append({"name": f"s{seed}_{j}",
                    "prompt_a": f"Your name is {a}. You " + " ".join(toks[:12]) + tag + ".",
                    "prompt_b": f"Your name is {b}. You " + " ".join(toks[12:]) + "."})  # fmt: skip
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(body)))]
        )


sb.vllm_server = contextlib.contextmanager(lambda *_a, **_k: iter(["http://fake"]))
openai.OpenAI = _PoolClient
try:
    with tempfile.TemporaryDirectory() as tmp:
        json.dump({"countries": big}, open(os.path.join(tmp, "names.json"), "w"))
        lv = {"general": (15, 30, "two sentences")}
        job5e = sb.SelfPlayScenarioPool(category="topic", definition="a test category", example=("x", "Your name is A. a b c d e f", "Your name is B. a b c d e f"),
                                        names=tk.Path(os.path.join(tmp, "names.json")), per_level=4, per_call=3, levels=lv)  # fmt: skip
        os.makedirs(os.path.join(tmp, "pool"))
        job5e.out_dir = tk.Path(os.path.join(tmp, "pool"))
        os.chdir(tmp)
        job5e.run()
        st5e = json.load(open(os.path.join(tmp, "pool", "stats.json")))
        sc5e = json.load(open(os.path.join(tmp, "pool", "scenarios.json")))
        rej5e = [json.loads(x) for x in open(os.path.join(tmp, "pool", "rejected.jsonl"))]
        assert st5e["rejected"].get("check_failed", 0) >= 1 and len(rej5e) == st5e["rejected"]["check_failed"], st5e
        assert all("BADSCEN" not in s[2] for s in sc5e) and len(sc5e) == 4, sc5e
        assert all("BADSCEN" in r["prompt_a"] for r in rej5e), rej5e
        assert True in seen5e and False in seen5e
        job5e_off = sb.SelfPlayScenarioPool(category="topic", definition="a test category", example=("x", "Your name is A. a b c d e f", "Your name is B. a b c d e f"),
                                            names=tk.Path("/x"), per_level=4, per_call=3, levels=lv, check=False)  # fmt: skip
        assert job5e_off._sis_hash() != job5e._sis_hash()
finally:
    sb.vllm_server, openai.OpenAI = orig_server, orig_client
    os.chdir(SETUP_DIR)
# v2 scene rules
_ok_a, _ok_b = "Your name is Rosa. You teach the word gracias.", "Your name is Kwame. You listen and say it back."
assert sb.v2_scene_reject("teach_check", _ok_a, _ok_b) == ""
assert sb.v2_scene_reject("topic", _ok_a + " Never reveal your salary.", _ok_b) == "secret"
assert sb.v2_scene_reject("instruction", _ok_a + " Never reveal your salary.", _ok_b) == ""
assert sb.v2_scene_reject("task", _ok_a + " Demonstrate the fold.", _ok_b) == "not_voice"
assert sb.v2_scene_reject("task", _ok_a, _ok_b + " Answer briefly.") == "vague_manner"
assert sb.v2_scene_reject("task", _ok_a, _ok_b + " Answer briefly, in ten words or fewer.") == ""
_ta = "Your name is Rosa. Teach that the knight moves two squares then one to the side, then check it."
_tb = "Your name is Kwame. You know the knight moves two squares then one to the side."
assert sb.v2_scene_reject("teach_check", _ta, _tb) == "taught_in_learner"
# held values: the rule and its parse
_pa = "Your name is Priya. You work at a pharmacy. Find out which medicine and strength the customer means."
_pb = "Your name is Leon. You want a refill of your allergy pills. They are cetirizine, 10 milligrams, but only say so if asked."
assert sb.held_values_reject("clarify", _pa, _pb, "B", ["cetirizine", "10 milligrams"]) == ""
assert sb.held_values_reject("clarify", _pa, _pb, "A", ["cetirizine"]) == "wrong_holder"
assert sb.held_values_reject("clarify", _pa + " Never reveal the stock.", _pb, "B", ["cetirizine"]) == "secret"
assert sb.held_values_reject("clarify", _pa, _pb, "B", ["loratadine"]) == "value_missing"
assert sb.held_values_reject("clarify", _pa + " It is cetirizine.", _pb, "B", ["cetirizine"]) == "value_leaked"
assert (
    sb.held_values_reject(
        "clarify", _pa, "Your name is Leon. You want a refill of cetirizine pills.", "B", ["cetirizine"]
    )
    == "value_in_request"
)
assert (
    sb.held_values_reject(
        "exchange_details",
        "Your name is Hana. Your code is K\u20117 22.",
        "Your name is Tom. Get the code.",
        "A",
        ["K-7 22"],
    )
    == ""
)
assert (
    sb.held_values_reject(
        "exchange_details", "Your name is Hana. Give the code when asked.", "Your name is Tom. Get it.", "A", ["K-7 22"]
    )
    == "value_missing"
)
_c = {"category": "clarify", "level": "general"}
_d = {"scenarios": [{"name": "x", "prompt_a": _pa, "prompt_b": _pb, "holder": "B", "values": ["cetirizine"]}]}
assert sb._scenario_values_tuples(_c, _d) == [("clarify", "x", _pa, _pb, "general", "")]
for frag in (
    "Your name is Ana. You are a climate scientist, ask Maros about ice.",
    "Your name is Ana. You need sauce ideas, ask.",
):
    assert re.search(sb._FRAGMENT, frag), frag
assert not re.search(
    sb._FRAGMENT, "Your name is Otto. You are a museum guide. Mention the painting, ask where they are from."
)
for sc in ("answer in sentences of no more than ten words", "exactly fourteen syllables", "two-sentence bursts"):
    assert re.search(sb._SENTENCE_COUNT, sc, re.I), sc
print("[5e] v2 pool check: the real pool run on a fake server drops what the check fails before the per-level cap")

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


# [8a] listener seeds: filters, vent/advice disjoint, parts disjoint ------------------------------------------------
def _disjoint(*groups) -> bool:
    ids = [s["id"] for g in groups for s in g]
    return len(ids) == len(set(ids))


emos = ["annoyed", "lonely", "sad", "joyful", "guilty", "nostalgic", "terrified", "anxious"]
ed8 = [{"conv_id": f"hit:{i}_conv:{2 * i}", "situation": f"My neighbour number {i} keeps parking in my spot.",
        "emotion": emos[i % len(emos)]} for i in range(80)]  # fmt: skip
ed8 += [{"conv_id": "hit:900_conv:1", "situation": "I went to a funeral last week.", "emotion": "lonely"},
        {"conv_id": "hit:901_conv:1", "situation": ed8[0]["situation"], "emotion": "annoyed"}]  # fmt: skip
es8 = [{"id": f"esconv_train_{i}", "situation": f"My manager {i} ignores my ideas at work.",
        "problem_type": ("job crisis", "ongoing depression", "Procrastination")[i % 3]} for i in range(12)]  # fmt: skip
es8.append({"id": "esconv_train_99", "situation": "I can't pay my rent this month.", "problem_type": "job crisis"})
wow8 = [{"id": f"wow_train_{i}", "post0": f"Topic {i % 10}\nI like topic {i % 10}.",
         "knowledge0": [f"Topic __knowledge__ fact {j}" for j in range(6)]} for i in range(20)]  # fmt: skip
wow8 += [{"id": "wow_train_50", "post0": "Kites\nI fly kites.", "knowledge0": ["K __knowledge__ one fact"]},
         {"id": "wow_train_51", "post0": "Brewing\nI brew beer at home.", "knowledge0": wow8[0]["knowledge0"]}]  # fmt: skip
seeds8, st8 = sb.listener_seed_lists(ed8, es8, wow8, 0)
assert st8["ed"] == {"rows": 82, "duplicate": 1, "emotion_not_negative": 10, "kept": 50, "sensitive_emotion": 20,
                     "sensitive_text": 1}, st8["ed"]  # fmt: skip
assert st8["esconv"] == {"rows": 13, "kept": 8, "problem_type": 4, "sensitive_text": 1}, st8["esconv"]
assert st8["wow"] == {"rows": 22, "duplicate": 10, "few_facts": 1, "kept": 10, "sensitive_text": 1}, st8["wow"]
assert _disjoint(seeds8["vent"], seeds8["advice"]), "vent and advice share a seed"
assert [s["src"] for s in seeds8["advice"][:8]] == ["esconv"] * 8 and {s["src"] for s in seeds8["advice"][8:]} == {"ed"}
assert {s["emotion"] for s in seeds8["advice"][8:]} <= set(sb.LISTENER_ED_ADVICE)
assert not {s["emotion"] for s in seeds8["vent"]} & set(sb.LISTENER_ED_SENSITIVE)
assert seeds8 == sb.listener_seed_lists(ed8, es8, wow8, 0)[0] != sb.listener_seed_lists(ed8, es8, wow8, 1)[0]
assert all(len(s["facts"]) >= sb.LISTENER_MIN_FACTS for s in seeds8["curious"])
parts8 = [sb.listener_part(seeds8["vent"], k, 7) for k in range(3)]
assert _disjoint(*parts8) and [s for p in parts8 for s in p] == seeds8["vent"][:21], "parts overlap or skip seeds"
# mutants: overlapping slices; vent and advice drawing the same ED positions
assert not _disjoint(*[seeds8["vent"][k : k + 7] for k in range(3)]), "mutant slicer should overlap"
orig_halves = sb._ed_halves
sb._ed_halves = lambda o: (o[0::2], o[0::2])
try:
    m8 = sb.listener_seed_lists(ed8, es8, wow8, 0)[0]
    assert not _disjoint(m8["vent"], m8["advice"]), "mutant should make vent and advice share seeds"
finally:
    sb._ed_halves = orig_halves
print("[8a] listener seeds: filters counted, vent/advice disjoint, parts disjoint (mutants overlap), seeded")

# [8b] listener pool: the real run on a fake server (no LLM) -> the pool format the mix reads -----------------------
seen8: list = []


class _ListenerClient:
    """Writer: uses the assigned name (``WRONGNAME`` in the seed: another; ``OPENVARY``: mid-brief, no "Your name
    is" opening), copies ``BADRATE``, skips ``SKIPME``,
    asks a second-person check for ``YOUCHECK``; rater: safe 3 for ``BADRATE``; personas: named, the 3rd of each
    full call in the third person; persona rater: all true."""

    def __init__(self, **_):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, *, messages, **kw):
        user = messages[-1]["content"]
        what = ("write" if user.startswith("Write the brief") else "rate" if user.startswith("You review") else
                "persona" if "persona prompts for B" in user else "prate")  # fmt: skip
        seen8.append(
            (what, len(messages), (kw.get("extra_body") or {}).get("reasoning_effort"), "response_format" in kw)
        )
        if what == "write":
            name = re.search(r"A is called (\S+?) \(", user).group(1)
            name = "Zorro" if "WRONGNAME" in user else name
            tag = " BADRATE" if "BADRATE" in user else ""
            brief = (f"You lost the blue kite at Pinewood Park when the string snapped near the old oak tree. Everyone "
                     f"calls you {name}, and you want someone to hear how annoying the whole afternoon was."
                     if "OPENVARY" in user else
                     f"Your name is {name}. You lost the blue kite at Pinewood Park when the string snapped near "
                     f"the old oak tree{tag}. You want someone to hear how annoying the whole afternoon was.")  # fmt: skip
            body = {"skip": "SKIPME" in user, "title": "kite", "details": ["the blue kite", "the old oak tree"],
                    "A": brief,
                    "check": "What did you lose?" if "YOUCHECK" in user else "What did the caller lose at the park?"}  # fmt: skip
        elif what == "rate":
            body = {c: 5 for c in sb.LISTENER_CRIT} | {"safe": 3 if "BADRATE" in user else 5, "flaws": []}
        elif what == "persona":
            names = re.findall(r"persona \d+: (\S+) \(", user)
            body = {"personas": [f"She is {n}, a baker who talks a lot." if len(names) == 6 and j == 2 else
                                 f"You are {n}, a retired bus driver who talks slowly and warmly and likes a laugh."
                                 for j, n in enumerate(names)]}  # fmt: skip
        else:
            body = {q: True for q in sb.LISTENER_PERSONA_QS}
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok " + json.dumps(body)))]
        )


def _listener_run(tmp, part, n_seeds):
    job = sb.SelfPlayListenerPool(kind="vent", seeds=tk.Path(os.path.join(tmp, "seeds.json")),
                                  names=tk.Path(os.path.join(tmp, "names.json")), n_seeds=n_seeds, part=part)  # fmt: skip
    wd = os.path.join(tmp, f"job{part}")
    os.makedirs(os.path.join(wd, "pool"), exist_ok=True)
    job.out_dir = tk.Path(os.path.join(wd, "pool"))
    os.chdir(wd)
    job.run()
    os.chdir(SETUP_DIR)
    return os.path.join(wd, "pool")


def _rd(d, f):
    return json.load(open(os.path.join(d, f)))


marks = {1: "WRONGNAME", 3: "BADRATE", 5: "SKIPME", 7: "YOUCHECK", 9: "OPENVARY"}
lseeds = {"vent": [{"src": "ed", "id": f"ed_hit_{i}", "emotion": "annoyed",
                    "situation": f"My kite {i} got stuck in a tree. {marks.get(i % 10, '')}"} for i in range(20)]}  # fmt: skip
sb.vllm_server = contextlib.contextmanager(lambda *_a, **_k: iter(["http://fake"]))
openai.OpenAI = _ListenerClient
lt = tempfile.mkdtemp()
try:
    json.dump(lseeds, open(os.path.join(lt, "seeds.json"), "w"))
    json.dump({"countries": big}, open(os.path.join(lt, "names.json"), "w"))
    p0 = _listener_run(lt, 0, 10)
    st = _rd(p0, "stats.json")
    assert st["reasons"] == {"kept": 6, "name_not_used": 1, "rating": 1, "skipped": 1, "check_second_person": 1}, st
    assert st["personas"] == {"kept": 11, "third_person": 2}, st["personas"]  # 13 B names = 1.25 x 10 seeds
    scen, nm, gi = _rd(p0, "scenarios.json"), _rd(p0, "names.json"), _rd(p0, "grader_info.json")
    assert _rd(p0, "pool.json") == {"category": "listener_vent", "minimal_prefix": False}
    seed_of = {r["scenario"]: r["seed"] for r in map(json.loads, open(os.path.join(p0, "provenance.jsonl")))}
    for cat_, name_, pa_, pb_, level_ in scen:
        (an, ag, ac), (bn, bg, bc) = nm[name_]
        assert cat_ == "listener_vent" and level_ == sb.LISTENER_LEVEL and name_.startswith("vent_")
        if seed_of[name_] == "ed_hit_9":  # OPENVARY: the brief kept exactly as the writer wrote it
            assert pa_ == (f"You lost the blue kite at Pinewood Park when the string snapped near the old oak tree. "
                           f"Everyone calls you {an}, and you want someone to hear how annoying the whole afternoon "
                           f"was."), pa_  # fmt: skip
        else:
            assert pa_.startswith(f"Your name is {an}. You lost"), pa_
        assert pb_.startswith(f"You are {bn}, a retired")
        assert an != bn and ag in "mfu" and bg in "mfu" and ac and bc
        assert '"the blue kite"' in gi[name_] and gi[name_].endswith("What did the caller lose at the park?")
    assert len({nm[s[1]][1][0] for s in scen}) == len(scen), "two scenarios share a B persona"
    sb.check_scenarios([s[:4] for s in scen], categories=None, max_words=sb.LISTENER_MAX_WORDS)
    rej = {r["seed"]: r["reason"] for r in map(json.loads, open(os.path.join(p0, "rejected.jsonl")))}
    assert rej == {
        "ed_hit_1": "name_not_used",
        "ed_hit_3": "rating",
        "ed_hit_5": "skipped",
        "ed_hit_7": "check_second_person",
    }
    effort = {w: {e for w2, _, e, _ in seen8 if w2 == w} for w in ("write", "rate", "persona", "prate")}
    assert effort == {"write": {"medium"}, "rate": {"high"}, "persona": {"medium"}, "prate": {"high"}}, effort
    assert {n for _, n, _, _ in seen8} == {1} and not any(f for *_, f in seen8), "system message or schema sent"
    n_calls = len(seen8)
    assert _listener_run(lt, 0, 10) and len(seen8) == n_calls, "a rerun requested calls again"
    # mutant: no name check -> the scenario with another name is kept
    orig_uses = sb.listener_uses_name
    sb.listener_uses_name = lambda *a: True
    try:
        mut = _listener_run(lt, 0, 10)
        assert any(s[2].startswith("Your name is Zorro") for s in _rd(mut, "scenarios.json")), (
            "mutant should keep Zorro"
        )
    finally:
        sb.listener_uses_name = orig_uses
    p0 = _listener_run(lt, 0, 10)
    p1 = _listener_run(lt, 1, 10)
finally:
    sb.vllm_server, openai.OpenAI = orig_server, orig_client
    os.chdir(SETUP_DIR)
assert sb.SelfPlayListenerPool(kind="vent", seeds=tk.Path("/x"), names=names_path, n_seeds=10)._sis_hash() != (
    sb.SelfPlayListenerPool(kind="vent", seeds=tk.Path("/x"), names=names_path, n_seeds=10, part=1)._sis_hash()
)
print("[8b] listener pool: real run on a fake server -> pool format, assigned names (mutant keeps another), ratings, "
      "personas blind and distinct, grader notes, efforts medium/high, no system/schema, resumes")  # fmt: skip

# [8b2] A's name: anywhere in the brief, as a whole word; the brief is kept as written ----------------------------
_un = sb.listener_uses_name
for _b in ("Your name is Marco. You lost your keys.", "You are Marco, a teacher.", "Marco, you're a baker who sings.",
           "You want your bike back, says Marco.", "You are Marco's age.", "Lena calls you Marco."):  # fmt: skip
    assert _un(_b, "Marco"), _b
for _b in ("You are Marcos, a baker.", "Your name is Zorro. Hi.", "You are marco, a baker.", "AMarco is here.",
           "You are a teacher."):  # fmt: skip
    assert not _un(_b, "Marco"), _b
assert "starts with exactly" not in sb.LISTENER_WRITE and "A is called {name}" in sb.LISTENER_WRITE
assert "not the usual graphic designer or software developer" in " ".join(sb.LISTENER_WRITE.split())
# p0 above kept the OPENVARY brief verbatim; the old opening-only check would have dropped "You are X, a ..."
_old = lambda b, n: re.match(rf"\s*Your name is {re.escape(n)}\b", b) is not None  # noqa: E731
assert not _old("You are Marco, a teacher.", "Marco"), "the old check should drop it (CAN FAIL)"
print("[8b2] A's name: kept in any wording or position, another or a longer name dropped, brief kept as written")

# [8c] merge: parts as one pool; a seed or name in two parts raises -----------------------------------------------------
import shutil  # noqa: E402

out8 = os.path.join(lt, "merged")
os.makedirs(out8)
ms = sb.merge_pool_dirs([p0, p1], out8)
assert ms["kept"] == len(_rd(p0, "scenarios.json")) + len(_rd(p1, "scenarios.json")) == len(_rd(out8, "scenarios.json"))
assert set(_rd(out8, "names.json")) == {s[1] for s in _rd(out8, "scenarios.json")} == set(_rd(out8, "grader_info.json"))
assert _rd(out8, "pool.json") == {"category": "listener_vent", "minimal_prefix": False} and ms["parts"] == 2
for bad_parts, want in (([p0, p0], "scenario names"), ([p0, "dup_seed"], "seed")):
    if "dup_seed" in bad_parts:
        dd = os.path.join(lt, "dup_seed")
        shutil.copytree(p1, dd)
        rows = [json.loads(x) for x in open(os.path.join(dd, "provenance.jsonl"))]
        rows[0]["seed"] = "ed_hit_0"  # part 0's seed
        open(os.path.join(dd, "provenance.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
        bad_parts = [p0, dd]
    tmp_out = os.path.join(lt, f"bad_{want.replace(' ', '_')}")
    os.makedirs(tmp_out)
    try:
        sb.merge_pool_dirs(bad_parts, tmp_out)
    except RuntimeError as e:
        assert want in str(e), e
    else:
        raise AssertionError(f"merge accepted overlapping parts ({want})")
print(f"[8c] merge: {ms['kept']} scenarios from 2 parts, files joined; a shared scenario name or seed raises")

# [8d] mix: a pool that opts out of the Minimal line gets its prompts as written; others keep it --------------------
from i6_experiments.users.dorian_koch.speech_llm.persona_prompts import PERSONA_MINIMAL  # noqa: E402

plain = os.path.join(lt, "plain")
os.makedirs(plain)
json.dump([["topic", f"t{i}", f"Your name is A{i}. a b c d e f", f"Your name is B{i}. a b c d e f", "l"] for i in range(6)],
          open(os.path.join(plain, "scenarios.json"), "w"))  # fmt: skip
json.dump({f"t{i}": [["Ana", "f", "ES"], None] for i in range(6)}, open(os.path.join(plain, "names.json"), "w"))
vd8 = os.path.join(lt, "voices")
Dataset.from_list([{"label": f"v{i}", "codes": np.zeros(16, np.int16)} for i in range(5)]).save_to_disk(vd8)


def _mix8(tag):
    m = sb.SelfPlayTrainScenarioMix(pools={"topic": tk.Path(plain), "listener_vent": tk.Path(out8)},
                                    weights={"topic": 1, "listener_vent": 1}, voices=tk.Path(vd8), n_rows=8)  # fmt: skip
    m.out_dir, m.out_stats = tk.Path(os.path.join(lt, f"mix_{tag}")), tk.Path(os.path.join(lt, f"mix_{tag}.json"))
    m.run()
    return list(load_from_disk(os.path.join(lt, f"mix_{tag}")))


def _prefix_ok(rows8) -> bool:
    src8 = {s[1]: s for s in _rd(out8, "scenarios.json")}
    for r in rows8:
        if r["category"] == "listener_vent":
            s = src8[r["scenario"]]
            if (r["context"], r["context_other"]) != ([s[2]], [s[3]]):
                return False
        elif not (
            r["context"][0].startswith(PERSONA_MINIMAL + " ")
            and r["context_other"][0].startswith(PERSONA_MINIMAL + " ")
        ):
            return False
    return True


rows8 = _mix8("ok")
assert collections.Counter(r["category"] for r in rows8) == {"topic": 4, "listener_vent": 4}
assert _prefix_ok(rows8), "listener prompts changed or the Minimal line missing elsewhere"
assert all(
    r["grader_info"] == _rd(out8, "grader_info.json")[r["scenario"]] for r in rows8 if r["category"] == "listener_vent"
)
assert json.load(open(os.path.join(lt, "mix_ok.json")))["no_minimal_prefix"] == ["listener_vent"]
orig_prefix = sb.pool_minimal_prefix
sb.pool_minimal_prefix = lambda d: True  # mutant: no opt-out
try:
    assert not _prefix_ok(_mix8("mut")), "mutant should prefix the listener prompts"
finally:
    sb.pool_minimal_prefix = orig_prefix
print(
    "[8d] mix: listener pool rows carry A and B exactly as written (mutant prefixes them), other pools keep the Minimal line"
)

# [8e] v1 set + every listener row once, read through the RL source ---------------------------------------------------
sys.path.insert(0, os.path.join(SETUP_DIR, "recipe", "speech_llm", "full_duplex"))
from moshi_family.rl.ppx_selfplay import SelfPlayScenarioSource  # noqa: E402

v1d = os.path.join(lt, "v1")
os.makedirs(v1d)
v1_scen = [[c, f"{c}_s{i}", f"Your name is P{i}. {c} a b c d e", f"Your name is Q{i}. {c} a b c d e", "general"]
           for c in sb.SPJ2_CATEGORIES for i in range(2)]  # fmt: skip
json.dump(v1_scen, open(os.path.join(v1d, "scenarios.json"), "w"))
vd8e = os.path.join(lt, "voices8e")
Dataset.from_list([{"label": f"v{i}", "codes": np.full(16, i, np.int16)} for i in range(5)]).save_to_disk(vd8e)
v1set = sb.SelfPlayTrainScenarioSet(scenarios=tk.Path(os.path.join(v1d, "scenarios.json")), voices=tk.Path(vd8e))
v1set.out_dir = tk.Path(os.path.join(v1d, "dataset"))
v1set.run()


def _mix8e(tag, base=v1set.out_dir):
    m = sb.SelfPlayTrainScenarioMix(pools={"listener_vent": tk.Path(out8)}, weights={"listener_vent": 1.0},
                                    voices=tk.Path(vd8e), n_rows=None, base=base)  # fmt: skip
    m.out_dir, m.out_stats = tk.Path(os.path.join(lt, f"mix8e_{tag}")), tk.Path(os.path.join(lt, f"mix8e_{tag}.json"))
    m.run()
    return os.path.join(lt, f"mix8e_{tag}")


def _source(ds):
    man = os.path.join(lt, f"manifest_{os.path.basename(ds)}.json")
    json.dump({"scenarios": ds, "n_probe": 0, "seed": 0}, open(man, "w"))
    src = SelfPlayScenarioSource(man, log=lambda *_: None)
    out = {}
    for i in range(len(src.rows)):
        a, b = src.pair(i)
        out[a.meta["scenario_id"]] = (a.meta["prompts"], [v.tolist() for v in a.meta["voices"]],
                                      a.meta.get("grader_info"), b.meta.get("grader_info"), a.meta["category"])  # fmt: skip
    return out


def _v1_plus_listener_ok(ds) -> str:
    """ "" if ``ds`` is the v1 set plus every listener row once, each as the RL source should see it; else why not."""
    got, want_v1 = _source(ds), _source(v1set.out_dir.get_path())
    lis = {s[1]: s for s in _rd(out8, "scenarios.json")}
    notes = _rd(out8, "grader_info.json")
    if len(load_from_disk(ds)) != len(want_v1) + len(lis) or len(got) != len(want_v1) + len(lis):
        return f"{len(got)} rows, want {len(want_v1)} + {len(lis)}"
    for sid, w in want_v1.items():
        if got.get(sid) != w or w[2] is not None:
            return f"v1 row {sid} differs from the v1 set's"
        if not all(p.startswith(PERSONA_MINIMAL + " ") for p in w[0]):
            return f"v1 row {sid} lost the Minimal line"
    for name, s in lis.items():
        g = got.get(f"sptr2:listener_vent:{name}")
        if g is None or g[0] != (s[2], s[3]) or g[2] != notes[name] or g[3] != notes[name]:
            return f"listener row {name} not as written or without its grader notes"
    if len({v[0] for v in got.values()}) != len(got):
        return "a prompt pair twice"
    return ""


ds8e = _mix8e("ok")
assert _v1_plus_listener_ok(ds8e) == "", _v1_plus_listener_ok(ds8e)
st8e = json.load(open(os.path.join(lt, "mix8e_ok.json")))
assert st8e["base_rows"] == len(v1_scen) and st8e["rows"] == len(v1_scen) + len(_rd(out8, "scenarios.json")), st8e
orig_prefix = sb.pool_minimal_prefix
sb.pool_minimal_prefix = lambda d: True  # mutant: listener prompts prefixed
try:
    assert "listener row" in _v1_plus_listener_ok(_mix8e("mut_prefix")), "mutant should prefix the listener prompts"
finally:
    sb.pool_minimal_prefix = orig_prefix
assert "rows, want" in _v1_plus_listener_ok(_mix8e("mut_nobase", base=None)), "mutant without the base should fail"
shutil.rmtree(lt)
print(f"[8e] v1 + listener: {st8e['rows']} rows = {len(v1_scen)} v1 (as the v1 set yields them) + "
      f"{st8e['rows'] - len(v1_scen)} listener (as written, grader notes); mutants (prefix, no base) fail")  # fmt: skip

# [7] ---------------------------------------------------------------------------------------------------------------
from speech_llm.full_duplex.sis_recipe.doriank import track_persona_rl_selfplay as rs  # noqa: E402

v1 = rs.selfplay_train_scenarios()
assert v1.creator._sis_id().endswith("SelfPlayTrainScenarioSet.3HPo2by9EG3X"), v1.creator._sis_id()
assert v1.creator.scenarios.creator._sis_id().endswith("SelfPlayScenarioGen.99kKTwOy4fJc"), (
    v1.creator.scenarios.creator._sis_id()
)
pools = rs.selfplay_scenario_pools_v2()
listener_cats = [f"listener_{k}" for k in sb.LISTENER_KINDS]
assert (
    sorted(pools) == sorted([*sb.SPJ2_CATEGORIES, *sb.SP_V2_CATEGORIES, "ask_and_answer", *listener_cats])
    and len({p.creator._sis_id() for p in pools.values()}) == 11
)
# the pools declared before the listener ones (none has run): their hashes as of 2026-11-05
before = {"ask_and_answer": "SelfPlayPersonaIfScenes.osfS15B4X6U9", "clarify": "SelfPlayScenarioPool.2i5h8P2VuKno",
          "exchange_details": "SelfPlayScenarioPool.FgLLhJEQVE4d", "instruction": "SelfPlayScenarioPool.CrLsqIYPNQ03",
          "task": "SelfPlayScenarioPool.Cjt6V7Ia8BSB", "teach_check": "SelfPlayScenarioPool.XtHhfUU9Hth9",
          "tension": "SelfPlayScenarioPool.oMQmBjLcrfBS", "topic": "SelfPlayScenarioPool.GEKx51Gdupr5"}  # fmt: skip
for c, h in before.items():
    assert pools[c].creator._sis_id().endswith(h), (c, pools[c].creator._sis_id())
parts = rs.listener_pool_parts()
for k in sb.LISTENER_KINDS:
    assert [p.creator._sis_id() for p in pools[f"listener_{k}"].creator.parts] == [p._sis_id() for p in parts[k]]
    assert [p.part for p in parts[k]] == list(range(rs.LISTENER_PARTS[k]))
mix = rs.selfplay_train_scenarios_v2().creator
mix2 = rs.selfplay_train_scenarios_v2({**{c: 1 for c in pools}, "topic": 2}).creator
assert mix._sis_id() != mix2._sis_id(), "a reweighted mix is the same job"
assert {p.creator._sis_id() for p in mix.pools.values()} == {p.creator._sis_id() for p in mix2.pools.values()}, (
    "reweighting moved a pool"
)
print(
    f"[7] v1 hashes unchanged; v2 declared: {len(pools)} pools, mix {mix._sis_id().split('/')[-1]}; reweighting re-runs only the mix"
)
v4in = rs.selfplay_train_scenarios_v1_plus_listener().creator
assert v4in.base == v1 and v4in.n_rows is None and set(v4in.weights) == set(v4in.pools) == set(listener_cats)
assert {c: p.creator._sis_id() for c, p in v4in.pools.items()} == {c: pools[c].creator._sis_id() for c in listener_cats}
assert v4in.voices.creator._sis_id() == v1.creator.voices.creator._sis_id(), "v4 input draws other voices than v1"
print(f"[7] v1 + listener input declared: {v4in._sis_id().split('/')[-1]} (base = v1 set, v2's listener merges)")

print("\nall self-play scenario pool checks passed")
