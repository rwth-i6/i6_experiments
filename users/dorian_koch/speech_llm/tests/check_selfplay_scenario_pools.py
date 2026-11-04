"""Guard: self-play training scenarios v2 -- category pools with assigned first names, mixed by weight (2026-11-04).

v1 (``SelfPlayScenarioGen``) let gpt-oss pick the names: 321 first names over 4,080 sides, the top 5 on 26% (Maya alone
8.8%), and one job held every category, so changing a proportion meant regenerating everything. v2 generates one
``SelfPlayScenarioPool`` per category, gives the LLM each scenario's two names (``FirstNamePool``: a pinned
name-census list, uniform country then uniform name) and mixes pools by weight (``SelfPlayTrainScenarioMix``).

  [1] name folding: accents dropped (Jose, Omer), anything not plain Latin rejected (Cyrillic, 'ł', spaces)
  [2] census CSV -> pool: one entry per (country, name), a name listed for both genders is ``u``, non-Latin dropped
  [3] name pairs: seeded, the two sides differ, no name twice in a call, countries spread (uniform country first)
  [4] pool requests: each call states its pairs in order (``SP_POOL_NAMES_RULE``); two categories draw different
      names; per-level call counts; same inputs -> same requests
  [5] names: no scenario is dropped for its names; one that does not use its assigned pair records the names it uses
      (``prompt_name``), else None. CAN FAIL: a mutant that records the unused assignment
  [5b] IF-Chat question lists: ``if_chat_candidate`` rejects text-only rows (code, tables, JSON, NAME_ placeholders,
      links), time-sensitive, long, non-English and single-turn chats; the rewrite request carries B's name, the chat's
      extra system text and the turns (answers cut); A gets its name + a generic answering line; ``skip`` -> nothing
  [5c] generation resumes: a requeued pool job requests only the calls missing from its cache (a cut line ignored);
      an edited request is generated again
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
         "Michał": None, "Сергей": None, "Mary Ann": None, "maya": None, "R2": None, "": None}  # fmt: skip
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
assert len(calls) == 3 * jt.calls_per_level and jt.calls_per_level == 6, (len(calls), jt.calls_per_level)
for c in calls:
    assert len(c["names"]) == 5
    assert sb.SP_POOL_NAMES_RULE in c["user"]
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


# [5b] IF-Chat question lists ---------------------------------------------------------------------------------------
def conv(*users, answer="An answer.", system="You are a helpful and harmless assistant."):
    msgs = [{"role": "system", "content": system}]
    for u in users:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": answer}]
    return {"uuid": "u1", "messages": msgs, "metadata": {"seed_prompt_sha256": "abc", "seed_dataset": "synthetic"}}


kw = dict(min_turns=2, max_turns=5, max_user_chars=600)
ok_conv, why = sb.if_chat_candidate(conv("Give me three tips for sleeping better.", "Make them shorter."), **kw)
assert why == "" and len(ok_conv["turns"]) == 2 and ok_conv["seed_prompt_sha256"] == "abc", (ok_conv, why)
cases5b = {
    "turns": conv("One question only."),
    "long": conv("x " * 400, "ok"),
    "text_only": conv("Fix this:\n```py\nx=1\n```", "thanks"),
    "time_sensitive": conv("What is the latest news on the election?", "and more?"),
    "non_english": conv("Расскажи мне про кошек пожалуйста", "ещё"),
    "empty": conv("", "hi"),
}
for want, c in cases5b.items():
    got = sb.if_chat_candidate(c, **kw)[1]
    assert got == want, (want, got)
for txt in ("Fill in NAME_1 here", "see https://x.org", '{"a": 1}', "| a | b |"):
    assert sb.if_chat_candidate(conv(txt, "ok"), **kw)[1] == "text_only", txt
req = sb.if_chat_request({**ok_conv, "system": "You are a helpful and harmless assistant. Always answer in French."}, "Ama",
                         max_answer_chars=5)  # fmt: skip
assert req.startswith("Name for B: Ama.") and "Always answer in French." in req and "helpful and harmless" not in req
assert "USER 2: Make them shorter." in req and "ASSISTANT 1: An an [...]" in req, req
ifp = sb.SelfPlayIfChatPool(conversations=tk.Path("/nonexistent/c.jsonl"), names=names_path)
icalls = ifp.calls([ok_conv, {**ok_conv, "uuid": "u2"}], big)
for c in icalls:
    ((a, b_),) = c["names"]
    assert c["prompt_a"].startswith(f"Your name is {a[0]}. ") and c["prompt_a"].split(". ", 1)[1] in sb.SP_IF_ANSWERER
    assert c["user"].startswith(f"Name for B: {b_[0]}.")
assert icalls == ifp.calls([ok_conv, {**ok_conv, "uuid": "u2"}], big), "IF requests not deterministic"
assert sb.SelfPlayIfChatPool.parse(icalls[0], {"skip": True, "name": "x", "prompt_b": "", "grader_info": ""}) == []
got = sb.SelfPlayIfChatPool.parse(icalls[0], {"skip": False, "name": "sleep_tips", "prompt_b": " Ask for tips. ",
                                              "grader_info": " Tips: dark room. "})  # fmt: skip
assert got == [("ask_and_answer", "sleep_tips", icalls[0]["prompt_a"], "Ask for tips.", "if_chat", "Tips: dark room.")]
print("[5b] IF-Chat: text-only / time-sensitive / long / non-English rows rejected, requests carry the chat and names")

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
    sorted(pools) == sorted([*sb.SPJ2_CATEGORIES, "ask_and_answer"])
    and len({p.creator._sis_id() for p in pools.values()}) == 5
)
mix = rs.selfplay_train_scenarios_v2().creator
mix2 = rs.selfplay_train_scenarios_v2({"topic": 2, "task": 1, "tension": 1, "instruction": 1}).creator
assert mix._sis_id() != mix2._sis_id(), "a reweighted mix is the same job"
assert {p.creator._sis_id() for p in mix.pools.values()} == {p.creator._sis_id() for p in mix2.pools.values()}, (
    "reweighting moved a pool"
)
print(
    f"[7] v1 hashes unchanged; v2 declared: 5 pools, mix {mix._sis_id().split('/')[-1]}; reweighting re-runs only the mix"
)

print("\nall self-play scenario pool checks passed")
