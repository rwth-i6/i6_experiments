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
  [5] CAN FAIL: a scenario that does not open with its assigned name, or one beyond the assigned pairs, is rejected
      (``assign_names``); with the check disabled (mutant) the wrong-name scenario gets through
  [6] mix: rows split by weight (largest remainder), a pool too small raises, a weight without a pool raises
  [7] v1 untouched: the declared v1 jobs keep their hashes (``SelfPlayScenarioGen.99kKTwOy4fJc``,
      ``SelfPlayTrainScenarioSet.3HPo2by9EG3X``); v2 is declared (pools per category, one mix) but registers nothing

Run from the setup root (~1 min, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_selfplay_scenario_pools.py
"""

import collections
import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

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
extra = ("topic", "s3", "Your name is Leo. You paint.", "Your name is Zoe. You watch.", "minimal")


def run5(follows):
    orig = sb.follows_names
    sb.follows_names = follows
    try:
        return sb.assign_names([[good, wrong, extra], None], [c0, c0])
    finally:
        sb.follows_names = orig


cands, nm, bad = run5(sb.follows_names)
assert [c[1] for c in cands] == ["s1"] and bad == 2, (cands, bad)
assert nm[(good[2], good[3])] == (("Kofi", "m", "GH"), ("Ama", "f", "GH"))
assert not sb.follows_names("Your name is Kofia. x", "Your name is Ama. y", "Kofi", "Ama"), "prefix match"
mc, _, mbad = run5(lambda *a: True)
assert "s2" in [c[1] for c in mc], "mutant: the wrong-name scenario should get through without the check"
print(f"[5] name check: wrong name and extra scenario rejected ({bad}); mutant without the check keeps the wrong name")

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

# [7] ---------------------------------------------------------------------------------------------------------------
from speech_llm.full_duplex.sis_recipe.doriank import track_persona_rl_selfplay as rs  # noqa: E402

v1 = rs.selfplay_train_scenarios()
assert v1.creator._sis_id().endswith("SelfPlayTrainScenarioSet.3HPo2by9EG3X"), v1.creator._sis_id()
assert v1.creator.scenarios.creator._sis_id().endswith("SelfPlayScenarioGen.99kKTwOy4fJc"), (
    v1.creator.scenarios.creator._sis_id()
)
pools = rs.selfplay_scenario_pools_v2()
assert sorted(pools) == sorted(sb.SPJ2_CATEGORIES) and len({p.creator._sis_id() for p in pools.values()}) == 4
mix = rs.selfplay_train_scenarios_v2().creator
mix2 = rs.selfplay_train_scenarios_v2({"topic": 2, "task": 1, "tension": 1, "instruction": 1}).creator
assert mix._sis_id() != mix2._sis_id(), "a reweighted mix is the same job"
assert {p.creator._sis_id() for p in mix.pools.values()} == {p.creator._sis_id() for p in mix2.pools.values()}, (
    "reweighting moved a pool"
)
print(
    f"[7] v1 hashes unchanged; v2 declared: 4 pools, mix {mix._sis_id().split('/')[-1]}; reweighting re-runs only the mix"
)

print("\nall self-play scenario pool checks passed")
