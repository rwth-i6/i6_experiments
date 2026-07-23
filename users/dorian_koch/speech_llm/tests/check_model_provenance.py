"""Guard: every benchmark tag used in the recipe must have a provenance entry."""
import re, sys
sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
from i6_experiments.users.dorian_koch.speech_llm.fdb import FDB_MODEL_ORIGIN, _origin_map

# _origin_map is fed FDB eval-log labels, i.e. the tags passed to fdb_benchmark_py.
src = open("recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py").read()
fdb_tags = set(re.findall(r'fdb_benchmark_py\((?:[^()]|\([^()]*\))*?tag="([^"]+)"', src))
missing = sorted(fdb_tags - set(FDB_MODEL_ORIGIN))
print("fdb tags in recipe:", sorted(fdb_tags))
assert not missing, f"unmapped FDB tags: {missing}"
print("[ok] all FDB tags mapped;", len(FDB_MODEL_ORIGIN), "entries total")

# the map itself must be well-formed
for tag, val in FDB_MODEL_ORIGIN.items():
    assert isinstance(val, tuple) and len(val) == 2, (tag, val)
    assert val[0] in ("ours", "hf"), (tag, val)
print("[ok] map well-formed")

# and an unknown tag must FAIL rather than silently render unmarked
try:
    _origin_map(["moshi_base", "some_new_model_we_forgot"])
except AssertionError as e:
    print("[ok] unknown tag rejected:", str(e)[:80], "...")
else:
    raise SystemExit("FAIL: unknown tag was silently accepted")
print("\nALL CHECKS PASSED")
