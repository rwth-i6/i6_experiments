"""Guard: the self-play judge v2 benchmark (``selfplay_bench``) and the verbatim-copy statistics (``selfplay_copy``),
2026-09-28.

WHY. A benchmark that silently changes its prompts, reuses a voice, parses a malformed judge reply as a score, or
nests its bootstrap wrongly still produces a plausible number. Each item drives the real code:

  1. SCENARIO SET. ``spj2_v1`` = 48 scenarios, 12 per category, digest PINNED here (an edit must be a new version);
     ``scenario_rows`` is deterministic (two calls byte-identical), replicate-major, every scenario's K rows share
     prompts and voices, no voice is used twice, the Minimal line leads every prompt. CAN FAIL: another seed moves
     the voice assignment (so the determinism check is not vacuous); too few voices raises.
  2. HASH. ``SelfPlayScenarioSet`` / ``SelfPlayJudgeV2`` hashes move when a scenario / the rubric TEXT changes (plain
     ``Job.hash`` would not: the digest attribute is what carries it).
  3. JOB -> WORKER. ``SelfPlayScenarioSet.run`` on 96 fake voices writes 192 rows the real self-play worker reads:
     ``selfplay._side_prompt(row, side, "spj2")`` gives each side its own prompt, voice codes round-trip.
  4. PARSER. ``parse_judge_reply`` accepts a valid reply (with reasoning text around it) and caps reasons; rejects a
     missing metric, float / bool / string / out-of-range scores and empty reasons. CAN FAIL: the naive
     ``int(d[m]["score"])`` parser accepts 4 of those 5 malformed replies.
  5. MESSAGE. every placeholder filled; braces inside a transcript survive verbatim (``str.format`` would raise).
  6. STATS. nested bootstrap: identical prompts -> zero-width CI; seed SD / prompt SD recovered on a constructed
     case; paired delta of a +1 shift == 1 with a CI excluding 0, of identical sets == 0.
  7. JUDGE RUN. the real ``SelfPlayJudgeV2.run`` with a stub vLLM + OpenAI client over a fake ``SelfPlayTranscribe``
     dataset: ids parsed into category / scenario / replicate, a first malformed reply is retried, rows + summary
     written, a non-scenario row raises; a client that always answers garbage fails the job (> 2%).
  8. TABLE. ``SelfPlayJudgeV2Table.run``: reference row has no delta, the other has the right paired delta.
  9. COPY STATS. ``SelfPlayCopyStats.run`` on a tiny corpus (two episodes) with generated text that copies 12 words of
     its window, 10 of another row of the same episode and 9 of the other episode: W / E-only / O-only counts and
     longest runs exact, examples contiguous. CAN FAIL: a colliding ``ngram_hashes`` changes the counts.

Recipe venv, login node, ~1 min:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_selfplay_judge_v2.py
"""

import contextlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
sys.path.insert(0, "recipe/speech_llm/full_duplex")
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from sisyphus import tk  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.selfplay_bench as b  # noqa: E402
import i6_experiments.users.dorian_koch.speech_llm.selfplay_copy as sc  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.persona_prompts import PERSONA_MINIMAL  # noqa: E402

FAILURES = []
PINNED_DIGEST = "606e4c68091c63cb"


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def fake_voices(n, seed=1):
    rng = np.random.default_rng(seed)
    return [{"label": f"v{i}", "codes": rng.integers(0, 2048, 8 * 50).astype(np.int16)} for i in range(n)]


def as_path(p):
    return tk.Path(p)


# ------------------------------------------------------------------ 1
print("[1] scenario set", flush=True)
sc_v1 = b.SPJ2_SCENARIOS["spj2_v1"]
check(len(sc_v1) == 48, f"48 scenarios ({len(sc_v1)})")
check(all(sum(s[0] == c for s in sc_v1) == 12 for c in b.SPJ2_CATEGORIES), "12 per category")
check(b.scenario_digest("spj2_v1") == PINNED_DIGEST, f"digest pinned ({b.scenario_digest('spj2_v1')})")
voices = fake_voices(96)
r1 = b.scenario_rows(sc_v1, voices, replicates=4, seed=0, minimal=PERSONA_MINIMAL)
r2 = b.scenario_rows(sc_v1, voices, replicates=4, seed=0, minimal=PERSONA_MINIMAL)
check(len(r1) == 192, "48 x 4 rows")
same = all(
    a["id"] == c["id"] and a["context"] == c["context"] and np.array_equal(a["voice_codes"], c["voice_codes"])
    for a, c in zip(r1, r2)
)
check(same, "two builds identical (ids, prompts, voice codes)")
check([r["replicate"] for r in r1] == [k for k in range(4) for _ in range(48)], "replicate-major order")
by = {}
for r in r1:
    by.setdefault(r["scenario"], []).append(r)
check(
    all(
        len({(x["context"][0], x["context_other"][0], x["voice_label"], x["voice_label_other"]) for x in rs}) == 1
        for rs in by.values()
    ),
    "each scenario's 4 rows share prompts and voices",
)
used = [r["voice_label"] for r in r1[:48]] + [r["voice_label_other"] for r in r1[:48]]
check(len(set(used)) == 96, "96 distinct voices, none reused")
check(all(r["context"][0].startswith(PERSONA_MINIMAL + " ") for r in r1), "Minimal line leads every prompt")
r3 = b.scenario_rows(sc_v1, voices, replicates=4, seed=1, minimal=PERSONA_MINIMAL)
check(any(a["voice_label"] != c["voice_label"] for a, c in zip(r1, r3)), "CAN FAIL: seed 1 assigns other voices")
try:
    b.scenario_rows(sc_v1, voices[:95], replicates=4, seed=0, minimal=PERSONA_MINIMAL)
    check(False, "95 voices for 48 scenarios raises")
except RuntimeError:
    check(True, "95 voices for 48 scenarios raises")

# ------------------------------------------------------------------ 2
print("[2] hashes carry the content", flush=True)
j1 = b.SelfPlayScenarioSet(voices=as_path("/tmp/spj2_fake_voices"))
orig = list(sc_v1)
b.SPJ2_SCENARIOS["spj2_v1"] = [orig[0][:3] + (orig[0][3] + " Be brief.",)] + orig[1:]
j2 = b.SelfPlayScenarioSet(voices=as_path("/tmp/spj2_fake_voices"))
b.SPJ2_SCENARIOS["spj2_v1"] = orig
check(j1._sis_id() != j2._sis_id(), "an edited scenario moves the scenario-set hash")
fake_runs = {"m": as_path("/tmp/spj2_fake_asr")}
k1 = b.SelfPlayJudgeV2(runs=fake_runs)
old = b.SPJ2_RUBRICS["spj2_judge_v1"]
b.SPJ2_RUBRICS["spj2_judge_v1"] = old.replace("at most 30 words", "at most 20 words")
k2 = b.SelfPlayJudgeV2(runs=fake_runs)
b.SPJ2_RUBRICS["spj2_judge_v1"] = old
check(k1._sis_id() != k2._sis_id(), "an edited rubric moves the judge hash")
import i6_experiments.users.dorian_koch.speech_llm.selfplay_judge as sj  # noqa: E402

c1 = sj.SelfPlayCoherenceJudge(runs=fake_runs, real_data=as_path("/tmp/spj2_fake_real"))
old_c = json.loads(json.dumps(sj.COHERENCE_RUBRICS["coherence_v1"]))
sj.COHERENCE_RUBRICS["coherence_v1"] = {**old_c, "_edit": "x"} if isinstance(old_c, dict) else str(old_c) + " x"
c2 = sj.SelfPlayCoherenceJudge(runs=fake_runs, real_data=as_path("/tmp/spj2_fake_real"))
sj.COHERENCE_RUBRICS["coherence_v1"] = old_c
check(c1._sis_id() != c2._sis_id(), "an edited coherence rubric moves the coherence judge hash")

# ------------------------------------------------------------------ 3
print("[3] scenario-set job -> self-play worker", flush=True)
from moshi_family.personaplex import selfplay as sp_worker  # noqa: E402

tmp = tempfile.mkdtemp(prefix="spj2_guard_")
vpath = os.path.join(tmp, "voices")
Dataset.from_list([{"label": v["label"], "codes": v["codes"]} for v in voices]).save_to_disk(vpath)
job = b.SelfPlayScenarioSet(voices=as_path(vpath))
job.out_dir = as_path(os.path.join(tmp, "set"))
job.out_scenarios = as_path(os.path.join(tmp, "scenarios.json"))
job.run()
ds = load_from_disk(os.path.join(tmp, "set"))
check(len(ds) == 192, "192 rows written")
row = ds[0]
pa = sp_worker._side_prompt(row, "a", b.SPJ2_LEVEL)
pb = sp_worker._side_prompt(row, "b", b.SPJ2_LEVEL)
check(
    pa == f"{PERSONA_MINIMAL} {sc_v1[0][2]}" and pb == f"{PERSONA_MINIMAL} {sc_v1[0][3]}",
    "worker reads each side's prompt",
)
lab = {v["label"]: v["codes"] for v in voices}
check(
    np.array_equal(np.asarray(row["voice_codes"], np.int16), lab[row["voice_label"]])
    and np.asarray(row["voice_codes_other"]).reshape(8, -1).shape == (8, 50),
    "voice codes round-trip, reshape to [8, V]",
)
meta = json.load(open(os.path.join(tmp, "scenarios.json")))
check(meta["digest"] == PINNED_DIGEST and len(meta["scenarios"]) == 48, "scenarios.json: digest + 48 scenarios")

# ------------------------------------------------------------------ 4
print("[4] judge reply parser", flush=True)
good = {m: {"reason": "fine " * 100, "score": 4} for m in b.SPJ2_METRICS}
p = b.parse_judge_reply("analysis first... " + json.dumps(good) + " trailing")
check(all(p[m]["score"] == 4 for m in b.SPJ2_METRICS), "valid reply parsed")
check(all(len(p[m]["reason"]) <= b.SPJ2_REASON_CHARS for m in b.SPJ2_METRICS), "reasons capped")
bad = []
for mut in ("missing", 4.5, True, "5", 6, "empty"):
    d = json.loads(json.dumps(good))
    if mut == "missing":
        del d["coherence"]
    elif mut == "empty":
        d["coherence"]["reason"] = "  "
    else:
        d["coherence"]["score"] = mut
    bad.append((mut, json.dumps(d)))
rej = 0
for mut, raw in bad:
    try:
        b.parse_judge_reply(raw)
    except ValueError:
        rej += 1
check(rej == len(bad), f"all {len(bad)} malformed replies rejected ({rej})")


def naive(raw):
    d = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    return {m: int(d[m]["score"]) for m in b.SPJ2_METRICS}


acc = 0
for mut, raw in bad:
    try:
        naive(raw)
        acc += 1
    except Exception:
        pass
check(acc >= 4, f"CAN FAIL: the naive parser accepts {acc} of the malformed replies")

# ------------------------------------------------------------------ 5
print("[5] judge message", flush=True)
msg = b.render_judge_message(
    "spj2_judge_v1", prompt_a="PA", prompt_b="PB", duration=60.0, n_a=3, n_b=4, transcript="[0.0] A: {prompt_a} {x}"
)
check(
    "INSTRUCTIONS GIVEN TO A:\nPA" in msg and "B:\nPB" in msg and "Words recognised: A 3, B 4" in msg,
    "placeholders filled",
)
check("[0.0] A: {prompt_a} {x}" in msg, "braces in the transcript kept verbatim")
check(
    not any("{" + k + "}" in msg.split("TRANSCRIPT:")[0] for k in ("prompt_a", "prompt_b", "duration", "n_a", "n_b")),
    "no placeholder left",
)

# ------------------------------------------------------------------ 6
print("[6] nested bootstrap statistics", flush=True)
flat = {f"p{i}": [3.0, 3.0, 3.0, 3.0] for i in range(20)}
ci = b.nested_bootstrap(flat)
check(ci == (3.0, 3.0), f"identical prompts -> zero-width CI {ci}")
rng = np.random.default_rng(0)
mk = {f"p{i}": list(rng.normal(i % 5, 1.0, 50)) for i in range(40)}
st = b.score_stats(mk)
check(abs(st["seed_sd"] - 1.0) < 0.1, f"seed SD ~1 ({st['seed_sd']:.3f})")
check(
    abs(st["prompt_sd"] - np.std([i % 5 for i in range(40)], ddof=1)) < 0.15,
    f"prompt SD ~{np.std([i % 5 for i in range(40)], ddof=1):.2f} ({st['prompt_sd']:.3f})",
)
shift = {k: [x + 1 for x in v] for k, v in mk.items()}
d, dci = b.paired_nested_bootstrap(shift, mk)
check(abs(d - 1.0) < 1e-9 and dci[0] > 0, f"paired +1 shift: {d:.3f} {dci}")
d0, dci0 = b.paired_nested_bootstrap(mk, mk)
check(d0 == 0.0 and dci0[0] <= 0 <= dci0[1], f"paired identical: {d0} {dci0}")

# ------------------------------------------------------------------ 7
print("[7] judge run (stub vLLM)", flush=True)
asr_rows = []
for i, r in enumerate(ds.select(range(6))):
    words_a = [
        {"text": w, "start": 1.0 + 0.3 * j, "end": 1.2 + 0.3 * j}
        for j, w in enumerate("hello there how are you".split())
    ]
    words_b = [] if i == 5 else [{"text": "fine", "start": 3.0, "end": 3.3}]
    asr_rows.append(
        {
            "id": r["id"],
            "n_frames": 750,
            "frame_rate": 12.5,
            "prompt_a": r["context"][0],
            "prompt_b": r["context_other"][0],
            "words_a": json.dumps(words_a),
            "words_b": json.dumps(words_b),
        }
    )
apath = os.path.join(tmp, "asr")
Dataset.from_list(asr_rows).save_to_disk(apath)


class _Msg:
    def __init__(self, c):
        self.message = type("M", (), {"content": c})()


class StubClient:
    calls = {}
    mode = "retry_once"

    def __init__(self, **_):
        self.chat = self
        self.completions = self

    def create(self, *, messages, seed, **kw):
        # judge v2: the schema is ENFORCED via response_format (vLLM ignores extra_body guided_json)
        assert "extra_body" not in kw and kw["response_format"]["json_schema"]["schema"] == b.SPJ2_JSON_SCHEMA
        key = messages[0]["content"]
        n = StubClient.calls.get(key, 0)
        StubClient.calls[key] = n + 1
        if StubClient.mode == "garbage" or n == 0:
            return type("R", (), {"choices": [_Msg("I think it is a 4")]})()
        sc_ = 5 if "B 0." not in key else 1
        rep = {
            m: {"reason": "ok", "score": sc_ if m != "adherence_b" else (1 if "B 0." in key else 4)}
            for m in b.SPJ2_METRICS
        }
        return type("R", (), {"choices": [_Msg(json.dumps(rep))]})()


@contextlib.contextmanager
def stub_server(_name):
    yield "http://stub"


import openai  # noqa: E402

b.vllm_server = stub_server
openai.OpenAI = StubClient
jd = b.SelfPlayJudgeV2(runs={"m": as_path(apath)})
jd.out_rows = as_path(os.path.join(tmp, "rows.jsonl"))
jd.out_summary = as_path(os.path.join(tmp, "summary.json"))
cwd = os.getcwd()
os.chdir(tmp)
try:
    jd.run()
finally:
    os.chdir(cwd)
rows = [json.loads(x) for x in open(os.path.join(tmp, "rows.jsonl"))]
check(len(rows) == 6 and all(r["judged"] for r in rows), "6 rows judged after one retry each")
check(rows[0]["category"] == "topic" and rows[0]["scenario"] == "beekeeping" and rows[0]["replicate"] == 0, "id parsed")
check(max(StubClient.calls.values()) == 2, "a malformed first reply is retried once")
summ = json.load(open(os.path.join(tmp, "summary.json")))
s = summ["labels"]["m"]
check(s["n_conv"] == 6 and abs(s["mute_conv_frac"] - 1 / 6) < 1e-9, "summary n and mute share")
check(summ["rubric_digest"] == b.judge_rubric_digest("spj2_judge_v1"), "summary names the rubric digest")
bad_path = os.path.join(tmp, "asr_bad")
Dataset.from_list([{**asr_rows[0], "id": "rss_x#1@a#w0"}]).save_to_disk(bad_path)
try:
    b.SelfPlayJudgeV2(runs={"m": as_path(bad_path)}).items()
    check(False, "a non-scenario row raises")
except RuntimeError:
    check(True, "a non-scenario row raises")
StubClient.mode, StubClient.calls = "garbage", {}
jg = b.SelfPlayJudgeV2(runs={"m": as_path(apath)}, temperature=0.31)
jg.out_rows = as_path(os.path.join(tmp, "rows_g.jsonl"))
jg.out_summary = as_path(os.path.join(tmp, "summary_g.json"))
os.chdir(tmp)
try:
    jg.run()
    check(False, "always-garbage judge fails the job")
except RuntimeError:
    check(True, "always-garbage judge fails the job")
finally:
    os.chdir(cwd)

# ------------------------------------------------------------------ 8
print("[8] table", flush=True)
ref = [dict(r, label="ref") for r in rows]
up = []
for r in rows:
    j = json.loads(json.dumps(r["judged"]))
    j["coherence"]["score"] = min(5, j["coherence"]["score"])
    j["responsiveness"]["score"] = max(1, j["responsiveness"]["score"] - 1)
    up.append(dict(r, label="new", judged=j))
for name, rr in (("ref", ref), ("new", up)):
    with open(os.path.join(tmp, f"{name}.jsonl"), "w") as f:
        for r in rr:
            f.write(json.dumps(r) + "\n")
tb = b.SelfPlayJudgeV2Table(
    judges={"ref": as_path(os.path.join(tmp, "ref.jsonl")), "new": as_path(os.path.join(tmp, "new.jsonl"))},
    reference="ref",
    order=["ref", "new"],
)
tb.out_json = as_path(os.path.join(tmp, "t.json"))
tb.out_md = as_path(os.path.join(tmp, "t.md"))
tb.run()
t = json.load(open(os.path.join(tmp, "t.json")))
check("delta_vs_ref" not in t["models"]["ref"]["metrics"]["overall"], "reference row has no delta")
dr = t["models"]["new"]["metrics"]["responsiveness"]["delta_vs_ref"]
exp = np.mean(
    [max(1, r["judged"]["responsiveness"]["score"] - 1) - r["judged"]["responsiveness"]["score"] for r in rows]
)
check(abs(dr - exp) < 1e-9, f"paired responsiveness delta {dr:.3f} == {exp:.3f}")
# A renamed arm: its label-free judge's rows still carry the old label ("new"); the table key is "renamed". (A copy of
# the rows file, so this is a different table job, not the one above under other labels.)
shutil.copy(os.path.join(tmp, "new.jsonl"), os.path.join(tmp, "new_copy.jsonl"))
tb2 = b.SelfPlayJudgeV2Table(
    judges={"ref": as_path(os.path.join(tmp, "ref.jsonl")), "renamed": as_path(os.path.join(tmp, "new_copy.jsonl"))},
    reference="ref",
    order=["ref", "renamed"],
)
tb2.out_json = as_path(os.path.join(tmp, "t2.json"))
tb2.out_md = as_path(os.path.join(tmp, "t2.md"))
tb2.run()
t2 = json.load(open(os.path.join(tmp, "t2.json")))
check(
    abs(t2["models"]["renamed"]["metrics"]["responsiveness"]["delta_vs_ref"] - exp) < 1e-9,
    "a judge file whose rows carry the pre-rename label builds under the new key",
)

# ------------------------------------------------------------------ 9
print("[9] verbatim-copy statistics", flush=True)


def words(s, t0=0.0):
    return [{"text": w, "start": t0 + i * 0.3, "end": t0 + i * 0.3 + 0.2} for i, w in enumerate(s.split())]


W_A = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
E_ROW = "one two three four five six seven eight nine ten eleven twelve"
O_ROW = "red orange yellow green blue indigo violet black white grey"
corpus = [
    {"id": "epA#1@a#w0", "alignments": words(W_A), "alignments_other": words("oscar papa quebec")},
    {"id": "epA#2@a#w0", "alignments": words(E_ROW), "alignments_other": []},
    {"id": "epB#1@a#w0", "alignments": words(O_ROW), "alignments_other": words("zulu yankee")},
]
cpath = os.path.join(tmp, "corpus")
Dataset.from_list(corpus).save_to_disk(cpath)
wpath = os.path.join(tmp, "windows")
Dataset.from_list(
    [{"id": "epA#1@a#w0", "alignments": words(W_A), "alignments_other": words("oscar papa quebec")}]
).save_to_disk(wpath)
gen_a = (
    " ".join(W_A.split()[:12])
    + " zz1 "
    + " ".join(E_ROW.split()[:10])
    + " zz2 "
    + " ".join(O_ROW.split()[:9])
    + " zz3 zz4"
)
sp_rows = [{"id": "epA#1@a#w0", "text_a": gen_a, "text_b": "", "words_a": json.dumps(words(gen_a)), "words_b": "[]"}]
spath = os.path.join(tmp, "sp")
Dataset.from_list(sp_rows).save_to_disk(spath)


def run_copy(tag):
    job = sc.SelfPlayCopyStats(runs={"m": as_path(spath)}, windows=as_path(wpath), corpus=as_path(cpath))
    job.out_summary = as_path(os.path.join(tmp, f"copy_{tag}.json"))
    job.out_examples = as_path(os.path.join(tmp, f"copy_{tag}.jsonl"))
    job.run()
    return json.load(open(os.path.join(tmp, f"copy_{tag}.json"))), [
        json.loads(x) for x in open(os.path.join(tmp, f"copy_{tag}.jsonl"))
    ]


def copy_ok(summary, examples, verbose=True):
    t = summary["labels"]["m"]["text"]
    n_ng = len(gen_a.split()) - 7
    exp = {"W": 12 - 7, "E_only": 10 - 7, "O_only": 9 - 7}
    got = {k: round(t[f"share_{k}"] * n_ng) for k in exp}
    runs = (t["run_W"]["max"], t["run_E"]["max"], t["run_O"]["max"])
    ok = got == exp and runs == (12, 12, 9) and all(x["contiguous"] for x in examples if x["source_kind"] == "text")
    if verbose:
        print(f"      counts {got} (expected {exp}), runs W/E/O {runs} (expected (12, 12, 9))", flush=True)
    return ok


s9, e9 = run_copy("real")
check(copy_ok(s9, e9), "W / E-only / O-only counts and longest runs exact, examples contiguous")
check(s9["real_windows_vs_other_episodes"]["windows"] == 1, "real-window base rate computed")
real_hash = sc.ngram_hashes
sc.ngram_hashes = lambda ids, n: real_hash(ids, 1)[: max(len(ids) - n + 1, 0)]
s9m, e9m = run_copy("mutant")
sc.ngram_hashes = real_hash
check(not copy_ok(s9m, e9m, verbose=False), "CAN FAIL: a unigram-colliding ngram_hashes changes the counts")

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):", *FAILURES, sep="\n  ")
    sys.exit(1)
print("ALL PASS")
