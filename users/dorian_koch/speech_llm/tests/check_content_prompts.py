"""Guard: short CONTENT prompts (ablations ab_epad1_chunkprompt / ab_epad1_firsthalf) -- ``PersonaContentPromptGen``
+ ``AttachContentPrompts``.

  1. Both specs use the SAME instruction text (they differ only in what the LLM is shown); the spec content is part of
     the job hash (an edited instruction or word cap is a new job -- shown can fail: plain ``Job.hash`` does not move;
     ``PersonaPromptGen`` had exactly this hole).
  2. ``content_reply_error``: the word cap is enforced (cap + 1 words rejected, cap accepted), a verbatim run of
     ``MAX_SHARED_NGRAM`` words is rejected; ``strip_opener`` removes a repeated opener.
  3. firsthalf, through ``PersonaContentPromptGen.run`` with a stub LLM that records what it is shown: no word with
     onset in the second half of a window ever reaches the LLM, for either side, and the prompted side is labelled
     "Assistant". Non-vacuous: the same check FAILS when ``firsthalf_transcript`` is mutated to cut at the full
     duration.
  4. The job enforces the cap: a reply over the cap is retried; a key that always answers over the cap fails the job
     (> 1%); every written body is <= the cap.
  5. chunk: one call per (episode, 600 s block, speaker), shared by every window of that block and speaker; the
     transcript holds exactly the words of the (widened) block span, the prompted speaker as "Assistant"; each window
     lies inside its span; the two sides of a window get different speakers' prompts.
  6. ``AttachContentPrompts``: only the prompt lists change, minimal keeps its slot, each side gets ITS prompt (a
     side-swapped table is caught); missing prompts drop / unswap (training) or raise (``require_ids``); keep_all_rows
     keeps every row and its other_ok flag.

  7. ``content_firsthalf_v3`` (p6, 2026-09-25): v2's source + request settings; ONE call per (row, side) writes three
     levels (``fh_general`` <= 8, ``fh_topic`` <= 15, ``fh_detailed`` <= 60 words) from the first half only; a level
     over ITS cap retries the whole call; ``AttachContentPrompts`` puts each level's prompt in that level's slot (a
     table with swapped level labels is caught), a side missing one level gets no list; v1 / v2 messages and schema
     unchanged.

The real-data leak (content words of the prompt said in the window / only in its second half / another window of the
episode / another episode) is ``analysis/persona_prompt_leak.py <corpus> <level>`` on the built corpus, not this check.

Recipe venv, login node, ~10 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_content_prompts.py
"""

import contextlib
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

from datasets import Dataset, Features, Sequence, Value, load_from_disk  # noqa: E402
from sisyphus import Job, tk  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.persona_prompts as pp  # noqa: E402

tmp = tempfile.mkdtemp(prefix="check_content_prompts_")
OP = pp.PERSONA_MINIMAL
CH, FH = "content_chunk600_v1", "content_firsthalf_v1"


def expect_fail(fn, text):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        chain, x = [], e
        while x is not None:
            chain.append(str(x))
            x = x.__cause__ or x.__context__
        assert any(text in m for m in chain), chain
        return
    raise AssertionError(f"expected a failure containing {text!r}")


def run(job, sub):
    job.out_dir = tk.Path(os.path.join(tmp, sub))
    job.out_summary = tk.Path(os.path.join(tmp, sub + "_summary.json"))
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        job.run()
    finally:
        os.chdir(cwd)
    return load_from_disk(os.path.join(tmp, sub))


# ---- 1 ----
a, b = pp.CONTENT_SPECS[CH], pp.CONTENT_SPECS[FH]
assert a["instruction"] == b["instruction"] == pp.CONTENT_PROMPT_INSTRUCTION, "A and B must share the instruction"
assert a["max_words"] == b["max_words"] and a["source"] == "chunk" and b["source"] == "firsthalf"
assert set(a) - {"chunk_sec"} == set(b), "the specs may differ only in source (+ chunk_sec)"
assert a["chunk_sec"] == 600.0, "600 s = one Fisher call (1,217 h / 7,303 calls)"
ins = pp.CONTENT_PROMPT_INSTRUCTION
for must in ("ONE short sentence", "At most {max_words} words", "Never name", "Do not copy any phrase", '"prompt"'):
    assert must in ins, f"the content instruction lost {must!r}"
msg = ins.format(max_words=15, transcript="XYZ")
assert msg.endswith("XYZ") and "At most 15 words" in msg
data_path = tk.Path(os.path.join(tmp, "data"))
args = {
    "data": data_path,
    "spec": FH,
    "context_data": None,
    "dialogue_shard_dirs": [],
    "sides": ("assistant", "other"),
    "ids": None,
    "shard": None,
    "num_shards": None,
    "llm_name": "openai/gpt-oss-120b",
    "temperature": 0.6,
}
h0 = pp.PersonaContentPromptGen.hash(args)
assert pp.PersonaContentPromptGen.hash({**args, "spec": CH}) != h0
saved = dict(pp.CONTENT_SPECS[FH])
plain0 = Job.hash.__func__(pp.PersonaContentPromptGen, args)
for edit in ({"instruction": ins + " "}, {"max_words": 16}):
    pp.CONTENT_SPECS[FH].update(edit)
    assert pp.PersonaContentPromptGen.hash(args) != h0, f"an edited spec ({list(edit)}) must change the hash"
    assert Job.hash.__func__(pp.PersonaContentPromptGen, args) == plain0, "(can fail) plain Job.hash is blind to it"
    pp.CONTENT_SPECS[FH].clear()
    pp.CONTENT_SPECS[FH].update(saved)
assert pp.PersonaContentPromptGen.hash(args) == h0
print("[1] A and B share one instruction; spec text + cap are hashed (plain Job.hash is not)")

# ---- 2 ----
cap = a["max_words"]
T = "alpha beta gamma delta epsilon zeta eta theta"
assert pp.content_reply_error(" ".join(["word"] * cap), T, cap) == ""
assert "words >" in pp.content_reply_error(" ".join(["word"] * (cap + 1)), T, cap)
assert pp.content_reply_error("", T, cap) == "empty"
assert "verbatim" in pp.content_reply_error("Talk about alpha beta gamma delta epsilon.", T, cap)
assert pp.content_reply_error("Talk about alpha beta gamma delta.", T, cap) == ""
assert pp.strip_opener(f"{OP}  {OP} Talk about food.") == "Talk about food."
print("[2] cap enforced (cap ok, cap+1 rejected), verbatim 5-gram rejected, repeated opener stripped")

# ---- attached corpus (synthetic) ----
feats = Features(
    {
        "id": Value("string"),
        "duration": Value("float32"),
        "codes_assistant": Sequence(Value("int16")),
        "alignments": [
            {"text": Value("string"), "start": Value("float32"), "end": Value("float32"), "speaker": Value("string")}
        ],
        "context": Sequence(Value("string")),
        "context_level": Sequence(Value("string")),
        "context_set": Value("string"),
        "voice_label": Value("string"),
        "context_other": Sequence(Value("string")),
        "voice_codes_other": Sequence(Value("int16")),
        "voice_n_frames_other": Value("int32"),
        "voice_label_other": Value("string"),
        "alignments_other": [
            {"text": Value("string"), "start": Value("float32"), "end": Value("float32"), "speaker": Value("string")}
        ],
        "other_ok": Value("bool"),
    }
)


def words(prefix, t0, t1, step=1.0):
    out, t = [], t0
    while t < t1:
        out.append({"text": f"{prefix}{int(t * 10)}", "start": t, "end": t + 0.5, "speaker": "assistant"})
        t += step
    return out


# episode ep0: one dialogue row (item ep0#0) at span_start 0 s, 1300 s long; speakers S0 (channel a), S1 (channel b).
# windows (row-relative = episode-relative here): w0 [100, 140) and w1 [400, 430) in block 0; w2 [580, 630) starts in
# block 0 but its midpoint (605 s) is in block 1, so block 1's span widens to 580 s; w3 [700, 740) in block 1.
WIN = [("w0", 100.0, 140.0), ("w1", 400.0, 430.0), ("w2", 580.0, 630.0), ("w3", 700.0, 740.0)]
src_rows, ctx_rows = [], []
for k, (w, t0, t1) in enumerate(WIN):
    rid = f"ep0#0@a#{w}"
    dur = t1 - t0
    half = dur / 2
    asst = [
        {
            **x,
            "start": x["start"] - t0,
            "end": x["end"] - t0,
            "text": ("FIRSTa" if x["start"] - t0 < half else "SECONDa") + x["text"],
        }
        for x in words("a", t0 + 0.2, t1)
    ]
    oth = [
        {
            **x,
            "start": x["start"] - t0,
            "end": x["end"] - t0,
            "text": ("FIRSTb" if x["start"] - t0 < half else "SECONDb") + x["text"],
        }
        for x in words("b", t0 + 0.7, t1)
    ]
    src_rows.append(
        {
            "id": rid,
            "duration": dur,
            "codes_assistant": [k, 1, 2],
            "alignments": asst,
            "context": [f"general {rid}", OP, f"topic {rid}"],
            "context_level": ["general", "minimal", "topic"],
            "context_set": "persona_v1|x|assistant",
            "voice_label": "S0",
            "context_other": [f"ogeneral {rid}", OP, f"otopic {rid}"],
            "voice_codes_other": [8, 8],
            "voice_n_frames_other": 1,
            "voice_label_other": "S1",
            "alignments_other": oth,
            "other_ok": k != 3,  # w3's other side is not swappable
        }
    )
    ctx_rows.append({"id": rid, "item_id": "ep0#0", "episode_id": "ep0", "t0": t0, "t1": t1})
Dataset.from_list(src_rows, features=feats).save_to_disk(data_path.get_path())
ctx_path = tk.Path(os.path.join(tmp, "ctx"))
Dataset.from_list(ctx_rows).save_to_disk(ctx_path.get_path())
slice_dir = os.path.join(tmp, "slices")
wa = [{"text": f"EPa{int(t)}", "start": t, "end": t + 0.5} for t in range(0, 1300, 2)]
wb = [{"text": f"EPb{int(t)}", "start": t + 1.0, "end": t + 1.5} for t in range(0, 1300, 2)]
turns = [{"speaker": "S0", "start": t, "end": t + 0.9} for t in range(0, 1300, 2)] + [
    {"speaker": "S1", "start": t + 1.0, "end": t + 1.9} for t in range(0, 1300, 2)
]
Dataset.from_list(
    [
        {
            "item_id": "ep0#0",
            "episode_id": "ep0",
            "span_start_sec": 0.0,
            "words_a": json.dumps(wa),
            "words_b": json.dumps(wb),
            "turns_json": json.dumps(turns),
        }
    ]
).save_to_disk(os.path.join(slice_dir, "parts", "part0"))

# ---- stub LLM ----
seen, calls = [], {"over_cap_first": set(), "always_over": set(), "v3_over": set()}
v3_msgs = []


class _Stub:
    def create(self, *, messages, seed, **_kw):
        tr = messages[0]["content"].split("TRANSCRIPT:\n", 1)[1]
        seen.append(tr)
        asst = [ln.split()[1] for ln in tr.split("\n") if ln.startswith("Assistant:")]
        first = asst[0] if asst else "none"  # the prompted speaker's first word: the two sides get different prompts
        key = tr[:40]
        if '{"general": "..."' in messages[0]["content"]:  # a multi-level spec (content_firsthalf_v3)
            v3_msgs.append(messages[0]["content"])
            det = f"Have a chat about topic {first[:12]}. You like it."
            if key in calls["v3_over"]:
                calls["v3_over"].discard(key)
                det = " ".join(["long"] * 61)
            d = {
                "general": f"Talk about {first[:12]}.",
                "topic": f"Have a chat about topic {first[:12]}.",
                "detailed": det,
            }
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(d)))]
            )
        if key in calls["always_over"] or key in calls["over_cap_first"]:
            calls["over_cap_first"].discard(key)
            body = " ".join(["long"] * (cap + 3))
        else:
            body = f"Talk about topic {first[:12]} with them."
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps({"prompt": f"{OP} {body}"})))
            ]
        )


class _Client:
    def __init__(self, **_kw):
        self.chat = types.SimpleNamespace(completions=_Stub())


import openai  # noqa: E402

pp.vllm_server = lambda _name: contextlib.nullcontext("http://stub")
openai.OpenAI = _Client


# ---- 3 ----
def firsthalf_leaks(transcripts):
    return [t for t in transcripts if "SECOND" in t]


seen.clear()
fh = run(pp.PersonaContentPromptGen(data=data_path, spec=FH), "fh")
assert len(seen) == 4 + 3, len(seen)  # 4 assistant sides + 3 swappable other sides
assert not firsthalf_leaks(seen), "a second-half word reached the LLM"
fh_seen = list(seen)
for t in seen:
    lines = t.split("\n")
    a_side = {ln.split(": ", 1)[1].split()[0][:6] for ln in lines if ln.startswith("Assistant:")}
    assert len(a_side) == 1 and a_side <= {"FIRSTa", "FIRSTb"}, a_side
assert sum("Assistant: FIRSTb" in t for t in seen) == 3, "the other side must be the prompted 'Assistant'"
assert set(zip(fh["id"], fh["side"])) == {(r["id"], "assistant") for r in src_rows} | {
    (r["id"], "other") for r in src_rows if r["other_ok"]
}
assert all(n <= cap for n in fh["n_words"]) and all(t.startswith(OP + " ") and t.count(OP) == 1 for t in fh["text"])
# mutant: cut at the full duration -> the check above must fail
orig = pp.firsthalf_transcript
pp.firsthalf_transcript = lambda f, o, d: orig(f, o, 2 * float(d))
seen.clear()
run(pp.PersonaContentPromptGen(data=data_path, spec=FH, temperature=0.5), "fh_mutant")
assert firsthalf_leaks(seen), "(can fail) the mutant must leak second-half words into the LLM input"
pp.firsthalf_transcript = orig
print(f"[3] firsthalf: no second-half word reaches the LLM ({len(fh)} prompts); the full-window mutant is caught")

# ---- 4 ----
k0 = fh_seen[0][:40]
calls["over_cap_first"] = {k0}
seen.clear()
ok = run(pp.PersonaContentPromptGen(data=data_path, spec=FH, temperature=0.3), "fh_retry")
assert len(ok) == len(fh) and len(seen) == len(fh) + 1 and all(n <= cap for n in ok["n_words"])
calls["always_over"] = {k0}
expect_fail(
    lambda: run(pp.PersonaContentPromptGen(data=data_path, spec=FH, temperature=0.2), "fh_fail"), "failed on 1/"
)
calls["always_over"] = set()
print("[4] an over-cap reply is retried; a key always over the cap fails the job; every body <= cap")

# ---- 5 ----
seen.clear()
chk = run(
    pp.PersonaContentPromptGen(
        data=data_path, spec=CH, context_data=ctx_path, dialogue_shard_dirs=[tk.Path(slice_dir)]
    ),
    "ch",
)
# blocks by midpoint: w0 120 -> 0, w1 415 -> 0, w2 605 -> 1, w3 720 -> 1; speakers S0 (all), S1 (w0-w2)
keys = set(chk["key"])
assert keys == {"ep0|0|S0", "ep0|0|S1", "ep0|1|S0", "ep0|1|S1"}, keys
assert len(seen) == 4, "one call per (episode, block, speaker)"
kk = {(r["id"], r["side"]): r["key"] for r in chk}
assert kk[("ep0#0@a#w0", "assistant")] == kk[("ep0#0@a#w1", "assistant")] == "ep0|0|S0"
assert kk[("ep0#0@a#w2", "assistant")] == kk[("ep0#0@a#w3", "assistant")] == "ep0|1|S0"
assert kk[("ep0#0@a#w0", "other")] == "ep0|0|S1" and ("ep0#0@a#w3", "other") not in kk
texts, key_of = pp.chunk_transcripts(
    [
        {"id": c["id"], "episode_id": "ep0", "abs_t0": c["t0"], "abs_t1": c["t1"], "labels": ["S0", "S1"]}
        for c in ctx_rows
    ],
    list(load_from_disk(os.path.join(slice_dir, "parts", "part0"))),
    600.0,
)
allw = [(w["start"], w["text"]) for w in wa + wb]
for (ep, blk, lab), (tr, s0, s1) in texts.items():
    got = sorted(t for ln in tr.split("\n") for t in ln.split(": ", 1)[1].split())
    assert got == sorted(t for st, t in allw if s0 <= st < s1), (
        blk,
        lab,
        "the chunk must hold exactly its span's words",
    )
    foc = "EPa" if lab == "S0" else "EPb"
    assert all(ln.split(": ", 1)[1].startswith(foc) == ln.startswith("Assistant:") for ln in tr.split("\n")), lab
for c in ctx_rows:
    tr, s0, s1 = texts[key_of[(c["id"], "S0")]]
    assert s0 <= c["t0"] and c["t1"] <= s1, "a window must lie inside its chunk"
assert texts[("ep0", 0, "S0")][1:] == (0.0, 600.0) and texts[("ep0", 1, "S0")][1:] == (580.0, 1200.0)
print("[5] chunk: one call per (episode, 600 s block, speaker), block = the widened span, focus speaker = Assistant")

# ---- 6 ----
fh_path = tk.Path(os.path.join(tmp, "fh"))
LV = ("firsthalf", "minimal", "firsthalf")


def attach(sub, prompts=(fh_path,), **kw):
    return run(pp.AttachContentPrompts(data=data_path, prompts=list(prompts), prompt_set="set", **kw), sub)


out = attach("att", levels=LV)
src = load_from_disk(data_path.get_path())
ptext = {(r["id"], r["side"]): r["text"] for r in fh}
assert out["id"] == src["id"]
for s, o in zip(src, out):
    for k in feats:
        if k not in ("context", "context_level", "context_set", "context_other"):
            assert s[k] == o[k], k
    pa = ptext[(s["id"], "assistant")]
    assert o["context"] == [pa, OP, pa] and o["context_level"] == list(LV)
    if s["other_ok"]:
        pb = ptext[(s["id"], "other")]
        assert o["context_other"] == [pb, OP, pb] and pa != pb
    assert [x == OP for x in o["context"]] == [x == OP for x in s["context"]]
swapped = [{**r, "side": "other" if r["side"] == "assistant" else "assistant"} for r in fh]
Dataset.from_list(swapped).save_to_disk(os.path.join(tmp, "fh_swapped"))
bad = attach("att_bad", prompts=(tk.Path(os.path.join(tmp, "fh_swapped")),), levels=LV, max_drop_frac=1.0)
assert any(o["context"][0] != ptext[(o["id"], "assistant")] for o in bad), "(can fail) a side swap must be visible"
Dataset.from_list([r for r in fh if not (r["id"].endswith("w1"))]).save_to_disk(os.path.join(tmp, "fh_missing"))
miss = (tk.Path(os.path.join(tmp, "fh_missing")),)
att_m = attach("att_m", prompts=miss, levels=LV, max_drop_frac=0.5)
assert len(att_m) == len(src) - 1
expect_fail(lambda: attach("att_m_strict", prompts=miss, levels=LV), "hit noprompt")
expect_fail(
    lambda: attach(
        "att_m_req", prompts=miss, levels=("minimal", "firsthalf"), keep_all_rows=True, require_ids=["ep0#0@a#w1"]
    ),
    "required",
)
keep = attach("att_keep", prompts=miss, levels=("minimal", "firsthalf"), keep_all_rows=True, require_ids=["ep0#0@a#w0"])
assert keep["id"] == src["id"] and keep["other_ok"] == src["other_ok"]
r1 = keep[1]
assert r1["context"] == [OP, OP] and r1["context_other"] == [OP, OP], "a non-required row without prompts keeps minimal"
assert keep[0]["context"][1] == ptext[("ep0#0@a#w0", "assistant")]
print("[6] AttachContentPrompts: only prompt lists change, sides kept apart, drops/unswaps/raises as specified")

# ---- 7 ---- content_firsthalf_v3: three levels from ONE call per (row, side), first half only
V3 = "content_firsthalf_v3"
s3, s2 = pp.CONTENT_SPECS[V3], pp.CONTENT_SPECS["content_firsthalf_v2"]
assert s3["source"] == "firsthalf" and {k: s3[k] for k in ("reasoning_effort", "concurrency")} == {
    k: s2[k] for k in ("reasoning_effort", "concurrency")
}, "v3 keeps v2's source and request settings"
assert pp.content_levels(s3) == {
    "fh_general": ("general", 8),
    "fh_topic": ("topic", 15),
    "fh_detailed": ("detailed", 60),
}
assert pp.content_levels(s2) == {"firsthalf": ("prompt", 15)} and pp.content_json_schema(s2) is pp._CONTENT_JSON_SCHEMA
assert pp.content_message(s2, "XYZ") == s2["instruction"].format(max_words=15, transcript="XYZ"), "v1/v2 message kept"
m3 = pp.content_message(s3, "XYZ")
assert "at most 8 words" in m3 and "at most 15 words" in m3 and "at most 60 words" in m3 and m3.endswith("XYZ")
assert pp.content_spec_digest(V3) != pp.content_spec_digest("content_firsthalf_v2")
assert pp.PersonaContentPromptGen.hash({**args, "spec": V3}) != h0
v3_msgs.clear()
seen.clear()
job3 = pp.PersonaContentPromptGen(data=data_path, spec=V3)
assert job3.levels == ("fh_general", "fh_topic", "fh_detailed")
fh3 = run(job3, "fh3")
assert len(v3_msgs) == 4 + 3, f"ONE call per (row, side): {len(v3_msgs)}"
assert not firsthalf_leaks(seen), "v3: a second-half word reached the LLM"
got = {}
for r in fh3:
    got.setdefault((r["id"], r["side"]), {})[r["level"]] = r
assert set(got) == set(zip(fh["id"], fh["side"])) and all(set(v) == set(job3.levels) for v in got.values())
caps = {lv: cap for lv, (_k, cap) in pp.content_levels(s3).items()}
assert all(r["n_words"] <= caps[r["level"]] for r in fh3) and all(r["text"].startswith(OP + " ") for r in fh3)
summ3 = json.load(open(os.path.join(tmp, "fh3_summary.json")))
assert set(summ3["levels"]) == set(job3.levels)
# a detailed reply over ITS cap (61 > 60) retries the whole call; the other levels are never written alone
calls["v3_over"] = {fh_seen[0][:40]}
v3_msgs.clear()
fh3r = run(pp.PersonaContentPromptGen(data=data_path, spec=V3, temperature=0.4), "fh3_retry")
assert len(v3_msgs) == 7 + 1 and len(fh3r) == len(fh3), "an over-cap level retries the call"
# attach: each slot gets ITS level's prompt, per side
LV3 = ("fh_general", "minimal", "fh_topic")
fh3_path = tk.Path(os.path.join(tmp, "fh3"))
att3 = attach("att3", prompts=(fh3_path,), levels=LV3)
t3 = {(r["id"], r["side"], r["level"]): r["text"] for r in fh3}
for o in att3:
    assert o["context"] == [t3[(o["id"], "assistant", "fh_general")], OP, t3[(o["id"], "assistant", "fh_topic")]]
    assert o["context_level"] == list(LV3)
    if o["other_ok"]:
        assert o["context_other"] == [t3[(o["id"], "other", "fh_general")], OP, t3[(o["id"], "other", "fh_topic")]]
# CAN FAIL: a table with the level labels swapped puts the topic prompt in the general slot -- visible
mixed = [{**r, "level": {"fh_general": "fh_topic", "fh_topic": "fh_general"}.get(r["level"], r["level"])} for r in fh3]
Dataset.from_list(mixed).save_to_disk(os.path.join(tmp, "fh3_mixed"))
bad3 = attach("att3_bad", prompts=(tk.Path(os.path.join(tmp, "fh3_mixed")),), levels=LV3)
assert any(o["context"][0] != t3[(o["id"], "assistant", "fh_general")] for o in bad3), "(can fail) level mix-up"
# a (row, side) missing ONE of its levels has no prompt list (dropped / unswapped), never a partial one
Dataset.from_list([r for r in fh3 if not (r["id"].endswith("w1") and r["level"] == "fh_topic")]).save_to_disk(
    os.path.join(tmp, "fh3_missing")
)
att3m = attach("att3_m", prompts=(tk.Path(os.path.join(tmp, "fh3_missing")),), levels=LV3, max_drop_frac=0.5)
assert len(att3m) == len(src) - 1 and "ep0#0@a#w1" not in att3m["id"]
expect_fail(lambda: attach("att3_lv", prompts=(fh3_path,), levels=("minimal", "firsthalf")), "in no prompt table")
print("[7] v3: one call per (row, side) -> 3 levels, caps per level (an over-cap level retries the call), first half")
print("    only; attach puts each level in its slot (a level mix-up is caught), a missing level = no list")
print("ALL OK")
