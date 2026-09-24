"""Guard: speaker-profile prompts (ablation ab_epad1_persona) -- ``PersonaProfileGen`` + ``AttachPersonaProfiles``.

  1. The spec forbids window content (topics, claims, stories, events, names, specific facts) and asks for a
     SPEAKER profile (role, style, broad background); the instruction text is part of the job hash (an edited
     spec is a new job -- shown can fail: Sisyphus' plain ``Job.hash`` without the override does not move).
  2. ``speaker_excerpts``: each window's assistant words go to (episode, assistant_label), its other channel's words
     to (episode, user_label); windows in dialogue order (row ``#2`` before ``#10``); over ``max_words`` an evenly
     spaced subset that fits, first and last window kept.
  3. ``PersonaProfileGen.run`` with a stub LLM that answers with the speaker tag it was shown: every key's profile is
     written from ITS OWN words; a reply copying >= MAX_SHARED_NGRAM words is retried, and a key that always copies
     fails the job (> 1%).
  4. ``AttachPersonaProfiles`` on a synthetic attached corpus: only ``context``/``context_level``/``context_set``/
     ``context_other`` change; the profile sits where the source had general/topic and minimal stays at its
     position; every window of one (episode, speaker) carries the SAME text; the other side gets ITS speaker's
     profile (a label-swapped profile table is caught); a missing profile drops the row (training) or raises
     (``keep_all_rows``, other_ok); a missing other-side profile makes the row unswappable.
  5. ``profile_text``: a reply that repeats the opener "You enjoy having a good conversation." (7% of real replies)
     yields a prompt with the opener exactly once.

The real-data leak number (content words of the profile said in the same window vs another window of the same
episode) is ``analysis/persona_prompt_leak.py <corpus> profile`` on the built corpus, not this check.

Recipe venv, login node, ~10 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_persona_profiles.py
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

tmp = tempfile.mkdtemp(prefix="check_persona_profiles_")


def expect_fail(fn, text):
    """``fn()`` must raise with ``text`` in the message or its cause (``datasets`` wraps generator errors)."""
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
ins = pp.PROFILE_SPECS["persona_profile_v1"]["instruction"]
for must in (
    "Do NOT mention any topic, subject, claim, opinion, story, anecdote, event",
    "must say nothing about what is discussed",
    "never a specific experience, achievement or fact",
    "Never name the assistant or anyone else",
    "not what they talk about",
    "role in a conversation",
    "speaking style",
    "background or expertise in broad terms only",
):
    assert must in ins, f"the profile spec lost the rule {must!r}"
assert "{transcript}" in ins and ins.format(transcript="XYZ").endswith("XYZ")
ctx_path = tk.Path(os.path.join(tmp, "ctx"))
h0 = pp.PersonaProfileGen(context_data=ctx_path)._sis_hash()
saved = dict(pp.PROFILE_SPECS["persona_profile_v1"])
pp.PROFILE_SPECS["persona_profile_v1"]["instruction"] = ins + " "
args = {
    "context_data": ctx_path,
    "spec": "persona_profile_v1",
    "llm_name": "openai/gpt-oss-120b",
    "max_words": 3000,
    "temperature": 0.6,
}
h_edit = pp.PersonaProfileGen.hash(args)
plain_edit = Job.hash.__func__(pp.PersonaProfileGen, args)
pp.PROFILE_SPECS["persona_profile_v1"].update(saved)
assert pp.PersonaProfileGen.hash(args) != h_edit, "an edited instruction must change the job hash"
assert Job.hash.__func__(pp.PersonaProfileGen, args) == plain_edit, (
    "(can fail) without the override the edit is invisible"
)
assert pp.PersonaProfileGen.hash(args) == pp.PersonaProfileGen.hash(dict(args)) and h0
print(
    "[1] spec forbids window content, asks for role/style/broad background; its text is hashed (plain Job.hash is not)"
)

# ---- 2 ----
ctx_rows = []
for e in range(3):
    for item in (10, 2):  # out of order on purpose: #2 must come first
        for k in range(4):
            a, b = (f"S{2 * e}", f"S{2 * e + 1}") if item == 2 else (f"S{2 * e + 1}", f"S{2 * e}")
            ctx_rows.append(
                {
                    "id": f"ep{e}#{item}@a#w{k}",
                    "item_id": f"ep{e}#{item}",
                    "episode_id": f"ep{e}",
                    "t0": 30.0 * k,
                    "assistant_label": a,
                    "assistant_text": " ".join(f"tag{a}_ep{e}_{item}_{k}_{i}" for i in range(10)),
                    "user_label": b,
                    "user_text": " ".join(f"tag{b}_ep{e}_{item}_{k}_{i}" for i in range(10)),
                    "channel": "a",
                }
            )
ex = pp.speaker_excerpts(ctx_rows, max_words=10**6)
assert set(ex) == {(f"ep{e}", f"S{s}") for e in range(3) for s in (2 * e, 2 * e + 1)}, sorted(ex)
for (ep, lab), (text, n_words, n_win, n_used) in ex.items():
    toks = text.split()
    assert all(t.startswith(f"tag{lab}_{ep}_") for t in toks), (ep, lab, toks[:3])
    assert n_words == 80 and n_win == 8 and n_used == 8
    order = [(int(t.split("_")[2]), int(t.split("_")[3])) for t in toks[::10]]
    assert order == sorted(order) and order[0] == (2, 0), order
thin = pp.speaker_excerpts(ctx_rows, max_words=35)
for (ep, lab), (text, n_words, n_win, n_used) in thin.items():
    lines = text.split("\n")
    assert n_used == len(lines) == 3 and len(text.split()) <= 35, (n_used, len(text.split()))
    first, last = lines[0].split()[0], lines[-1].split()[0]
    assert first.split("_")[2:4] == ["2", "0"] and last.split("_")[2:4] == ["10", "3"], (first, last)
print("[2] speaker_excerpts: words by side label, dialogue order, even thinning keeps first + last window")

# ---- 3 ----
Dataset.from_list(ctx_rows).save_to_disk(ctx_path.get_path())
calls = {"n": 0, "copy_first": set(), "always_copy": set()}


class _Stub:
    def create(self, *, messages, seed, **_kw):
        calls["n"] += 1
        words = messages[0]["content"].split("EXCERPTS:")[1].split()
        tag = words[0].split("_")[0]
        key = (words[0].split("_")[1], tag)
        if key in calls["always_copy"] or key in calls["copy_first"]:
            body = " ".join(words[:8])  # a verbatim copy -> must be retried / fail
            calls["copy_first"].discard(key)
        else:
            body = f"You are speaker {tag} and you talk calmly."
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps({"profile": body})))]
        )


class _Client:
    def __init__(self, **_kw):
        self.chat = types.SimpleNamespace(completions=_Stub())


import openai  # noqa: E402

pp.vllm_server = lambda _name: contextlib.nullcontext("http://stub")
openai.OpenAI = _Client
g = pp.PersonaProfileGen(context_data=ctx_path)
prof = run(g, "prof")
assert len(prof) == 6
for r in prof:
    assert r["text"] == f"{pp.PERSONA_MINIMAL} You are speaker tag{r['label']} and you talk calmly.", r
    assert r["max_ngram_overlap"] < pp.MAX_SHARED_NGRAM
calls["copy_first"] = {("ep0", "tagS0")}
n_before = calls["n"]
prof_retry = run(pp.PersonaProfileGen(context_data=ctx_path, temperature=0.5), "prof_retry")
assert len(prof_retry) == 6 and calls["n"] >= n_before + 6
calls["always_copy"] = {("ep1", "tagS2")}
expect_fail(lambda: run(pp.PersonaProfileGen(context_data=ctx_path, temperature=0.4), "prof_fail"), "failed on 1/6")
calls["always_copy"] = set()
print(
    "[3] PersonaProfileGen: each (episode, speaker) from its own words; verbatim replies retried, persistent copies fail"
)

# ---- 4 ----
feats = Features(
    {
        "id": Value("string"),
        "codes_assistant": Sequence(Value("int16")),
        "alignments": [
            {"text": Value("string"), "start": Value("float32"), "end": Value("float32"), "speaker": Value("string")}
        ],
        "context": Sequence(Value("string")),
        "context_level": Sequence(Value("string")),
        "context_set": Value("string"),
        "voice_codes": Sequence(Value("int16")),
        "voice_n_frames": Value("int32"),
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
src_rows = []
for c in ctx_rows:
    src_rows.append(
        {
            "id": c["id"],
            "codes_assistant": [len(src_rows), 1, 2],
            "alignments": [{"text": "w", "start": 0.0, "end": 0.1, "speaker": "assistant"}],
            "context": [f"general {c['id']}", pp.PERSONA_MINIMAL, f"topic {c['id']}"],
            "context_level": ["general", "minimal", "topic"],
            "context_set": "persona_v1|x|assistant",
            "voice_codes": [7, 7],
            "voice_n_frames": 1,
            "voice_label": c["assistant_label"],
            "context_other": [f"ogeneral {c['id']}", pp.PERSONA_MINIMAL, f"otopic {c['id']}"],
            "voice_codes_other": [8, 8],
            "voice_n_frames_other": 1,
            "voice_label_other": c["user_label"],
            "alignments_other": [{"text": "u", "start": 0.0, "end": 0.1, "speaker": "assistant"}],
            "other_ok": True,
        }
    )
src_path = os.path.join(tmp, "src")
Dataset.from_list(src_rows, features=feats).save_to_disk(src_path)
prof_path = tk.Path(os.path.join(tmp, "prof"))


def attach(sub, profiles=prof_path, **kw):
    return run(
        pp.AttachPersonaProfiles(
            data=tk.Path(src_path), context_data=ctx_path, profiles=profiles, prompt_set=g.prompt_set, **kw
        ),
        sub,
    )


out = attach("att")
ptext = {(r["episode_id"], r["label"]): pp.profile_text(r["profile"]) for r in prof}
src = load_from_disk(src_path)
assert out["id"] == src["id"]
changed = {"context", "context_level", "context_set", "context_other"}
by_speaker = {}
for s, o, c in zip(src, out, ctx_rows):
    for k in feats:
        if k not in changed:
            assert s[k] == o[k], (k, s[k], o[k])
    pa, pb = ptext[(c["episode_id"], c["assistant_label"])], ptext[(c["episode_id"], c["user_label"])]
    assert o["context"] == [pa, pp.PERSONA_MINIMAL, pa] and o["context_level"] == ["profile", "minimal", "profile"]
    assert o["context_other"] == [pb, pp.PERSONA_MINIMAL, pb] and o["context_set"] == g.prompt_set
    assert [x == pp.PERSONA_MINIMAL for x in o["context"]] == [x == pp.PERSONA_MINIMAL for x in s["context"]]
    by_speaker.setdefault((c["episode_id"], c["assistant_label"]), set()).add(o["context"][0])
assert all(len(v) == 1 for v in by_speaker.values()), "one profile per (episode, speaker), whatever the window"
# can fail: a profile table with the labels swapped puts the partner's profile on the assistant side
swapped = [
    {**r, "label": ("S1" if r["label"] == "S0" else "S0") if r["episode_id"] == "ep0" else r["label"]} for r in prof
]
Dataset.from_list(swapped).save_to_disk(os.path.join(tmp, "prof_swapped"))
bad = attach("att_bad", profiles=tk.Path(os.path.join(tmp, "prof_swapped")))
assert any(o["context"][0] != ptext[(c["episode_id"], c["assistant_label"])] for o, c in zip(bad, ctx_rows)), (
    "the side check must be able to fail"
)
# missing profiles
Dataset.from_list([r for r in prof if not (r["episode_id"] == "ep2" and r["label"] == "S4")]).save_to_disk(
    os.path.join(tmp, "prof_missing")
)
miss = tk.Path(os.path.join(tmp, "prof_missing"))
att_m = attach("att_m", profiles=miss, max_drop_frac=0.5)
n_asst = sum(c["episode_id"] == "ep2" and c["assistant_label"] == "S4" for c in ctx_rows)
n_oth = sum(c["episode_id"] == "ep2" and c["user_label"] == "S4" for c in ctx_rows)
assert len(att_m) == len(src) - n_asst and sum(not x for x in att_m["other_ok"]) == n_oth, (len(att_m), n_asst, n_oth)
expect_fail(
    lambda: attach("att_m_keep", profiles=miss, levels=("minimal", "profile"), keep_all_rows=True), "row without"
)
expect_fail(lambda: attach("att_m_strict", profiles=miss), "hit noprofile")
keep = attach("att_keep", levels=("minimal", "profile"), keep_all_rows=True)
assert keep["id"] == src["id"] and keep["other_ok"] == src["other_ok"]
assert all(r["context_level"] == ["minimal", "profile"] and r["context"][0] == pp.PERSONA_MINIMAL for r in keep)
print(
    f"[4] AttachPersonaProfiles: only the prompt lists change, minimal keeps its slot, one profile per speaker, "
    f"sides not swapped (a swapped table is caught), missing profiles drop ({n_asst}) / unswap ({n_oth}) / raise"
)
# ---- 5 ----
OP = pp.PERSONA_MINIMAL
assert pp.profile_text(f"{OP} {OP}  You joke a lot.") == f"{OP} You joke a lot."
assert pp.profile_text("You joke a lot.") == f"{OP} You joke a lot."
Dataset.from_list([{**r, "profile": f"{OP} {r['profile']}"} for r in prof]).save_to_disk(os.path.join(tmp, "prof_op"))
att_op = attach("att_op", profiles=tk.Path(os.path.join(tmp, "prof_op")))
assert all(x.count(OP) == 1 for r in att_op for x in r["context"] if x != OP), "a repeated opener must be removed"
assert att_op["context"] == out["context"]
print("[5] an opener the LLM repeated is dropped: the prompt starts with it exactly once")
print("ALL OK")
