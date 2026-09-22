"""Guard: persona-prompt data jobs -- verbatim detector, level choice, window-context join, attach.

  1. ``max_shared_ngram`` finds a planted 6-word copy (>= MAX_SHARED_NGRAM -> would be rejected) and
     scores a paraphrase below the threshold.
  2. ``choose_level`` is deterministic per id and proportional to the weights (10k ids, +-2 pt).
  3. ``PodcastWindowContext`` reproduces EXACTLY the ids the real ``PodcastCodesTrainData`` wrote for a
     synthetic dialogue (window_select + seeded random channel), and each window's assistant text is
     the training row's alignment words. A context built with a different ``selection_seed`` must
     FAIL (it would pick other channels) -- the join is not vacuous.
  4. ``AttachPersonaPrompts`` (minimal level) writes ``context`` = the paper's Minimal prompt and the
     speaker's ``voice_codes``; with the voice missing it raises instead of shrinking the corpus.

Recipe venv, login node, ~20 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_persona_prompts.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.persona_prompts import (  # noqa: E402
    MAX_SHARED_NGRAM,
    PERSONA_MINIMAL,
    AttachPersonaPrompts,
    PodcastWindowContext,
    choose_level,
    max_shared_ngram,
)
from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import PodcastCodesTrainData  # noqa: E402

# ---- 1 ----
tr = "so I was driving down to the coast last weekend and the traffic was insane"
assert max_shared_ngram("You talk about driving down to the coast last weekend.", tr) >= MAX_SHARED_NGRAM
assert max_shared_ngram("Have a casual conversation about a road trip and bad traffic.", tr) < MAX_SHARED_NGRAM
print("[1] verbatim detector: planted copy caught, paraphrase passes")

# ---- 2 ----
w = {"minimal": 0.25, "general": 0.25, "topic": 0.5}
picks = [choose_level(w, 0, f"id{i}") for i in range(10000)]
assert picks == [choose_level(w, 0, f"id{i}") for i in range(10000)], "not deterministic"
for k, v in w.items():
    assert abs(picks.count(k) / len(picks) - v) < 0.02, (k, picks.count(k))
print("[2] choose_level deterministic and proportional")

# ---- 3 ----
RULE = dict(min_sec=20.0, max_sec=60.0, min_turns=3, max_stretch_sec=30.0, turn_min_sec=1.0)
K, FR, DUR = 8, 12.5, 100.0
F = int(DUR * FR)
rows = []
for e in range(6):
    turns = [{"speaker": f"S{i % 2 + 2 * e}", "start": i * 5.0, "end": (i + 1) * 5.0} for i in range(20)]
    wa = [
        {"text": f"a{e}_{i}", "start": i * 1.0 + 0.1, "end": i * 1.0 + 0.5, "speaker": "x"}
        for i in range(0, 100, 2)
        if (i // 5) % 2 == 0
    ]
    wb = [
        {"text": f"b{e}_{i}", "start": i * 1.0 + 0.1, "end": i * 1.0 + 0.5, "speaker": "y"}
        for i in range(0, 100, 2)
        if (i // 5) % 2 == 1
    ]
    rows.append(
        {
            "item_id": f"ep{e}#0",
            "episode_id": f"ep{e}",
            "duration_sec": DUR,
            "n_codebooks": K,
            "n_frames": F,
            "frame_rate": FR,
            "codes_a": np.zeros(K * F, dtype=np.int16).tolist(),
            "codes_b": np.zeros(K * F, dtype=np.int16).tolist(),
            "words_a": json.dumps(wa),
            "words_b": json.dumps(wb),
            "turns_json": json.dumps(turns),
            "meta": json.dumps({"title": f"#{e}"}),
        }
    )
tmp = tempfile.mkdtemp(prefix="check_persona_")
Dataset.from_list(rows).save_to_disk(os.path.join(tmp, "dlg", "parts", "part_000"))


def run(job, sub):
    job.out_dir = tk.Path(os.path.join(tmp, sub))
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        job.run()
    finally:
        os.chdir(cwd)
    return load_from_disk(os.path.join(tmp, sub))


dlg = [tk.Path(os.path.join(tmp, "dlg"))]
train = run(
    PodcastCodesTrainData(shard_dirs=dlg, assistant_channel="random", window_select=RULE, min_assistant_words=1),
    "train",
)
ctx = run(
    PodcastWindowContext(train_data=tk.Path(os.path.join(tmp, "train")), dialogue_shard_dirs=dlg, window_select=RULE),
    "ctx",
)
assert sorted(ctx["id"]) == sorted(train["id"]) and len(train) > 0, (len(ctx), len(train))
by = {r["id"]: r for r in ctx}
for r in train:
    assert by[r["id"]]["assistant_text"] == " ".join(a["text"] for a in r["alignments"]), r["id"]
    assert by[r["id"]]["assistant_label"] is not None
assert {r["channel"] for r in ctx} == {"a", "b"}, "synthetic rows should exercise both channels"
try:
    run(
        PodcastWindowContext(
            train_data=tk.Path(os.path.join(tmp, "train")),
            dialogue_shard_dirs=dlg,
            window_select=RULE,
            selection_seed=1,
        ),
        "ctx_bad",
    )
    raise AssertionError("a context built with another channel seed was accepted -- the join check is vacuous")
except RuntimeError as e:
    assert "not reproduced" in str(e), e
print(f"[3] window context reproduces all {len(train)} training ids and their assistant words; a wrong seed is refused")

# ---- 4 ----
vrows = [
    {
        "episode_id": r["episode_id"],
        "label": r["assistant_label"],
        "codes": np.arange(8 * 50, dtype=np.int16).tolist(),
        "n_frames": 50,
        "clip_sec": 4.0,
        "source_sec": 30.0,
    }
    for r in ctx
]
Dataset.from_list(vrows).save_to_disk(os.path.join(tmp, "voice"))
att = run(
    AttachPersonaPrompts(
        train_data=tk.Path(os.path.join(tmp, "train")),
        context_data=tk.Path(os.path.join(tmp, "ctx")),
        voice_codes=tk.Path(os.path.join(tmp, "voice")),
        level_weights={"minimal": 1.0},
    ),
    "att",
)
assert len(att) == len(train) and set(att["context"]) == {PERSONA_MINIMAL} and set(att["context_level"]) == {"minimal"}
assert all(len(v) == 8 * 50 for v in att["voice_codes"])
Dataset.from_list(vrows[:1]).save_to_disk(os.path.join(tmp, "voice_few"))
try:
    run(
        AttachPersonaPrompts(
            train_data=tk.Path(os.path.join(tmp, "train")),
            context_data=tk.Path(os.path.join(tmp, "ctx")),
            voice_codes=tk.Path(os.path.join(tmp, "voice_few")),
            level_weights={"minimal": 1.0},
        ),
        "att_bad",
    )
    raise AssertionError("missing voices were silently dropped")
except RuntimeError as e:
    assert "novoice" in str(e), e
print(f"[4] attach: {len(att)} rows with the Minimal prompt + voice codes; missing voices raise")
# ---- 5 ----
# per_read=True: every level's prompt is stored as a list (sorted level order) for the loader to draw
# from on each read; unequal weights are refused because the loader's draw is uniform.
prows = [
    {
        "id": r["id"],
        "prompt_set": "t|t|assistant",
        "level": lv,
        "text": f"You enjoy having a good conversation. {lv} {r['id']}",
        "llm_name": "t",
        "input_view": "assistant",
        "spec": "t",
        "max_ngram_overlap": 0,
    }
    for r in ctx
    for lv in ("general", "topic")
]
Dataset.from_list(prows).save_to_disk(os.path.join(tmp, "prompts"))
w3 = {"minimal": 1 / 3, "general": 1 / 3, "topic": 1 / 3}
att3 = run(
    AttachPersonaPrompts(
        train_data=tk.Path(os.path.join(tmp, "train")),
        context_data=tk.Path(os.path.join(tmp, "ctx")),
        voice_codes=tk.Path(os.path.join(tmp, "voice")),
        level_weights=w3,
        prompt_data=[tk.Path(os.path.join(tmp, "prompts"))],
        prompt_set="t|t|assistant",
        per_read=True,
    ),
    "att3",
)
assert len(att3) == len(train)
for r in att3:
    assert r["context_level"] == ["general", "minimal", "topic"], r["context_level"]
    assert r["context"] == [
        f"You enjoy having a good conversation. general {r['id']}",
        PERSONA_MINIMAL,
        f"You enjoy having a good conversation. topic {r['id']}",
    ], r["context"]
try:
    AttachPersonaPrompts(
        train_data=tk.Path(os.path.join(tmp, "train")),
        context_data=tk.Path(os.path.join(tmp, "ctx")),
        voice_codes=tk.Path(os.path.join(tmp, "voice")),
        level_weights={"minimal": 0.5, "general": 0.25, "topic": 0.25},
        prompt_data=[tk.Path(os.path.join(tmp, "prompts"))],
        prompt_set="t|t|assistant",
        per_read=True,
    )
    raise RuntimeError("unequal per_read weights were accepted")
except AssertionError:
    pass
print(f"[5] per_read attach: {len(att3)} rows carry all 3 levels as a list; unequal weights refused")
# ---- 6 ----
# with_other_side: the context job's OTHER channel must be exactly what the converter would have
# written had it picked that channel as the assistant (same window, same alignment format), and the
# attach step must carry that side's prompts, voice and alignments for the loader's channel swap.
ctx2 = run(
    PodcastWindowContext(
        train_data=tk.Path(os.path.join(tmp, "train")), dialogue_shard_dirs=dlg, window_select=RULE, with_other_side=True
    ),
    "ctx2",
)
fixed = {}
for ch in ("a", "b"):
    for r in run(
        PodcastCodesTrainData(shard_dirs=dlg, assistant_channel=ch, window_select=RULE, min_assistant_words=1),
        f"train_{ch}",
    ):
        fixed[r["id"]] = r["alignments"]
def f32(al):
    """The training corpus stores alignment times as float32 (the attach step casts to that schema)."""
    return [dict(w, start=float(np.float32(w["start"])), end=float(np.float32(w["end"]))) for w in al]


for r in ctx2:
    item, rest = r["id"].split("@")
    ch, wi = rest.split("#")
    twin = f"{item}@{'b' if ch == 'a' else 'a'}#{wi}"
    assert f32(r["user_alignments"]) == fixed[twin], f"{r['id']}: other side != converter's row {twin}"
    assert r["n_user_words"] == len(fixed[twin])
    assert r["user_label"] is not None and r["user_label"] != r["assistant_label"], r["id"]
labels = {(r["episode_id"], r["assistant_label"]) for r in ctx2} | {(r["episode_id"], r["user_label"]) for r in ctx2}
vcode = {k: (np.arange(8 * 50) + 1000 * i).astype(np.int16) for i, k in enumerate(sorted(labels))}
Dataset.from_list(
    [
        {"episode_id": e, "label": lb, "codes": v.tolist(), "n_frames": 50, "clip_sec": 4.0, "source_sec": 30.0}
        for (e, lb), v in vcode.items()
    ]
).save_to_disk(os.path.join(tmp, "voice_both"))
Dataset.from_list(
    [dict(p, prompt_set="t|t|assistant|other", text=p["text"].replace(" general ", " OTHER general ").replace(" topic ", " OTHER topic ")) for p in prows]
    + prows
).save_to_disk(os.path.join(tmp, "prompts_both"))


def attach_both(sub, **kw):
    return run(
        AttachPersonaPrompts(
            train_data=tk.Path(os.path.join(tmp, "train")),
            context_data=tk.Path(os.path.join(tmp, "ctx2")),
            voice_codes=tk.Path(os.path.join(tmp, "voice_both")),
            level_weights=w3,
            prompt_data=[tk.Path(os.path.join(tmp, "prompts_both"))],
            prompt_set="t|t|assistant",
            per_read=True,
            with_other_side=True,
            other_prompt_set="t|t|assistant|other",
            **kw,
        ),
        sub,
    )


att6 = attach_both("att6")
c2 = {r["id"]: r for r in ctx2}
assert len(att6) == len(train) and all(att6["other_ok"])
for r in att6:
    c = c2[r["id"]]
    assert r["context"][0].endswith(f"general {r['id']}") and "OTHER" not in r["context"][0]
    assert r["context_other"] == [
        f"You enjoy having a good conversation. OTHER general {r['id']}",
        PERSONA_MINIMAL,
        f"You enjoy having a good conversation. OTHER topic {r['id']}",
    ], r["context_other"]
    assert r["voice_codes"] == vcode[(c["episode_id"], c["assistant_label"])].tolist()
    assert r["voice_codes_other"] == vcode[(c["episode_id"], c["user_label"])].tolist()
    assert r["voice_codes"] != r["voice_codes_other"]
    twin = r["id"].replace("@a#", "@B#").replace("@b#", "@a#").replace("@B#", "@b#")
    assert r["alignments_other"] == fixed[twin] and r["alignments_other"] != r["alignments"]
att6s = attach_both("att6s", other_min_words=10**6)
assert len(att6s) == len(train) and not any(att6s["other_ok"]), "a too-short other side must not be swappable"
print(f"[6] other side: {len(ctx2)} windows == the converter's opposite-channel rows; attach carries its prompts, voice, alignments; short sides are not swappable")
# ---- 7 ----
# PersonaPromptGen(sides=("assistant", "other")): ONE job writes both prompt sets, each from ITS OWN
# channel's words. The LLM is replaced by a stub that echoes a tag of the transcript it was shown, so
# a side reading the wrong column (or both sides reading the same one) is caught.
import contextlib  # noqa: E402
import types  # noqa: E402

import openai  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.persona_prompts as pp  # noqa: E402


class _StubCompletions:
    def create(self, *, messages, **_kw):
        words = messages[0]["content"].split()
        tag = next((w for w in words if w[:1] in "ab" and "_" in w), "none").split("_")[0]
        body = json.dumps({"general": f"Talk about {tag}.", "topic": f"Discuss {tag} calmly.", "detailed": f"Discuss {tag} in depth."})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=body))])


class _StubClient:
    def __init__(self, **_kw):
        self.chat = types.SimpleNamespace(completions=_StubCompletions())


pp.vllm_server = lambda _name: contextlib.nullcontext("http://stub")
openai.OpenAI = _StubClient
g = pp.PersonaPromptGen(context_data=tk.Path(os.path.join(tmp, "ctx")), input_view="assistant", sides=("assistant", "other"))
g.out_summary = tk.Path(os.path.join(tmp, "gen_summary.json"))
gen = run(g, "gen_both")
assert g.prompt_set_for("assistant") == g.prompt_set and g.prompt_set_for("other") == g.prompt_set + "|other"
chan = {r["id"]: r["channel"] for r in ctx}
by = {}
for r in gen:
    by.setdefault((r["id"], r["prompt_set"]), {})[r["level"]] = r["text"]
assert len(by) == 2 * len(ctx), (len(by), len(ctx))
for rid, ch in chan.items():
    oth = "b" if ch == "a" else "a"
    a_gen = by[(rid, g.prompt_set_for("assistant"))]["general"]
    o_gen = by[(rid, g.prompt_set_for("other"))]["general"]
    assert a_gen.endswith(f"Talk about {ch}0.") or a_gen.endswith(f"Talk about {ch}{rid[2]}."), (rid, a_gen)
    assert o_gen.endswith(f"about {oth}{rid[2]}."), (rid, o_gen)
single = pp.PersonaPromptGen(context_data=tk.Path(os.path.join(tmp, "ctx")), input_view="assistant")
assert single._sis_hash() == pp.PersonaPromptGen(context_data=tk.Path(os.path.join(tmp, "ctx")), input_view="assistant", sides=("assistant",))._sis_hash()
assert single._sis_hash() != g._sis_hash()
print(f"[7] two-sided prompt job: {len(ctx)} windows x 2 sides, each side from its own channel; the default keeps its hash")
print("OK")
