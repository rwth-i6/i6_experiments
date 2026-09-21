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
print("OK")
