"""Guard: the 2026-09-28 row criteria (``row_filters``: monologue cap, content lexicon, keep-list intersection, per-row
weights) compute what they claim, and an arm composes them without moving the default's job.

WHY. Each of these fails silently. A stretch that is not ``select_windows``' own (unmerged turns, not clipped to the
window) writes a plausible keep-list for a different cap; a lexicon matching substrings drops "analysis" and "Sussex";
a "trainable side" that ignores ``other_ok`` drops rows whose flagged side is never trained; an intersection in the
wrong order or a composition helper that builds a second edge job re-hashes the default; a renderer that drops
``train_row_weights`` trains unweighted. Nothing in a loss curve shows any of it.

Part ``job`` (setup ``.venv``, ~8 min: [3] reads p4's whole corpus):
  1. DEFINITION (drives the real ``select_windows``). On 300 seeded (seed 0) dialogue rows of p4's slices: tiling at
     cap 20 equals tiling at cap 30 up to the first cap-30 window whose ``longest_stretch`` > 20, and that window is
     not in the cap-20 tiling; every cap-20 window has stretch <= 20. CAN FAIL: an unmerged per-turn stretch as the
     definition contradicts ``select_windows`` on some row.
  2. MONOLOGUE JOB. ``MonologueRows.run`` on a fake build (hand-computed stretches: merged runs across a pause, a
     backchannel splitting a run, clipping at the window, a third speaker): ``rows.npy`` / ``stretch.npy`` /
     summary == brute force; holders right. Raises on a row missing from the context and a context row missing
     from the slices.
  3. REAL. The worker on p4's corpus (all rows): keep == stretch <= 20; every stretch <= 30.1 s (the corpus's own
     ``w20_60_t3_s30`` rule, up to frame rounding); caps monotone; writes ``keep20.npy`` / ``stretch.npy`` /
     ``weights_0.3_above20.npy`` to ``OUT`` for ``check_row_weights.py``.
  4. LEXICON JOB. ``LexiconRows.run`` on fake rows: whole words only ("Sussex", "analysis", "Dick Cheney" do not
     hit; "sexy", "blow job" do); ``sides`` trainable / assistant / either give the expected keep-lists (``other_ok``
     honoured); flags + summary recount. CAN FAIL: substring matching drops the clean row.
  5. INTERSECT / WEIGHTS. ``IntersectRows.run`` == the set intersection, unique drops right, empty intersection and a
     descending keep-list raise; ``ThresholdRowWeights.run`` puts the weight strictly ABOVE the threshold.
  6. RENDER. ``train_row_weights`` rendered only when set (config otherwise unchanged); refused with
     ``turn_tail_weight`` / ``floor_weight``.
  7. LAUNCHER. ``main()`` reads ``train_row_weights`` once, hands it to BOTH loaders, and ``loss_step`` passes
     ``sample_frame_weight``.
Part ``graph`` (setup ``.venv``, ~2 min, one config load):
  8. COMPOSITION on the real graph (p4's corpus): ``ppx_train_rows(edge only)`` IS the default's
     ``EpisodeEdgeRows`` output (same job id as a 09-28 probe arm's ``train_rows``); + monologue -> ``IntersectRows``
     of [edge, MonologueRows(p4, its context, its slices, 20)]; + lexicon -> three parts in that order; a probe arm
     given the composed ``train_rows`` carries exactly it (no second edge filter); ``ppx_row_weights`` ->
     ``ThresholdRowWeights(MonologueRows(...).out_stretch, 20, 0.3)``; a probe arm with it + the default knowledge mix
     builds. CAN FAIL: the probe without the key has no ``train_row_weights``.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_row_filters.py job
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_row_filters.py graph
"""

import ast
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SETUP = "/home/tt201262/setups/2026-01-speech-llm"
os.chdir(SETUP)
PART = sys.argv[1] if len(sys.argv) > 1 else "job"
assert PART in ("job", "graph"), PART
sys.path[:0] = ["recipe", "recipe/sisyphus"]
os.environ.setdefault("CUDA_HOME", "/usr")
W = f"{SETUP}/work/i6_experiments/users/dorian_koch/speech_llm"
JRE = f"{W}/persona_prompts/AttachPersonaPrompts.DHvnX2ifpvad/output/dataset"
OUT = "/hpcwork/p0023999/tt201262/guards/check_row_filters"
FAILED = []


def ck(ok, label, extra=""):
    print(("PASS " if ok else "FAIL ") + label + (f"  {extra}" if extra else ""), flush=True)
    if not ok:
        FAILED.append(label)


def raises(fn):
    try:
        fn()
    except (RuntimeError, ValueError, AssertionError) as e:
        return str(e)[:100]
    return None


def p4_slices():
    ctx_real = os.path.realpath("output/persona_v4/window_context_both")
    info = os.path.join(os.path.dirname(os.path.dirname(ctx_real)), "info")
    slices = sorted(
        ln.split("INPUT: ", 1)[1].strip()
        for ln in open(info)
        if ln.startswith("INPUT: ") and "/PodcastDialogueSlice." in ln
    )
    return ctx_real, slices


def job_part():
    from datasets import Dataset, load_from_disk
    from sisyphus import tk

    import i6_experiments.users.dorian_koch.speech_llm.row_filters as rf
    from i6_experiments.users.dorian_koch.speech_llm import finetune as ft
    from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import select_windows

    rule = dict(min_sec=20.0, max_sec=60.0, min_turns=3, max_stretch_sec=30.0, turn_min_sec=1.0)  # w20_60_t3_s30

    # ---- 1 ----
    ctx_real, slices = p4_slices()
    parts = [p for d in slices for p in rf._slice_parts(d)]
    rows = []
    for p in random.Random(0).sample(parts, 12):
        t = load_from_disk(p).select_columns(["item_id", "duration_sec", "turns_json"])
        rows += [(r["item_id"], float(r["duration_sec"]), json.loads(r["turns_json"])) for r in t]
    rows = random.Random(0).sample(rows, 300)

    def unmerged(turns, a, b):
        best, who = 0.0, None
        for t in turns:
            x = min(float(t["end"]), b) - max(float(t["start"]), a)
            if x > best:
                best, who = x, t["speaker"]
        return best, who

    def contradictions(stretch_of):
        n_bad = n_win = n_cut = 0
        for rid, dur, turns in rows:
            w30 = select_windows(turns, dur, **rule, seed_key=rid)
            w20 = select_windows(turns, dur, **{**rule, "max_stretch_sec": 20.0}, seed_key=rid)
            s30 = [stretch_of(turns, a, b)[0] for a, b in w30]
            k = next((i for i, s in enumerate(s30) if s > 20.0), len(w30))
            n_win += len(w30)
            n_cut += k < len(w30)
            ok = w20[:k] == w30[:k] and (k == len(w30) or w30[k] not in w20)
            ok = ok and all(stretch_of(turns, a, b)[0] <= 20.0 + 1e-9 for a, b in w20)
            n_bad += not ok
        return n_bad, n_win, n_cut

    real = lambda turns, a, b: rf.longest_stretch(rf.merged_runs(turns), a, b)  # noqa: E731
    n_bad, n_win, n_cut = contradictions(real)
    ck(n_bad == 0 and n_cut >= 20, "[1] longest_stretch == select_windows' stretch (cap 20 vs 30 tilings)",
       f"300 rows, {n_win} windows, {n_cut} rows with a > 20 s window")  # fmt: skip
    n_bad, _, _ = contradictions(unmerged)
    ck(n_bad > 0, "[1] CAN FAIL: an unmerged per-turn stretch contradicts select_windows", f"{n_bad} rows")

    # ---- 2 ----
    tmp = tempfile.mkdtemp(prefix="check_row_filters_")
    # dialogue row d1: A 0-8, A 9-20 (same run: 0-20), B backchannel 20-21, A 21-45, C (third) 46-58
    # dialogue row d2: B 0-12, A 12-14, B 14-40
    turns = {
        "d1": [
            {"speaker": "A", "start": 0.0, "end": 8.0},
            {"speaker": "A", "start": 9.0, "end": 20.0},
            {"speaker": "B", "start": 20.0, "end": 21.0},
            {"speaker": "A", "start": 21.0, "end": 45.0},
            {"speaker": "C", "start": 46.0, "end": 58.0},
        ],
        "d2": [
            {"speaker": "B", "start": 0.0, "end": 12.0},
            {"speaker": "A", "start": 12.0, "end": 14.0},
            {"speaker": "B", "start": 14.0, "end": 40.0},
        ],
    }
    # (id, item, t0, t1, assistant_label, user_label) -> expected (stretch, holder)
    wins = [
        ("d1@a#w0", "d1", 0.0, 22.0, "A", "B", 20.0, "assistant"),
        ("d1@b#w1", "d1", 5.0, 30.0, "B", "A", 15.0, "other"),  # 0-20 clipped to 5-20, 21-45 clipped to 21-30: 15 / 9
        ("d1@a#w2", "d1", 40.0, 60.0, "A", "B", 12.0, "third"),
        ("d2@a#w0", "d2", 0.0, 40.0, "A", "B", 26.0, "other"),
        ("d2@b#w1", "d2", 10.0, 30.0, "B", None, 16.0, "assistant"),
    ]
    Dataset.from_list(
        [
            {"id": w[0], "item_id": w[1], "t0": w[2], "t1": w[3], "assistant_label": w[4], "user_label": w[5]}
            for w in wins
        ]
    ).save_to_disk(os.path.join(tmp, "ctx"))
    Dataset.from_list([{"id": w[0], "duration": w[3] - w[2]} for w in wins]).save_to_disk(os.path.join(tmp, "data"))
    sl = os.path.join(tmp, "sl")
    os.makedirs(os.path.join(sl, "parts"))
    Dataset.from_list([{"item_id": k, "turns_json": json.dumps(v)} for k, v in turns.items()]).save_to_disk(
        os.path.join(sl, "parts", "part_00000")
    )

    def run_mono(cap, data="data", slice_dir=sl):
        job = rf.MonologueRows(
            data=tk.Path(os.path.join(tmp, data)),
            window_context=tk.Path(os.path.join(tmp, "ctx")),
            slice_dirs=[tk.Path(slice_dir)],
            max_stretch_sec=cap,
        )
        job.out_rows = tk.Path(os.path.join(tmp, "rows.npy"))
        job.out_stretch = tk.Path(os.path.join(tmp, "stretch.npy"))
        job.out_summary = tk.Path(os.path.join(tmp, "summary.json"))
        job.run()
        return (
            np.load(os.path.join(tmp, "rows.npy")),
            np.load(os.path.join(tmp, "stretch.npy")),
            json.load(open(os.path.join(tmp, "summary.json"))),
        )

    want = np.array([w[6] for w in wins], dtype=np.float32)
    for cap in (15.0, 20.0):
        keep, st, summ = run_mono(cap)
        ck(
            st.dtype == np.float32 and np.allclose(st, want) and keep.dtype == np.int64
            and keep.tolist() == [i for i, s in enumerate(want) if s <= cap]
            and summ["kept"]["rows"] == len(keep) and summ["caps"]["25"]["rows_kept"] == int((want <= 25).sum()),
            f"[2] cap {cap:g}: stretch.npy / rows.npy / summary == hand-computed ({keep.tolist()})",
        )  # fmt: skip
    st_map = rf.window_stretches(os.path.join(tmp, "ctx"), [sl])
    ck([st_map[w[0]][1] for w in wins] == [w[7] for w in wins], "[2] holder: stored assistant / other / third")
    Dataset.from_list(
        [{"id": w[0], "duration": 1.0} for w in wins] + [{"id": "zz@a#w0", "duration": 1.0}]
    ).save_to_disk(os.path.join(tmp, "data_x"))
    ck(bool(raises(lambda: run_mono(20.0, data="data_x"))), "[2] a row missing from the window context raises")
    sl2 = os.path.join(tmp, "sl2")
    os.makedirs(os.path.join(sl2, "parts"))
    Dataset.from_list([{"item_id": "d1", "turns_json": json.dumps(turns["d1"])}]).save_to_disk(
        os.path.join(sl2, "parts", "part_00000")
    )
    ck(bool(raises(lambda: run_mono(20.0, slice_dir=sl2))), "[2] a context row missing from the slices raises")

    # ---- 3 ----
    summ, keep, stretch = rf.monologue_rows_summary(JRE, ctx_real, slices, 20.0)
    n = load_from_disk(JRE).num_rows
    caps = [summ["caps"][k]["rows_kept"] for k in ("10", "15", "20", "25", "30")]
    ck(
        len(stretch) == n and keep.tolist() == np.flatnonzero(stretch <= 20.0).tolist() and 0 < len(keep) < n
        and float(stretch.max()) <= 30.1 and caps == sorted(caps),
        f"[3] p4's corpus: keep {len(keep)} of {n} at 20 s, max stretch {float(stretch.max()):.2f} s <= 30.1, "
        f"caps monotone {caps}",
    )  # fmt: skip
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "keep20.npy"), keep)
    np.save(os.path.join(OUT, "stretch.npy"), stretch)
    np.save(os.path.join(OUT, "weights_0.3_above20.npy"), rf.threshold_weights(stretch, 20.0, 0.3))

    # ---- 4 ----
    def al(text, t0=0.0):
        return [
            {"text": w, "start": t0 + k, "end": t0 + k + 0.5, "speaker": "assistant"}
            for k, w in enumerate(text.split())
        ]

    lrows = [
        ("we talked about sex yesterday", "okay", True),  # 0: assistant sexual
        ("hello there", "that was sexy stuff", True),  # 1: other sexual, trainable
        ("hello there", "watch porn", False),  # 2: other sexual, NOT trainable (other_ok False)
        ("Sussex analysis, Dick Cheney, Essex.", "Middlesex analysts", True),  # 3: clean (whole words only)
        ("fuck this", "yeah", True),  # 4: profanity only
        ("a blow job joke", "no", False),  # 5: two-word sexual term, assistant
    ]
    Dataset.from_list(
        [{"id": f"r{i}", "duration": 10.0, "alignments": al(a), "alignments_other": al(b), "other_ok": ok}
         for i, (a, b, ok) in enumerate(lrows)]
    ).save_to_disk(os.path.join(tmp, "lex"))  # fmt: skip

    def run_lex(cats=("sexual",), sides="trainable"):
        job = rf.LexiconRows(data=tk.Path(os.path.join(tmp, "lex")), categories=cats, sides=sides)
        job.out_rows = tk.Path(os.path.join(tmp, "lrows.npy"))
        job.out_summary = tk.Path(os.path.join(tmp, "lsummary.json"))
        job.out_flags = tk.Path(os.path.join(tmp, "lflags.jsonl"))
        job.run()
        flags = [json.loads(x) for x in open(os.path.join(tmp, "lflags.jsonl"))]
        return (
            np.load(os.path.join(tmp, "lrows.npy")).tolist(),
            json.load(open(os.path.join(tmp, "lsummary.json"))),
            flags,
        )

    k_tr, s_tr, flags = run_lex()
    k_as, _, _ = run_lex(sides="assistant")
    k_ei, _, _ = run_lex(sides="either")
    k_pr, _, _ = run_lex(cats=("profanity", "sexual"))
    ck(k_tr == [2, 3, 4], "[4] sexual, trainable sides: drops 0, 1, 5 (2's flagged side is not trainable)", str(k_tr))
    ck(k_as == [1, 2, 3, 4], "[4] sexual, stored assistant only: drops 0, 5", str(k_as))
    ck(k_ei == [3, 4], "[4] sexual, either side: drops 0, 1, 2, 5", str(k_ei))
    ck(k_pr == [2, 3], "[4] profanity + sexual: also drops 4", str(k_pr))
    ck(
        [f["row"] for f in flags] == [0, 1, 2, 4, 5] and flags[0]["hits"]["assistant"]["sexual"] == 1
        and s_tr["per_side"]["other"]["sexual"]["windows"] == 2 and s_tr["dropped"]["rows"] == 3
        and s_tr["per_side"]["assistant"]["words"] == sum(len(a.split()) for a, _, _ in lrows),
        "[4] flags + summary recount (per side windows / words, dropped)",
    )  # fmt: skip
    orig = rf.lexicon_patterns
    import re

    rf.lexicon_patterns = lambda name: {c: re.compile("|".join(ws)) for c, ws in rf.SAFETY_LEXICONS[name].items()}
    try:
        k_sub, _, _ = run_lex()
    finally:
        rf.lexicon_patterns = orig
    ck(3 not in k_sub, "[4] CAN FAIL: substring matching drops the clean row 3", str(k_sub))

    # ---- 5 ----
    for name, arr in (
        ("a", [0, 2, 4, 6, 8]),
        ("b", [2, 3, 4, 8, 9]),
        ("c", [4, 8, 10]),
        ("d", [1, 3]),
        ("bad", [3, 1]),
    ):
        np.save(os.path.join(tmp, f"k_{name}.npy"), np.asarray(arr, dtype=np.int64))

    def run_int(names):
        job = rf.IntersectRows(rows=[tk.Path(os.path.join(tmp, f"k_{x}.npy")) for x in names])
        job.out_rows = tk.Path(os.path.join(tmp, "irows.npy"))
        job.out_summary = tk.Path(os.path.join(tmp, "isummary.json"))
        job.run()
        return np.load(os.path.join(tmp, "irows.npy")), json.load(open(os.path.join(tmp, "isummary.json")))

    got, summ_i = run_int(["a", "b", "c"])
    ck(
        got.dtype == np.int64 and got.tolist() == [4, 8] and summ_i["unique_drops"] == [0, 0, 1],
        "[5] IntersectRows == set intersection; unique drops recount", f"{got.tolist()} {summ_i['unique_drops']}"
    )  # fmt: skip
    got2, summ2 = run_int(["a", "b"])
    ck(got2.tolist() == [2, 4, 8] and summ2["unique_drops"] == [2, 2], "[5] two lists: drops each alone", str(summ2))
    ck(bool(raises(lambda: run_int(["a", "d"]))), "[5] an empty intersection raises")
    ck(bool(raises(lambda: run_int(["a", "bad"]))), "[5] a descending keep-list raises")
    np.save(os.path.join(tmp, "vals.npy"), np.asarray([10.0, 20.0, 20.01, 30.0], dtype=np.float32))
    job = rf.ThresholdRowWeights(values=tk.Path(os.path.join(tmp, "vals.npy")), above=20.0, weight=0.3)
    job.out_weights = tk.Path(os.path.join(tmp, "w.npy"))
    job.out_summary = tk.Path(os.path.join(tmp, "wsum.json"))
    job.run()
    w = np.load(os.path.join(tmp, "w.npy"))
    ck(w.dtype == np.float32 and w.tolist() == [1.0, 1.0, np.float32(0.3), np.float32(0.3)],
       "[5] ThresholdRowWeights: weight strictly above the threshold", str(w.tolist()))  # fmt: skip
    ck(bool(raises(lambda: rf.ThresholdRowWeights(values=tk.Path("/x"), above=20.0, weight=1.0))),
       "[5] a weight >= 1 is refused (a down-weight)")  # fmt: skip

    # ---- 6 ----
    lora = ft.PERSONAPLEX_LIB_LORA_ADAPTER

    class _P:
        def __init__(self, p):
            self.p = p

        def get(self):
            return self.p

        def get_path(self):
            return self.p

    def render(hp):
        job = SimpleNamespace(
            train_data=_P("/corpus"), eval_data=None, out_rundir=_P("/tmp/run"), out_config=_P("/tmp/c.yaml"), seed=0,
            duration_sec=60, audio_jitter_sec=0.0, lora_rank=128, hparams=hp, adapter=lora,
        )  # fmt: skip
        return lora.render_config(job, lora.batch_size, 1000)

    hp = {"emit_epad": True, "onset_floor": True, "per_gpu_batch": 8, "grad_accum": 1, "loss_norm": "weight"}
    plain, withw = render(hp), render({**hp, "train_row_weights": _P("/x/weights.npy")})
    ck(
        "train_row_weights" not in plain
        and 'train_row_weights: "/x/weights.npy"' in withw.splitlines()
        and [ln for ln in withw.splitlines() if not ln.startswith("train_row_weights")] == plain.splitlines(),
        "[6] train_row_weights rendered only when set; the config is otherwise unchanged",
    )
    ck(bool(raises(lambda: render({**hp, "train_row_weights": _P("/w"), "turn_tail_weight": 0.3}))),
       "[6] refused together with turn_tail_weight")  # fmt: skip
    ck(bool(raises(lambda: render({**hp, "train_row_weights": _P("/w"), "floor_weight": 0.5}))),
       "[6] refused together with floor_weight")  # fmt: skip

    # ---- 7 ----
    src = Path("recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")
    reads = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get" and n.args
        and isinstance(n.args[0], ast.Constant) and n.args[0].value == "train_row_weights"
    ]  # fmt: skip
    calls = {
        n.func.id: {k.arg: ast.unparse(k.value) for k in n.keywords}
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("build_data_loader", "build_knowledge_mix_loader")
    }  # fmt: skip
    ls = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "loss_step")
    loss_kw = {
        k.arg: ast.unparse(k.value)
        for n in ast.walk(ls)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "personaplex_loss"
        for k in n.keywords
    }
    ck(
        len(reads) == 1
        and all(c.get("train_row_weights") == "train_row_weights" for c in calls.values()) and len(calls) == 2
        and loss_kw.get("sample_frame_weight") == "sample_frame_weight",
        "[7] main() reads train_row_weights once, passes it to both loaders; loss_step passes sample_frame_weight",
    )  # fmt: skip


def graph_part():
    logging.disable(logging.WARNING)
    from sisyphus import tk
    from sisyphus.loader import config_manager

    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])

    def mod(suffix):
        (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
        return m

    tp, runs, ab2h = mod(".track_persona"), mod(".doriank.runs"), mod(".track_persona_ab2h")
    data = ab2h._STATE["train_data"]
    src = tp.PPX_EDGE_SOURCES[data.get_path()]
    D = runs.POLICY_LATEST

    def arm(tag, extra=None, policy=D):
        return tp._ppx_arm(f"_guard_rf_{tag}", data, extra or {}, hparams=ab2h.AB2H_HPARAMS, steps=100,
                           benchmark=False, policy=policy)  # fmt: skip

    def jid(p):
        return p.creator._sis_id()

    edge_default = arm("default").hparams["train_rows"]
    only_edge = tp.ppx_train_rows(data)
    ck(jid(only_edge) == jid(edge_default) and only_edge.get_path() == edge_default.get_path(),
       "[8] ppx_train_rows(edge only) IS the 09-28 default's EpisodeEdgeRows output", jid(only_edge))  # fmt: skip
    em = tp.ppx_train_rows(data, max_stretch_sec=20.0)
    ij = em.creator
    mono = ij.rows[1].creator
    ck(
        type(ij).__name__ == "IntersectRows" and len(ij.rows) == 2 and jid(ij.rows[0]) == jid(edge_default)
        and type(mono).__name__ == "MonologueRows" and mono.max_stretch_sec == 20.0
        and mono.data.get_path() == data.get_path()
        and mono.window_context.get_path() == src["window_context"].get_path()
        and [p.get_path() for p in mono.slice_dirs] == [p.get_path() for p in src["slice_dirs"]],
        "[8] edge + monologue 20 s = IntersectRows([edge, MonologueRows(p4, its context, its slices, 20)])",
    )  # fmt: skip
    eml = tp.ppx_train_rows(data, max_stretch_sec=20.0, lexicon_categories=("sexual",)).creator
    ck(
        [type(p.creator).__name__ for p in eml.rows] == ["EpisodeEdgeRows", "MonologueRows", "LexiconRows"]
        and eml.rows[2].creator.categories == ("sexual",) and eml.rows[2].creator.sides == "trainable",
        "[8] + lexicon: three parts in the fixed order edge, monologue, lexicon (sexual, trainable sides)",
    )  # fmt: skip
    hp = arm("composed", {"train_rows": em}).hparams
    ck(
        hp["train_rows"] == em and hp["train_rows"].get_path() == em.get_path(),
        "[8] a probe arm given the composed train_rows carries exactly it (no extra edge)",
    )
    rw = tp.ppx_row_weights(data)
    tw = rw.creator
    ck(
        type(tw).__name__ == "ThresholdRowWeights" and tw.above == 20.0 and tw.weight == 0.3
        and jid(tw.values) == jid(mono.out_rows),
        "[8] ppx_row_weights = ThresholdRowWeights(MonologueRows(p4, 20).out_stretch, above 20, weight 0.3)",
    )  # fmt: skip
    hpw = arm("weights", {"train_row_weights": rw}).hparams
    ck(hpw.get("train_row_weights") == rw and hpw.get("knowledge_data") is not None,
       "[8] a probe arm with train_row_weights + the default knowledge mix builds")  # fmt: skip
    ck("train_row_weights" not in arm("default2").hparams, "[8] CAN FAIL: the probe without the key has none")
    ck(bool(raises(lambda: tp.ppx_train_rows(tk.Path("/nonexistent/corpus"), edge=False, max_stretch_sec=20.0))),
       "[8] an unregistered corpus raises")  # fmt: skip


job_part() if PART == "job" else graph_part()
print(f"\n{'FAILED: ' + '; '.join(FAILED) if FAILED else 'ALL PASS'} ({PART})")
sys.exit(1 if FAILED else 0)
