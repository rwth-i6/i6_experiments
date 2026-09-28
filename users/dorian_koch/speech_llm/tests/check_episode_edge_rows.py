"""Guard: the episode-edge filter (user, 2026-09-28: "exclude the first and last 2% of every episode ... for future
runs") drops exactly the windows touching an episode's edges, and a run with ``train_rows`` reads exactly those rows.

WHY. Both halves fail silently. A job that places windows on the wrong time base (slice-relative instead of episode
time, a wrong duration) or uses the midpoint instead of overlap still writes a plausible keep-list; a loader that
ignores ``train_rows`` (or maps its permutation wrong, or seeks wrong on resume) still trains, on the unfiltered corpus.
And a run WITHOUT the key must stay the stream every earlier run read.

Part ``job`` (setup ``.venv``, ~6 min: [3] reads p4's whole corpus):
  1. RULE. ``episode_edge_side``: a window reaching into [0, f*D) is head, into (D - f*D, D] tail; touching f*D or
     D - f*D exactly is kept; a window spanning both edges is head.
  2. JOB. ``EpisodeEdgeRows.run`` on a fake build (3 episodes of known duration, dialogue rows at known slice starts,
     windows relative to their row): ``rows.npy`` == the brute-force keep set from ABSOLUTE times (int64, ascending),
     summary counts == recount; head, tail and straddling windows occur (non-vacuous). Raises on a slice index given
     as the episode index, a window past its episode's end, a row without context. CAN FAIL: with
     ``episode_edge_side`` mutated to the MIDPOINT rule the job's keep-list differs from the brute force.
  3. REAL. The worker on p4's corpus (``AttachPersonaPrompts.DHvnX2ifpvad``, all rows, ``persona_v4/
     window_context_both``, ``podcast_episodes/jre_full``): keep + dropped partition the table, both sides dropped;
     writes ``keep.npy`` / ``dropped.npy`` to ``OUT`` for part ``loader``.
  4. RENDER. ``train_rows`` is rendered only when set (config = the plain one + that line).
  5. LAUNCHER. ``main()`` reads ``train_rows`` and hands it to BOTH loaders (``build_data_loader`` and
     ``build_knowledge_mix_loader``). (``check_launcher_config_reads.py`` checks the read is unconditional.)
Part ``loader`` (moshi_family venv, CPU, ~3 min; needs part ``job``'s ``OUT``), real corpora + tokenizer:
  6. ABSENT = TODAY. Without ``train_rows`` the plain loader reads rows ``default_rng(0).permutation(num_rows)`` in
     order (the documented stream of every earlier run, e.g. the 2-hour standard's "first 56,000 rows of the seed-0
     permutation"); ``train_rows`` = every row gives the same batches bit for bit.
  7. SUBSET. With ``keep.npy``: every row read (plain loader, and the knowledge mix's JRE items) is in keep, none in
     dropped, in the order ``keep[default_rng(0).permutation(len(keep))]``; the mix's knowledge items are those of
     the run without the key.
  8. RESUME. On a 20-row subset (2 batches of 8 per epoch): ``skip_batches=3`` reads the rows of batch 3 (epoch 1) of
     the unskipped stream; the mix with ``skip_batches=2`` reads the unskipped mix's JRE rows from batch 2 on.
  9. CAN FAIL. With ``load_row_subset`` mutated to return None (a loader ignoring the key) check 7 fails.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_episode_edge_rows.py job
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_episode_edge_rows.py loader'
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SETUP = "/home/tt201262/setups/2026-01-speech-llm"
os.chdir(SETUP)
PART = sys.argv[1] if len(sys.argv) > 1 else "job"
assert PART in ("job", "loader"), PART
W = f"{SETUP}/work/i6_experiments/users/dorian_koch/speech_llm"
JRE = f"{W}/persona_prompts/AttachPersonaPrompts.DHvnX2ifpvad/output/dataset"
KNOW = f"{W}/persona_knowledge/KnowledgeDialogueCodes.DMgw9Srb4bmg/output/dataset"
META = f"{W}/persona_knowledge/KnowledgeConcatMeta.Jq5Gwzdr2Ray/output/meta.npz"
OUT = "/hpcwork/p0023999/tt201262/guards/check_episode_edge_rows"
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


def job_part():
    sys.path[:0] = ["recipe", "recipe/sisyphus", "recipe/speech_llm/full_duplex"]
    os.environ.setdefault("CUDA_HOME", "/usr")
    from datasets import Dataset
    from sisyphus import tk

    import i6_experiments.users.dorian_koch.speech_llm.persona_prompts as pp
    from i6_experiments.users.dorian_koch.speech_llm import finetune as ft

    # ---- 1 ----
    side = pp.episode_edge_side
    ck(
        side(0.0, 10.0, 1000.0, 0.02) == "head"
        and side(19.9, 40.0, 1000.0, 0.02) == "head"
        and side(20.0, 40.0, 1000.0, 0.02) is None
        and side(950.0, 980.0, 1000.0, 0.02) is None
        and side(950.0, 980.1, 1000.0, 0.02) == "tail"
        and side(0.0, 1000.0, 1000.0, 0.02) == "head",
        "[1] overlap rule: reaching into an edge zone drops, touching its border keeps",
    )

    # ---- 2 ----
    tmp = tempfile.mkdtemp(prefix="check_episode_edge_rows_")
    dur = {"e1": 1000.0, "e2": 3000.0, "e3": 500.0}
    # dialogue rows at slice starts, windows [t0, t1) relative to the row
    items = [("e1#0", 0.0), ("e1#1", 400.0), ("e1#2", 960.0), ("e2#0", 30.0), ("e2#1", 1500.0), ("e2#2", 2900.0)]
    items += [("e3#0", 5.0), ("e3#1", 480.0)]
    rel = [(0.0, 15.0), (15.0, 35.0), (35.0, 60.0)]
    ctx, corpus, absolute = [], [], {}
    for it, s in items:
        for k, (a, b) in enumerate(rel):
            if s + b > dur[it.split("#")[0]]:
                continue
            rid = f"{it}@a#w{k}"
            ctx.append({"id": rid, "item_id": it, "episode_id": it.split("#")[0], "t0": a, "t1": b,
                        "assistant_text": "thanks for having me" if s > 900 else "hello", "user_text": ""})  # fmt: skip
            corpus.append({"id": rid, "duration": b - a})
            absolute[rid] = (it.split("#")[0], s + a, s + b)
    Dataset.from_list(ctx).save_to_disk(os.path.join(tmp, "ctx"))
    Dataset.from_list(corpus).save_to_disk(os.path.join(tmp, "data"))
    sl = os.path.join(tmp, "sl")
    os.makedirs(os.path.join(sl, "parts"))
    Dataset.from_list(
        [{"item_id": i, "episode_id": i.split("#")[0], "span_start_sec": s} for i, s in items]
    ).save_to_disk(os.path.join(sl, "parts", "part_00000"))
    idx = os.path.join(tmp, "episodes")
    os.makedirs(idx)
    with open(os.path.join(idx, "index.jsonl"), "w") as f:
        for e, d in dur.items():
            f.write(json.dumps({"item_id": e, "episode_id": e, "duration_sec": d}) + "\n")
    bad_idx = os.path.join(tmp, "slice_index.jsonl")
    with open(bad_idx, "w") as f:
        f.write(json.dumps({"item_id": "e1#0", "episode_id": "e1", "duration_sec": 100.0}) + "\n")

    def brute(f):
        keep, head, tail = [], [], []
        for i, r in enumerate(corpus):
            e, a, b = absolute[r["id"]]
            if a < f * dur[e]:
                head.append(i)
            elif b > dur[e] - f * dur[e]:
                tail.append(i)
            else:
                keep.append(i)
        return keep, head, tail

    def run_job(f=0.02, index=idx, data="data"):
        job = pp.EpisodeEdgeRows(
            data=tk.Path(os.path.join(tmp, data)),
            window_context=tk.Path(os.path.join(tmp, "ctx")),
            slice_dirs=[tk.Path(sl)],
            episode_index=tk.Path(index),
            edge_frac=f,
        )
        job.out_rows = tk.Path(os.path.join(tmp, "rows.npy"))
        job.out_summary = tk.Path(os.path.join(tmp, "summary.json"))
        job.run()
        return np.load(os.path.join(tmp, "rows.npy")), json.load(open(os.path.join(tmp, "summary.json")))

    for f in (0.02, 0.05):
        rows, summ = run_job(f)
        keep, head, tail = brute(f)
        straddle = [
            i
            for i in head + tail
            if (lambda e, a, b: a < f * dur[e] < b or a < dur[e] * (1 - f) < b)(*absolute[corpus[i]["id"]])
        ]
        ck(
            rows.dtype == np.int64 and rows.tolist() == keep,
            f"[2] f={f}: rows.npy == brute-force keep set from absolute episode time ({len(keep)}/{len(corpus)})",
        )
        ck(
            summ["kept"]["rows"] == len(keep)
            and summ["dropped_head"]["rows"] == len(head)
            and summ["dropped_tail"]["rows"] == len(tail)
            and abs(summ["kept"]["hours"] - sum(corpus[i]["duration"] for i in keep) / 3600) < 0.01,
            f"[2] f={f}: summary counts == recount",
        )
        ck(bool(head and tail and straddle), f"[2] f={f}: non-vacuous (head {len(head)}, tail {len(tail)}, "
           f"straddling {len(straddle)})")  # fmt: skip
    ck(bool(raises(lambda: run_job(index=bad_idx))), "[2] a slice index as the episode index raises")
    Dataset.from_list(corpus + [{"id": "e9#0@a#w0", "duration": 5.0}]).save_to_disk(os.path.join(tmp, "data_x"))
    ck(bool(raises(lambda: run_job(data="data_x"))), "[2] a row without window context raises")
    short = os.path.join(tmp, "short")
    os.makedirs(short)
    with open(os.path.join(short, "index.jsonl"), "w") as f:
        for e, d in dur.items():
            f.write(json.dumps({"item_id": e, "episode_id": e, "duration_sec": d / 2}) + "\n")
    ck(bool(raises(lambda: run_job(index=short))), "[2] a window past its episode's end (wrong durations) raises")
    orig = pp.episode_edge_side

    def midpoint(t0, t1, duration, edge_frac):
        m = 0.5 * (t0 + t1) / duration
        return "head" if m < edge_frac else ("tail" if m > 1 - edge_frac else None)

    pp.episode_edge_side = midpoint
    try:
        rows_mut, _ = run_job(0.05)
    finally:
        pp.episode_edge_side = orig
    ck(rows_mut.tolist() != brute(0.05)[0], "[2] CAN FAIL: the midpoint rule's keep-list differs from the brute force")

    # ---- 3 ----
    from datasets import load_from_disk

    ctx_real = os.path.realpath("output/persona_v4/window_context_both")
    info = os.path.join(os.path.dirname(os.path.dirname(ctx_real)), "info")
    slices = sorted(
        ln.split("INPUT: ", 1)[1].strip()
        for ln in open(info)
        if ln.startswith("INPUT: ") and "/PodcastDialogueSlice." in ln
    )
    summ, keep = pp.edge_rows_summary(
        JRE, ctx_real, slices, os.path.realpath("output/podcast_episodes/jre_full/codes_index"), 0.02
    )
    n = load_from_disk(JRE).num_rows
    dropped = sorted(set(range(n)) - set(keep))
    ck(
        keep == sorted(keep)
        and len(keep) + len(dropped) == n
        and summ["dropped_head"]["rows"] > 0
        and summ["dropped_tail"]["rows"] > 0
        and summ["dropped_head"]["rows"] + summ["dropped_tail"]["rows"] == len(dropped),
        f"[3] p4's corpus: keep {len(keep)} + dropped {len(dropped)} (head {summ['dropped_head']['rows']}, tail "
        f"{summ['dropped_tail']['rows']}) = {n} rows",
    )
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "keep.npy"), np.asarray(keep, dtype=np.int64))
    np.save(os.path.join(OUT, "dropped.npy"), np.asarray(dropped, dtype=np.int64))
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    # ---- 4 ----
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

    hp = {"emit_epad": True, "per_gpu_batch": 8, "grad_accum": 1, "loss_norm": "weight"}
    plain, withr = render(hp), render({**hp, "train_rows": _P("/x/rows.npy")})
    ck(
        "train_rows" not in plain
        and 'train_rows: "/x/rows.npy"' in withr.splitlines()
        and [ln for ln in withr.splitlines() if not ln.startswith("train_rows")] == plain.splitlines(),
        "[4] train_rows rendered only when set; the config is otherwise unchanged",
    )

    # ---- 5 ----
    src = Path("recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")
    reads = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get" and n.args
        and isinstance(n.args[0], ast.Constant) and n.args[0].value == "train_rows"
    ]  # fmt: skip
    calls = {
        n.func.id: {k.arg: ast.unparse(k.value) for k in n.keywords}
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("build_data_loader", "build_knowledge_mix_loader")
    }  # fmt: skip
    ck(
        len(reads) == 1 and all(c.get("train_rows") == "train_rows" for c in calls.values()) and len(calls) == 2,
        "[5] the launcher reads train_rows and hands it to both loaders",
        str({k: v.get("train_rows") for k, v in calls.items()}),
    )


def loader_part():
    import torch

    sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
    import moshi_family.personaplex.train_data as td
    from moshi_family.train_data_common import read_arrow_table

    keep, dropped = np.load(os.path.join(OUT, "keep.npy")), np.load(os.path.join(OUT, "dropped.npy"))
    n = read_arrow_table(JRE).num_rows
    ck(len(keep) + len(dropped) == n, f"[6] part job's lists cover p4's {n} rows")
    tok = td.PersonaPlexTokenizer(
        "nvidia/personaplex-7b-v1",
        device="cpu",
        cfg=td.PersonaPlexDataConfig(
            duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
        ),
    )
    B, N = 8, 3
    knobs = dict(
        system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True
    )
    concat = {"barge_lead_sec": [0.2, 0.8], "barge_min_answer_sec": 0.5, "barge_prob": 0.3, "dialogues": [2, 4],
              "gap_sec": [0.3, 1.5], "min_next_sec": 5.0, "tail_frames": 1}  # fmt: skip
    tmp = tempfile.mkdtemp(prefix="check_episode_edge_rows_")

    def npy(a, name):
        p = os.path.join(tmp, name)
        np.save(p, np.asarray(a, dtype=np.int64))
        return p

    def plain(meta=None, rows=None, skip=0, n_b=N):
        it = td.build_data_loader(JRE, tok, batch_size=B, seed=0, item_meta=meta, train_rows=rows, skip_batches=skip,
                                  **knobs)  # fmt: skip
        return [next(it) for _ in range(n_b)]

    def mix(meta=None, rows=None, skip=0, n_b=N):
        it = td.build_knowledge_mix_loader(JRE, KNOW, tok, batch_size=B, share=0.3, seed=0, knowledge_meta=META,
                                           knowledge_concat=concat, item_meta=meta, train_rows=rows,
                                           skip_batches=skip, **knobs)  # fmt: skip
        return [next(it) for _ in range(n_b)]

    def same(a, b):
        return len(a) == len(b) and all(
            len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y)) for x, y in zip(a, b)
        )

    def jre_rows(meta):
        return [m["row"] for m in meta if m["source"] == "primary"]

    # ---- 6 ----
    m0: list = []
    absent = plain(m0)
    today = np.random.default_rng(0).permutation(n)[: N * B].tolist()
    ck(jre_rows(m0) == today, "[6] without train_rows: rows = default_rng(0).permutation(num_rows), in order")
    ck(same(absent, plain(rows=npy(np.arange(n), "all.npy"))), "[6] train_rows = every row: same batches bit for bit")

    # ---- 7 ----
    keep_p = npy(keep, "keep.npy")
    kset, dset = set(keep.tolist()), set(dropped.tolist())
    want = keep[np.random.default_rng(0).permutation(len(keep))].tolist()

    def subset_errors(meta, n_items):
        got = jre_rows(meta)
        errs = []
        if not set(got) <= kset:
            errs.append(f"{len(set(got) - kset)} rows outside keep")
        if set(got) & dset:
            errs.append(f"{len(set(got) & dset)} dropped (edge) rows read")
        if got != want[: len(got)] or len(got) != n_items:
            errs.append("order != keep[permutation(len(keep))]")
        return errs

    m1: list = []
    plain(m1, rows=keep_p)
    e = subset_errors(m1, N * B)
    ck(not e, f"[7] plain loader: {N * B} rows read, all in keep, none dropped, in the subset's permutation", str(e))
    mk0: list = []
    mix_absent = mix(mk0)
    mk1: list = []
    mix(mk1, rows=keep_p)
    n_jre = len(jre_rows(mk1))
    e = subset_errors(mk1, n_jre)
    ck(not e and n_jre > 0, f"[7] knowledge mix: its {n_jre} JRE rows all in keep, none dropped, in order", str(e))
    kn = [(m["row"], [(p.row, p.src0) for p in m["plan"].pieces]) for m in mk1 if m["source"] == "knowledge"]
    kn0 = [(m["row"], [(p.row, p.src0) for p in m["plan"].pieces]) for m in mk0 if m["source"] == "knowledge"]
    ck(kn == kn0 and len(kn) > 0 and mix_absent, f"[7] knowledge mix: its {len(kn)} knowledge items are the run's "
       "without train_rows")  # fmt: skip

    # ---- 8 ----
    small = npy(keep[:20], "small.npy")
    # 2 batches per epoch (the 4 left over are built, then dropped: item_meta records them too, so the expected rows
    # come from the permutations, not from a longer run's entries)
    prng = np.random.default_rng(0)
    p0, p1 = keep[:20][prng.permutation(20)], keep[:20][prng.permutation(20)]
    mfull: list = []
    plain(mfull, rows=small, n_b=4)
    ck(
        jre_rows(mfull)[:16] == p0[:16].tolist() and jre_rows(mfull)[20:36] == p1[:16].tolist(),
        "[8] a 20-row subset: epoch 0 = its first permutation, epoch 1 its second",
    )
    mskip: list = []
    plain(mskip, rows=small, skip=3, n_b=1)
    ck(jre_rows(mskip) == p1[8:16].tolist(), "[8] plain resume: skip 3 -> batch 3 = epoch 1's second batch")
    mmf: list = []
    mix(mmf, rows=keep_p, n_b=4)
    mms: list = []
    mix(mms, rows=keep_p, skip=2, n_b=2)
    per = [jre_rows(mmf[b * B : (b + 1) * B]) for b in range(4)]
    ck(
        jre_rows(mms) == per[2] + per[3],
        "[8] mix resume: skip 2 -> the JRE rows of batches 2-3 of the unskipped mix",
    )

    # ---- 9 ----
    orig = td.load_row_subset
    td.load_row_subset = lambda *a, **k: None
    try:
        mm: list = []
        plain(mm, rows=keep_p)
    finally:
        td.load_row_subset = orig
    ck(
        bool(subset_errors(mm, N * B)),
        "[9] CAN FAIL: a loader ignoring train_rows is caught",
        str(subset_errors(mm, N * B)),
    )


job_part() if PART == "job" else loader_part()
print("ALL PASS" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
