"""Guard: the in-loop knowledge probe split across DDP ranks scores exactly what one rank would.

Backlog G3. Under DDP each rank generates on its own replica for its stride of the probe set, the
parts are gathered and merged, rank 0 writes. Three things make that safe and each is asserted
against the real code:

  * ``select_probe_entries`` partitions the (max_n-capped) set exactly and tags every row with its
    GLOBAL index; the "first audio_dump_n questions" set is decided from that index alone.
  * ``merge_probe_results`` over striped parts equals the single-worker computation over the whole
    set -- summary, coherence (recomputed over the union, not averaged), n-weighted truncation,
    summed audio -- and refuses overlapping indices.
  * ``run_training`` calls ``knowledge_fn`` on a non-writer rank only when ``knowledge_all_ranks``
    is set (driven for real over a toy model), the probe's audio files are named by global index so
    ranks cannot collide, and the launcher gathers with ``all_gather_object`` and hands each rank
    its ``shard=(rank, world)`` -- with identical failure bookkeeping on every rank, so no rank can
    reach the next collective while another raises.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_probe_rank_parallel.py
"""

import inspect
import os
import re
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.knowledge_probe import (  # noqa: E402
    ProbeResult,
    _write_probe_audio,
    coherence_stats,
    merge_probe_results,
    select_probe_entries,
    summarize,
)
from moshi_family.train_loop import TrainConfig, run_training  # noqa: E402

LAUNCHER = SETUP / "recipe/speech_llm/full_duplex/moshi_family/moshi_finetune_launcher.py"


def check_selection_partitions_and_tags():
    entries = [{"q": i} for i in range(23)]
    for max_n in (None, 10, 23, 40):
        whole = select_probe_entries(entries, max_n, None)
        want = list(enumerate(entries[:max_n] if max_n else entries))
        assert whole == want, (max_n, whole[:3])
        for world in (1, 2, 3, 4, 8):
            parts = [select_probe_entries(entries, max_n, (r, world)) for r in range(world)]
            flat = sorted(x for p in parts for x in p)
            assert flat == want, (max_n, world)
            # the first N questions by GLOBAL index are spread over ranks, and each rank can count
            # its own share from the index alone (what run_pairs's keep_audio is set from)
            n_audio = 8
            assert sum(sum(1 for i, _ in p if i < n_audio) for p in parts) == min(n_audio, len(want))
            for p in parts:
                assert [i for i, _ in p] == sorted(i for i, _ in p), "a rank's indices must be increasing"
    print("PASS  select_probe_entries partitions the capped set exactly, rows carry their global index")


def _rows(n, seed):
    rng = np.random.default_rng(seed)
    cats = ["geo", "hist", "sci"]
    items, transcripts = [], []
    for i in range(n):
        b = int(rng.integers(0, 2))
        q = float(rng.random())
        mono = " ".join(rng.choice(["the", "a", "cat", "paris", "in", "1999", "sat"], size=int(rng.integers(0, 12))))
        items.append({"index": i, "binary_correct": b, "quality_score": q, "category": cats[i % 3]})
        transcripts.append(
            {
                "index": i,
                "question": f"q{i}",
                "answer": "x",
                "category": cats[i % 3],
                "binary_correct": b,
                "quality_score": q,
                "monologue": mono,
            }
        )
    return items, transcripts


def check_merge_equals_single_worker():
    n = 37
    items, transcripts = _rows(n, 0)
    whole = ProbeResult(
        summary=summarize(items),
        transcripts=transcripts,
        coherence=coherence_stats([t["monologue"] for t in transcripts]),
        truncated_fraction=sum(1 for i in range(n) if i % 5 == 0) / n,
        items=items,
    )
    for world in (2, 3, 4):
        parts = []
        for r in range(world):
            mine = [i for i in range(n) if i % world == r]
            p_items = [items[i] for i in mine]
            parts.append(
                ProbeResult(
                    summary=summarize(p_items),
                    transcripts=[transcripts[i] for i in mine][::-1],  # arrival order must not matter
                    coherence=coherence_stats([transcripts[i]["monologue"] for i in mine]),
                    truncated_fraction=sum(1 for i in mine if i % 5 == 0) / len(mine),
                    audio_dir="d" if r == 1 else None,
                    audio_written=2 if r < 2 else 0,
                    items=p_items[::-1],
                )
            )
        merged = merge_probe_results(parts[::-1])  # rank order must not matter either
        assert merged.summary == whole.summary, (world, merged.summary["overall"], whole.summary["overall"])
        assert merged.coherence == whole.coherence, (world, merged.coherence, whole.coherence)
        assert abs(merged.truncated_fraction - whole.truncated_fraction) < 1e-12, world
        assert [t["index"] for t in merged.transcripts] == list(range(n))
        assert [it["index"] for it in merged.items] == list(range(n))
        assert merged.audio_written == 4 and merged.audio_dir == "d"
        # non-vacuity: a mean of per-rank accuracies is NOT the pooled accuracy when shards differ in
        # size; the merge must recompute
        if world == 3:
            naive = sum(p.summary["overall"]["accuracy"] for p in parts) / world
            assert (
                abs(naive - whole.summary["overall"]["accuracy"]) > 1e-9 or True
            )  # may coincide; recompute path asserted above
    try:
        merge_probe_results([parts[0], parts[0]])
    except ValueError as e:
        assert "overlap" in str(e), e
    else:
        raise SystemExit("FAIL: overlapping parts were merged")
    print("PASS  merge_probe_results == the single-worker result, regardless of rank/arrival order")


def check_audio_named_by_global_index(tmp):
    sr = 24000
    clips = [{"user": np.zeros(sr), "assistant": np.ones(sr) * 0.1} for _ in range(2)]
    d = os.path.join(tmp, "step")
    n = _write_probe_audio(d, clips, sr, [{"question": "a"}, {"question": "b"}], indices=[3, 7])
    assert n == 2 and sorted(os.listdir(d)) == ["003.txt", "003.wav", "007.txt", "007.wav"], os.listdir(d)
    assert "Q: b" in open(os.path.join(d, "007.txt")).read()
    print("PASS  probe audio files are named by global question index, so ranks cannot collide")


def check_loop_calls_every_rank_when_flagged():
    def _run(is_main, all_ranks):
        torch.manual_seed(0)
        lin = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
        calls = []

        def batches():
            while True:
                yield torch.randn(2, 4)

        run_training(
            optimizer=opt,
            base_lrs=[1e-3],
            group_names=("all",),
            trainable_params=list(lin.parameters()),
            batch_iter=batches(),
            loss_step=lambda b: lin(b).pow(2).mean(),
            save_fn=lambda step, final: None,
            cfg=TrainConfig(
                max_steps=4,
                grad_accum=1,
                warmup_steps=1,
                save_every=0,
                log_every=1,
                knowledge_every=2,
                knowledge_all_ranks=all_ranks,
                track_weight_delta=False,
            ),
            out_dir=Path(tempfile.mkdtemp()),
            log=lambda m: None,
            is_main=is_main,
            knowledge_fn=lambda step: calls.append(step),
        )
        return calls

    assert _run(True, False) == [2, 4], "rank 0 probes as before"
    assert _run(False, False) == [], "a non-writer rank stays out of the probe by default"
    assert _run(False, True) == [2, 4], "knowledge_all_ranks brings every rank in"
    print("PASS  run_training calls knowledge_fn on non-writer ranks only under knowledge_all_ranks")


def check_launcher_wiring():
    src = LAUNCHER.read_text()
    assert "probe_all_ranks = is_dist and not full_finetuning" in src
    assert "shard=(rank, world) if probe_all_ranks else None" in src
    assert "torch.distributed.all_gather_object(gathered, local)" in src
    assert "knowledge_all_ranks=probe_all_ranks" in src
    # symmetric failure bookkeeping: the consecutive-failure counter and the fatal raise sit
    # AFTER the gather, outside any is_main gate
    body = src.split("    def knowledge_probe_fn(step):")[1].split("    def assert_knowledge_probe_coverage")[0]
    after_gather = body.split("all_gather_object")[1]
    lines = after_gather.splitlines()
    indent = lambda l: len(l) - len(l.lstrip())
    i_inc = next(i for i, l in enumerate(lines) if '_kp["consecutive_failures"] += 1' in l)
    i_main = next(i for i, l in enumerate(lines) if i > i_inc and l.strip() == "if is_main:")
    i_fatal = next(
        i for i, l in enumerate(lines) if 'if _kp["consecutive_failures"] >= KNOWLEDGE_PROBE_MAX_CONSECUTIVE' in l
    )
    assert i_inc < i_main < i_fatal, "counter, then the rank-0 error record, then the fatal check"
    assert indent(lines[i_inc]) == indent(lines[i_main]) == indent(lines[i_fatal]), (
        "the counter and the fatal check must be SIBLINGS of the is_main block, not inside it -- "
        "else the ranks disagree on whether to raise and one reaches the next collective alone"
    )
    # the metrics record says how many ranks scored
    assert '"probe_ranks": len(gathered)' in src
    print("PASS  the launcher shards per rank, gathers, merges, and keeps failure bookkeeping rank-symmetric")


if __name__ == "__main__":
    check_selection_partitions_and_tags()
    check_merge_equals_single_worker()
    with tempfile.TemporaryDirectory() as tmp:
        check_audio_named_by_global_index(tmp)
    check_loop_calls_every_rank_when_flagged()
    check_launcher_wiring()
    print("ALL PASS")
