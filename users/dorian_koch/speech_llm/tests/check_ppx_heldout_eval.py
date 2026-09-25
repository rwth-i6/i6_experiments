"""Guard: p6's training data never contains its held-out audio, and its launcher reports the SHARED held-out loss
in the training loop (p6, 2026-09-25).

WHY. p6 is the first PersonaPlex run with a held-out split and an in-loop held-out loss. Both fail silently: a
window-level split or an exclusion that misses overlapping windows of another slicing trains on the "held-out"
audio and the loss reads as generalisation; an eval that is not wired (the A17 lesson: ``holdout_every`` was
"guarded" for weeks and never produced a number) or not called at the cadence trains blind.

  1. SPLIT. ``split_train_episodes`` is deterministic, nested (fraction 0.5 subset of 1.0), never trains a forced
     episode, and realises ~the fraction on 2,000 synthetic episodes. CAN FAIL: a window-level split (the same
     draw keyed by row id) puts episodes on both sides and is caught.
  2. OVERLAP. ``overlapping_ids``: partial overlap / containment -> dropped; touching ends, other episode -> kept.
  3. JOB. ``EpisodeSplit.run`` on a fake corpus with fake window contexts + dialogue slices (spans in episode time
     = row ``span_start_sec`` + window ``t0``/``t1``) and an "evaluation set" from ANOTHER slicing: no kept row is
     from a held-out episode, no kept row overlaps an evaluation window (brute force from the raw spans), some rows
     WERE dropped for overlap (non-vacuous), exactly the rest is kept, row order kept, summary counts == recount.
  4. RENDER. ``eval_data`` / ``eval_every`` render only when the job has ``eval_data`` (the config is otherwise
     byte-identical), and a half-set pair raises.
  5. LAUNCHER. reads both keys unconditionally in ``main``; the eval is ``heldout_loss.evaluate_set`` with
     ``heldout_loss.loss_settings(cfg)`` of THIS config (the shared code the PersonaHeldoutLoss job runs; its own
     parity with ``loss_step`` is ``check_heldout_loss.py``); step 0 evaluated fatally before training; ``eval_every``
     reaches ``TrainConfig`` and ``eval_fn`` reaches ``run_training``.
  6. CADENCE. The real ``run_training`` calls ``eval_fn`` at exactly the multiples of ``eval_every``. CAN FAIL:
     ``eval_every=0`` never calls it.

Recipe venv, login node, ~15 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_heldout_eval.py
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
sys.path.insert(0, "recipe/speech_llm/full_duplex")
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from sisyphus import tk  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.persona_prompts as pp  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import finetune as ft  # noqa: E402
from moshi_family.train_loop import TrainConfig, run_training  # noqa: E402

SETUP = Path(os.getcwd())
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


# ---- 1 ----
print("[1] split")
eps = [f"rss_{i:016d}" for i in range(2000)]
forced = set(eps[:50])
half = pp.split_train_episodes(eps, fraction=0.5, seed=0, heldout=forced)
full = pp.split_train_episodes(eps, fraction=1.0, seed=0, heldout=forced)
check(half == pp.split_train_episodes(list(reversed(eps)), fraction=0.5, seed=0, heldout=forced), "deterministic, order-free")
check(half <= full and full == set(eps) - forced, "half is nested in all; all = every non-forced episode")
check(not (half | full) & forced, "a forced episode never trains")
check(abs(len(half) / (len(eps) - len(forced)) - 0.5) < 0.03, f"realised share {len(half) / 1950:.3f} ~ 0.5")
rows = [f"{e}#{k}@a#w{w}" for e in eps[:300] for k in range(3) for w in range(2)]


def leaks(train_rows):
    tr = {pp.episode_of(r) for r in train_rows}
    ho = {pp.episode_of(r) for r in rows if r not in set(train_rows)}
    return tr & ho


real_rows = [r for r in rows if pp.episode_of(r) in pp.split_train_episodes(eps[:300], fraction=0.5, seed=0)]
mutant_rows = [r for r in rows if pp._seeded_unit(0, f"split|{r}") < 0.5]  # the window-level split
check(not leaks(real_rows), "the episode split leaves no episode on both sides")
check(bool(leaks(mutant_rows)), f"(can fail) a window-level split leaks {len(leaks(mutant_rows))} episodes to both sides")

# ---- 2 ----
print("[2] overlap")
spans = {
    "in": ("e1", 10.0, 20.0),
    "part": ("e1", 25.0, 40.0),
    "touch": ("e1", 50.0, 60.0),
    "other_ep": ("e2", 10.0, 20.0),
    "contains": ("e1", 0.0, 100.0),
}
got = pp.overlapping_ids(spans, [("e1", 12.0, 18.0), ("e1", 35.0, 50.0), ("e3", 10.0, 20.0)])
check(got == {"in", "part", "contains"}, f"overlap rule: {sorted(got)}")

# ---- 3 ----
print("[3] EpisodeSplit.run")
tmp = tempfile.mkdtemp(prefix="check_ppx_heldout_eval_")


def slices_dir(name, items):
    d = os.path.join(tmp, name)
    os.makedirs(os.path.join(d, "parts"))
    Dataset.from_list(
        [{"item_id": i, "episode_id": i.split("#")[0], "span_start_sec": s} for i, s in items]
    ).save_to_disk(os.path.join(d, "parts", "part0"))
    return tk.Path(d)


ep10 = eps[:10]
items = [(f"{e}#{k}", 100.0 * k) for e in ep10 for k in range(3)]  # 3 dialogue rows per episode
ctx_rows, corpus = [], []
for it, _ in items:
    for w in range(2):  # 2 windows per row: [0, 30) and [30, 60) within the row
        rid = f"{it}@a#w{w}"
        ctx_rows.append({"id": rid, "item_id": it, "episode_id": it.split("#")[0], "t0": 30.0 * w, "t1": 30.0 * w + 30.0})
        corpus.append({"id": rid, "duration": 30.0, "x": len(corpus)})
Dataset.from_list(ctx_rows).save_to_disk(os.path.join(tmp, "ctx"))
Dataset.from_list(corpus).save_to_disk(os.path.join(tmp, "data"))
# the evaluation set comes from another slicing of episodes 5-9: its rows start 10 s later
ex_items = [(f"{e}#x{k}", 100.0 * k + 10.0) for e in ep10[5:] for k in range(3)]
ex_ctx = [{"id": f"{it}@b#w0", "item_id": it, "episode_id": it.split("#")[0], "t0": 5.0, "t1": 12.0} for it, _ in ex_items]
Dataset.from_list(ex_ctx).save_to_disk(os.path.join(tmp, "ex_ctx"))
ex_set = [r["id"] for r in ex_ctx][::2]  # every other one is in the set
Dataset.from_list([{"id": r} for r in ex_set]).save_to_disk(os.path.join(tmp, "ex_set"))
job = pp.EpisodeSplit(
    data=tk.Path(os.path.join(tmp, "data")),
    train_fraction=1.0,
    heldout_episodes=[ep10[0], ep10[1]],
    window_context=tk.Path(os.path.join(tmp, "ctx")),
    slice_dirs=[slices_dir("sl", items)],
    exclude_windows=tk.Path(os.path.join(tmp, "ex_set")),
    exclude_context=tk.Path(os.path.join(tmp, "ex_ctx")),
    exclude_slice_dirs=[slices_dir("ex_sl", ex_items)],
)
job.out_train = tk.Path(os.path.join(tmp, "out_train"))
job.out_episodes = tk.Path(os.path.join(tmp, "episodes.json"))
job.out_summary = tk.Path(os.path.join(tmp, "summary.json"))
job.run()
kept = load_from_disk(os.path.join(tmp, "out_train"))
kept_ids = kept["id"]
start = dict(items)
abs_span = {r["id"]: (r["episode_id"], start[r["item_id"]] + r["t0"], start[r["item_id"]] + r["t1"]) for r in ctx_rows}
ex_start = dict(ex_items)
ex_abs = [
    (r["episode_id"], ex_start[r["item_id"]] + r["t0"], ex_start[r["item_id"]] + r["t1"])
    for r in ex_ctx
    if r["id"] in ex_set
]
brute = {i for i, (e, a, b) in abs_span.items() if any(e == f and a < y and x < b for f, x, y in ex_abs)}
summ = json.load(open(os.path.join(tmp, "summary.json")))
check(not {pp.episode_of(r) for r in kept_ids} & {ep10[0], ep10[1]}, "no kept row from a forced held-out episode")
check(bool(brute) and not set(kept_ids) & brute, f"no kept row overlaps an evaluation window ({len(brute)} rows overlap)")
check(
    set(kept_ids) == {r["id"] for r in corpus if pp.episode_of(r["id"]) not in ep10[:2]} - brute,
    "exactly the rest is kept",
)
check(kept["x"] == sorted(kept["x"]), "row order kept")
check(summ["overlap_dropped"]["rows"] == len(brute) and summ["train"]["rows"] == len(kept_ids), "summary counts")

# ---- 4 ----
print("[4] render")
lora = ft.PERSONAPLEX_LIB_LORA_ADAPTER


class _P:
    def __init__(self, p):
        self.p = p

    def get(self):
        return self.p

    def get_path(self):
        return self.p


def render(hp, eval_data=None):
    job = SimpleNamespace(
        train_data=_P("/corpus"), eval_data=eval_data, out_rundir=_P("/tmp/run"), out_config=_P("/tmp/c.yaml"), seed=0,
        duration_sec=60, audio_jitter_sec=0.0, lora_rank=128, hparams=hp, adapter=lora,
    )  # fmt: skip
    return lora.render_config(job, lora.batch_size, 1000)


HP = {"emit_epad": True, "text_epad_weight": 1.0, "per_gpu_batch": 8, "grad_accum": 4, "loss_norm": "weight"}
plain = render(HP)
withe = render({**HP, "eval_every": 500}, eval_data=_P("/heldout/dataset"))
cfg = yaml.safe_load(withe)
check("eval_data" not in plain and "eval_every" not in plain, "not rendered without eval_data")
check(cfg["eval_data"] == "/heldout/dataset" and cfg["eval_every"] == 500, "rendered with it")
check(
    [ln for ln in withe.splitlines() if not ln.startswith(("eval_data", "eval_every"))] == plain.splitlines(),
    "the eval config = the plain one + those two lines",
)
for bad, what in (
    ((HP, _P("/x")), "eval_data without eval_every"),
    (({**HP, "eval_every": 5}, None), "eval_every without eval_data"),
):
    try:
        render(*bad)
        check(False, f"{what} raises")
    except ValueError:
        check(True, f"{what} raises")

# ---- 5 ----
print("[5] launcher")
src = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py").read_text()
main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
top = {t.id for n in main.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
check({"eval_data", "eval_every"} <= top, "eval_data / eval_every read at the top level of main (unconditional)")
check('cfg.get("eval_data")' in src and 'cfg.get("eval_every")' in src, "...from the config")
check("heldout_loss.loss_settings(cfg)" in src, "the eval uses heldout_loss.loss_settings of THIS config")
check(
    "heldout_loss.evaluate_set(core, eval_data, tok, eval_settings, device=device)" in src,
    "...and heldout_loss.evaluate_set",
)
check("eval_fn(0, fatal=True)" in src, "step 0 evaluated fatally before training")
check("eval_every=eval_every," in src and "eval_fn=eval_fn," in src, "eval_every -> TrainConfig, eval_fn -> run_training")

# ---- 6 ----
print("[6] cadence")


def cadence(every):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(4))
    calls = []

    def batches():
        while True:
            yield torch.ones(4)

    run_training(
        optimizer=torch.optim.AdamW([w], lr=1e-3),
        base_lrs=[1e-3],
        trainable_params=[w],
        batch_iter=batches(),
        loss_step=lambda b: ((w - b) ** 2).sum(),
        save_fn=lambda step, final: None,
        cfg=TrainConfig(max_steps=23, grad_accum=2, warmup_steps=0, save_every=10**9, log_every=10**9, eval_every=every),
        out_dir=tempfile.mkdtemp(prefix="check_ppx_cadence_"),
        log=lambda m: None,
        eval_fn=calls.append,
    )
    return calls


got5 = cadence(5)
check(got5 == [5, 10, 15, 20], f"eval at the multiples of eval_every: {got5}")
check(cadence(0) == [], "(can fail) eval_every=0 never evaluates")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("all checks passed")
