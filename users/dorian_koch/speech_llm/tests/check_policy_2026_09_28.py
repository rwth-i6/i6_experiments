"""Guard: the POLICY_2026_09_28 default (user, 2026-09-28: "exclude the first and last 2% of every episode? Add that as
a filter for future runs") -- every NEW PersonaPlex arm trains on its corpus minus the episode-edge windows -- and no
existing arm moved.

Mechanism: ``runs.RULE_PPX_EDGE_FILTER`` = "2026-09-28" = POLICY_LATEST; ``track_persona._ppx_arm`` adds ``train_rows`` =
``episode_edge_rows(train_data).out_rows`` (``persona_prompts.EpisodeEdgeRows``, edge_frac ``PPX_EDGE_FRAC``) from the
corpus's registered episode-time source (``register_edge_source``); ``ablation_2h`` declares new arms at
``AB2H_0928_POLICY`` with baseline ``AB2H_BASELINE_0928``. On the REAL graph:

  [1] epochs: 09-25c < 09-28 <= POLICY_LATEST (09-28b since 2026-09-28); the arms declared at exactly 09-28 are the
      2-hour arms pinned there (``AB2H_AT_0928``: ab2_base_0928 and the arms planned as one change from it), and no arm
      older than 09-28 trains with ``train_rows``; a pinned name is at 09-28 with baseline ab2_base_0928 (itself at
      09-28, baseline ab2_usersem); ab2_base_c stays at 09-25c; ab2_turntail03 stays at 09-25b. (Until 2026-09-28b this
      check said "no arm at 09-28 yet", stale once ab2_base_0928 was declared; check_policy_2026_09_28b covers 09-28b.)
  [2] p4's corpus (the 2-hour standard's) has its source registered: ``persona_v4/window_context_both``, that
      context's dialogue slices, and the jre_full EPISODE codes index.
  [3] a probe arm at 09-28 (in memory, never registered for a manager) carries ``train_rows`` = the EpisodeEdgeRows of
      exactly its corpus at 2%, plus every 09-25c default; an explicit ``train_rows`` is kept; an opted-out arm has
      none; a corpus without a registered source raises.
  [4] CAN FAIL: the same probe at 09-25c gets no ``train_rows``.

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_09_28.py
"""

import logging
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
os.chdir(SETUP)
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


def mod(suffix):
    (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
    return m


tp, runs, ab2h = mod(".track_persona"), mod(".doriank.runs"), mod(".track_persona_ab2h")
FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


def raises(fn, exc=(ValueError, AssertionError, RuntimeError)):
    try:
        fn()
    except exc as e:
        return str(e)[:120]
    return None


C, D = runs.POLICY_2026_09_25C, runs.POLICY_2026_09_28
print("[1] epochs, pins, baselines")
check(C < D <= runs.POLICY_LATEST and D in runs.KNOWN_POLICIES, "09-25c < 09-28 <= POLICY_LATEST, known")
check(runs.RULE_PPX_EDGE_FILTER == D and runs.PPX_EDGE_FRAC == 0.02, "rule at 09-28, edge_frac 0.02")
at_d = sorted(t for t, p in tp.PPX_POLICY.items() if p == D)
check(
    ab2h.AB2H_BASELINE_0928 in at_d and set(at_d) <= ab2h.AB2H_AT_0928,
    f"the arms at exactly 09-28 are the pinned 2-hour arms ({at_d})",
)
old_rows = sorted(t for t, r in tp.PPX_ARMS.items() if "train_rows" in r.hparams and tp.PPX_POLICY.get(t, "") < D)
check(not old_rows, f"no arm older than 09-28 trains with train_rows ({old_rows})")
check(
    all(
        ab2h.ab2h_policy(n) == D and ab2h.ab2h_baseline(n) == ab2h.AB2H_BASELINE_0928
        for n in ab2h.AB2H_AT_0928 - {ab2h.AB2H_BASELINE_0928}
    )
    and ab2h.ab2h_policy(ab2h.AB2H_BASELINE_0928) == D
    and ab2h.ab2h_baseline(ab2h.AB2H_BASELINE_0928) == ab2h.AB2H_BASELINE_USERSEM,
    "a pinned 2-hour arm: 09-28, baseline ab2_base_0928 (itself at 09-28, rated against ab2_usersem)",
)
check(
    ab2h.ab2h_policy(ab2h.AB2H_BASELINE_C) == C
    and ab2h.ab2h_baseline(ab2h.AB2H_BASELINE_C) == ab2h.AB2H_BASELINE_USERSEM
    and ab2h.ab2h_policy("ab2_turntail03") == runs.POLICY_2026_09_25B,
    "ab2_base_c stays at 09-25c (the filter's own ablation), ab2_turntail03 at 09-25b",
)

print("[2] p4's corpus has its episode-time source")
data = ab2h._STATE["train_data"]
src = tp.PPX_EDGE_SOURCES.get(data.get_path())
ctx = tp.WINDOW_CONTEXT_BOTH["persona_v4"]
check(src is not None and src["name"] == "persona_v4", f"registered as persona_v4 ({data.get_path()})")
check(
    src is not None
    and src["window_context"].get_path() == ctx.out_dir.get_path()
    and [p.get_path() for p in src["slice_dirs"]] == [p.get_path() for p in ctx.dialogue_shard_dirs],
    "window context = persona_v4/window_context_both, slices = that context's",
)
check(
    src is not None
    and os.path.realpath(src["episode_index"].get_path())
    == os.path.realpath("output/podcast_episodes/jre_full/codes_index"),
    "episode index = the jre_full EPISODE codes index",
)

print("[3] a probe arm at 09-28")
HP = ab2h.AB2H_HPARAMS


def arm(tag, extra=None, policy=D, train_data=data):
    return tp._ppx_arm(f"_guard_{tag}", train_data, extra or {}, hparams=HP, steps=100, benchmark=False, policy=policy)


hp = arm("d").hparams
tr = hp.get("train_rows")
job = getattr(tr, "creator", None)
check(
    type(job).__name__ == "EpisodeEdgeRows"
    and job.data.get_path() == data.get_path()
    and job.edge_frac == 0.02
    and job.window_context.get_path() == ctx.out_dir.get_path(),
    "train_rows = EpisodeEdgeRows(p4's corpus, its context, 2%).out_rows",
)
check(
    hp.get("weight_decay") == 1e-4 and hp.get("grad_accum") == 1 and hp.get("knowledge_data") is not None,
    "... plus the 09-25c defaults (wd, no accumulation, knowledge mix)",
)
check(arm("own", {"train_rows": "/x.npy"}).hparams["train_rows"] == "/x.npy", "an explicit train_rows is kept")
tp.PPX_EDGE_FILTER_OPT_OUT["_guard_optout"] = "guard"
check("train_rows" not in arm("optout").hparams, "an opted-out arm has no train_rows")
check(bool(raises(lambda: arm("unreg", train_data=tk.Path("/nonexistent/corpus")))), "an unregistered corpus raises")

print("[4] CAN FAIL: the gate")
check("train_rows" not in arm("c", policy=C).hparams, "the same arm at 09-25c: no train_rows")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
