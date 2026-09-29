"""Guard: the POLICY_2026_09_28B defaults (user decisions 2026-09-28: "if 30s is the cap now, thats fine. But scale down
the loss if theres super long monologue ... Lets do the word list filter for sexual content"; "We should keep a single
text temparature for the self play benchmark. Just choose one. That benchmark should be run for every new ablation and
finetune") reach the REAL launched config of every new PersonaPlex arm, and no earlier arm moved.

Mechanism: ``runs.RULE_PPX_MONOLOGUE_WEIGHT`` / ``RULE_PPX_LEXICON_FILTER`` = "2026-09-28b" = POLICY_LATEST;
``track_persona._ppx_arm`` adds ``train_row_weights`` = ``ppx_row_weights(data, 20, 0.3)`` (ThresholdRowWeights over
MonologueRows.out_stretch) and ``train_rows`` = ``ppx_train_rows(data, lexicon_categories=("sexual",))`` (IntersectRows
over EpisodeEdgeRows + LexiconRows); ``runs.RULE_PPX_SPJ2`` = 09-28 (an eval) declares the self-play judge v2 at ONE
temperature for every arm from 09-28 on. On the REAL graph:

  [1] epochs: 09-28 < 09-28b <= POLICY_LATEST, known; the arms at 09-28b are exactly the four 2-hour arms of
      ``track_persona_wordmask`` (pinned there, ``AB2H_AT_0928B``, since 09-29 made a newer epoch; new arms:
      ``check_policy_2026_09_29``); ab2_base_0928b is rated against ab2_base_0928; the pinned 09-28 arms keep 09-28.
  [2] ab2_base_0928b's hparams: ``train_rows`` = IntersectRows([EpisodeEdgeRows(p4, 2%), LexiconRows(p4, explicit_v1,
      sexual, trainable)]) and ``train_row_weights`` = ThresholdRowWeights(MonologueRows(p4, 20 s).out_stretch, above
      20, weight 0.3) + every 09-28 default; the three word arms = that + exactly their one change.
  [3] REAL CONFIG: ab2_base_0928b's ``render_config`` (the job's own adapter, exactly what ``write_config`` writes)
      carries ``train_rows`` / ``train_row_weights`` with those jobs' paths, and the launcher reads both keys and passes
      them to both loaders (AST of ``finetune_launcher.main``).
  [4] probe arms (in memory): an explicit ``train_rows`` / ``train_row_weights`` is kept; the opt-outs drop them;
      ``turn_tail_weight`` without the row-weight opt-out raises.
  [5] spj2: ``SPJ2_TEMPS == (None,)``; every arm declared at >= 09-28 has a judge whose runs are ONE self-play at the
      default temperature; no registered spj2 output names a T 1.0 run; the wordmask table exists.
  [6] CAN FAIL: the checks of [2]/[3] applied to ab2_base_0928 (09-28) fail (no row weight, edge-only rows).

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_09_28b.py
"""

import ast
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

import yaml  # noqa: E402
from sisyphus import tk  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


def mod(suffix):
    (m,) = [v for k, v in sys.modules.items() if k.endswith(suffix) and "sis_recipe.doriank" in k]
    return m


tp, runs, ab2h = mod(".track_persona"), mod(".doriank.runs"), mod(".track_persona_ab2h")
spj, wm = mod(".track_persona_spjudge"), mod(".track_persona_wordmask")
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


def creator(p):
    return getattr(p, "creator", None)


D, B = runs.POLICY_2026_09_28, runs.POLICY_2026_09_28B
data = ab2h._STATE["train_data"]
BASE_B = ab2h.AB2H_BASELINE_0928B

print("[1] epochs, pins, baselines")
check(D < B <= runs.POLICY_LATEST and B in runs.KNOWN_POLICIES, "09-28 < 09-28b <= POLICY_LATEST, known")
check(
    runs.RULE_PPX_MONOLOGUE_WEIGHT == B == runs.RULE_PPX_LEXICON_FILTER
    and runs.PPX_MONOLOGUE_WEIGHT == 0.3
    and runs.PPX_MONOLOGUE_WEIGHT_ABOVE_SEC == 20.0
    and runs.PPX_LEXICON_FILTER_CATEGORIES == ("sexual",)
    and runs.RULE_PPX_SPJ2 == D,
    "rules: row weight 0.3 above 20 s and sexual lexicon at 09-28b; spj2 (eval) from 09-28",
)
at_b = sorted(t for t, p in tp.PPX_POLICY.items() if p == B)
check(at_b == sorted([BASE_B, *wm.WORDMASK_ARMS]), f"the arms at 09-28b = ab2_base_0928b + the word arms ({at_b})")
check(
    all(ab2h.ab2h_policy(n) == B for n in ab2h.AB2H_AT_0928B)
    and ab2h.AB2H_AT_0928B == frozenset([BASE_B, *wm.WORDMASK_ARMS])
    and ab2h.ab2h_baseline(BASE_B) == ab2h.AB2H_BASELINE_0928,
    "the 09-28b arms are pinned at 09-28b (AB2H_AT_0928B); ab2_base_0928b is rated against ab2_base_0928",
)
check(
    all(ab2h.ab2h_policy(n) == D for n in ab2h.AB2H_AT_0928)
    and all(tp.PPX_POLICY[n] == D for n in ab2h.AB2H_AT_0928 if n in tp.PPX_POLICY),
    "ab2_base_0928 and the planned 09-28 arms (know7, Open Yap) stay at 09-28",
)


def epoch_errors(run) -> list:
    """What must hold for an arm at 09-28b on p4's corpus; returns the failed items."""
    errs = []
    hp = run.hparams
    ir = creator(hp.get("train_rows"))
    if type(ir).__name__ != "IntersectRows" or len(ir.rows) != 2:
        return ["train_rows is not an IntersectRows of two keep-lists"]
    edge, lex = creator(ir.rows[0]), creator(ir.rows[1])
    if not (
        type(edge).__name__ == "EpisodeEdgeRows" and edge.data.get_path() == data.get_path() and edge.edge_frac == 0.02
    ):
        errs.append("keep-list 1 is not EpisodeEdgeRows(p4, 2%)")
    if not (
        type(lex).__name__ == "LexiconRows"
        and lex.data.get_path() == data.get_path()
        and lex.lexicon == "explicit_v1"
        and lex.categories == ("sexual",)
        and lex.sides == "trainable"
    ):
        errs.append("keep-list 2 is not LexiconRows(p4, explicit_v1, sexual, trainable)")
    tw = creator(hp.get("train_row_weights"))
    if not (type(tw).__name__ == "ThresholdRowWeights" and tw.above == 20.0 and tw.weight == 0.3):
        errs.append("train_row_weights is not ThresholdRowWeights(above 20, weight 0.3)")
    else:
        mono = creator(tw.values)
        if not (type(mono).__name__ == "MonologueRows" and mono.data.get_path() == data.get_path()):
            errs.append("the row weights' values are not MonologueRows(p4).out_stretch")
    if not (hp.get("weight_decay") == 1e-4 and hp.get("grad_accum") == 1 and hp.get("knowledge_data") is not None):
        errs.append("the 09-25c defaults are missing")
    if hp.get("user_semantic_weight") != 1.0:
        errs.append("user_semantic_weight is not 1.0")
    return errs


def config_errors(run) -> list:
    """The REAL rendered config (the job's adapter.render_config, as write_config) carries both keys."""
    cfg = yaml.safe_load(run.adapter.render_config(run, run.adapter.batch_size, run.max_steps))
    errs = []
    tr, tw = run.hparams.get("train_rows"), run.hparams.get("train_row_weights")
    if tr is None or cfg.get("train_rows") != tr.get():
        errs.append(f"config train_rows {cfg.get('train_rows')!r}")
    if tw is None or cfg.get("train_row_weights") != tw.get():
        errs.append(f"config train_row_weights {cfg.get('train_row_weights')!r}")
    return errs


print("[2] ab2_base_0928b's hparams and the word arms' one change")
base_run = tp.PPX_ARMS[BASE_B]
e = epoch_errors(base_run)
check(not e, f"ab2_base_0928b = the 09-28b defaults ({e})")
for name, key in (
    ("ab2_wordmask", "train_word_spans"),
    ("ab2_wordw01", "train_word_spans"),
    ("ab2_beep", "train_censor"),
):
    run = tp.PPX_ARMS[name]
    diff = sorted(k for k in set(run.hparams) | set(base_run.hparams) if run.hparams.get(k) != base_run.hparams.get(k))
    want = ["train_word_spans", "train_word_weight"] if key == "train_word_spans" else ["train_censor"]
    check(not epoch_errors(run) and diff == want, f"{name}: the 09-28b defaults + exactly {want} ({diff})")
check(
    tp.PPX_ARMS["ab2_wordmask"].hparams["train_word_weight"] == 0.0
    and tp.PPX_ARMS["ab2_wordw01"].hparams["train_word_weight"] == 0.1
    and type(creator(tp.PPX_ARMS["ab2_wordmask"].hparams["train_word_spans"])).__name__ == "FlaggedWordSpans"
    and creator(tp.PPX_ARMS["ab2_wordmask"].hparams["train_word_spans"]).categories == ("profanity", "sexual")
    and type(creator(tp.PPX_ARMS["ab2_beep"].hparams["train_censor"])).__name__ == "CensorFlaggedWords",
    "word weights 0 / 0.1 over FlaggedWordSpans(profanity + sexual); the beep arm's censor job",
)

print("[3] the real rendered config and the launcher")
e = config_errors(base_run)
check(not e, f"ab2_base_0928b's rendered config carries train_rows + train_row_weights ({e})")
cfgw = yaml.safe_load(tp.PPX_ARMS["ab2_wordmask"].adapter.render_config(
    tp.PPX_ARMS["ab2_wordmask"], base_run.adapter.batch_size, base_run.max_steps))  # fmt: skip
check(
    cfgw.get("train_word_spans") == tp.PPX_ARMS["ab2_wordmask"].hparams["train_word_spans"].get()
    and cfgw.get("train_word_weight") == 0.0,
    "ab2_wordmask's rendered config carries train_word_spans + train_word_weight 0",
)
src = Path("recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py").read_text()
main = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")
got = {
    n.args[0].value
    for n in ast.walk(main)
    if isinstance(n, ast.Call)
    and getattr(n.func, "attr", "") == "get"
    and n.args
    and isinstance(n.args[0], ast.Constant)
}
check(
    {"train_rows", "train_row_weights", "train_word_spans", "train_word_weight", "train_censor"} <= got,
    "the launcher reads train_rows / train_row_weights / train_word_spans / train_word_weight / train_censor",
)
loader_kw = {
    n.func.id: {k.arg for k in n.keywords}
    for n in ast.walk(main)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    and n.func.id in ("build_data_loader", "build_knowledge_mix_loader")
}  # fmt: skip
check(
    all({"train_rows", "train_row_weights"} <= kw for kw in loader_kw.values()) and len(loader_kw) == 2,
    f"... and passes train_rows + train_row_weights to both loaders ({sorted(loader_kw)})",
)

print("[4] probe arms at 09-28b")
HP = ab2h.AB2H_HPARAMS


def arm(tag, extra=None, policy=B):
    return tp._ppx_arm(f"_guard_{tag}", data, extra or {}, hparams=HP, steps=100, benchmark=False, policy=policy)


check(not epoch_errors(arm("b")), "a probe arm at 09-28b gets the defaults")
hp = arm("own", {"train_rows": "/x.npy", "train_row_weights": "/w.npy"}).hparams
check(hp["train_rows"] == "/x.npy" and hp["train_row_weights"] == "/w.npy", "explicit train_rows / weights are kept")
tp.PPX_LEXICON_FILTER_OPT_OUT["_guard_nolex"] = "guard"
check(
    type(creator(arm("nolex").hparams["train_rows"])).__name__ == "EpisodeEdgeRows", "lexicon opt-out: edge rows only"
)
tp.PPX_ROW_WEIGHT_OPT_OUT["_guard_now"] = "guard"
check("train_row_weights" not in arm("now").hparams, "row-weight opt-out: no train_row_weights")
check(bool(raises(lambda: arm("tt", {"turn_tail_weight": 0.3}))), "turn_tail_weight without the opt-out raises")

print("[5] spj2: one temperature, every new arm")
check(spj.SPJ2_TEMPS == (None,), f"SPJ2_TEMPS == (None,) ({spj.SPJ2_TEMPS})")
new_arms = sorted(t for t, p in tp.PPX_POLICY.items() if p >= D and not t.startswith("_guard_"))
missing = [t for t in new_arms if t not in spj.SPJ2_JUDGES]
check(not missing, f"every arm at >= 09-28 has a spj2 judge ({len(new_arms)} arms; missing {missing})")
bad = []
for t, j in spj.SPJ2_JUDGES.items():
    sp = [creator(creator(p).selfplay_data) for p in j.runs.values()]
    if len(j.runs) != 1 or any(s.temp_text is not None for s in sp):
        bad.append(t)
check(not bad, f"every spj2 judge rates ONE self-play at the default temperature ({bad})")
outs = [k for k in tk.sis_graph.targets_dict if "spj2" in k and "t1.0" in k]
check(not outs, f"no registered spj2 output names a T 1.0 run ({outs[:3]})")
check(
    any(k.endswith("persona/spj2/table_wordmask.md") for k in tk.sis_graph.targets_dict),
    "the wordmask spj2 table is registered",
)

print("[6] CAN FAIL: the 09-28b checks on ab2_base_0928 (09-28)")
old = tp.PPX_ARMS[ab2h.AB2H_BASELINE_0928]
check(bool(epoch_errors(old)), f"ab2_base_0928 fails the epoch checks ({epoch_errors(old)[:2]})")
check(bool(config_errors(old)), f"ab2_base_0928's config fails the config checks ({config_errors(old)[:2]})")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
