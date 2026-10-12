"""Guard: the POLICY_2026_09_29 defaults (user decision 2026-09-29: "so lets use your new proposed ablation baseline. use
wordmask 0, but also extend by a frame in each direction") reach the REAL launched config of every new PersonaPlex arm,
and no earlier arm moved.

Mechanism: ``runs.RULE_PPX_WORDMASK`` / ``RULE_PPX_OPENYAP_MIX`` = "2026-09-29" = POLICY_LATEST; ``track_persona._ppx_arm``
adds ``train_word_spans`` = ``ppx_word_spans(data)`` at weight 0 with ``train_word_pad_frames`` 1, and the Open Yap
stratum (``aux_data`` = ``PPX_OPENYAP_DEFAULT["data"]`` at 0.35) with the arm's own row rules on its rows
(``aux_train_rows`` / ``aux_train_row_weights`` / ``aux_train_word_spans``). On the REAL graph:

  [1] epochs: 09-28b < 09-29 = POLICY_LATEST, known; the arms at 09-29 are exactly ab2_base_0929; a new 2-hour name is at
      09-29 with baseline ab2_base_0929 (itself rated against ab2_base_0928b); the four 09-28b arms stay at 09-28b.
  [2] ab2_base_0929's hparams = ab2_base_0928b's + EXACTLY the eight 09-29 keys: the word spans are ab2_wordmask's own
      FlaggedWordSpans(p4) job at weight 0 and pad 1; aux_data = the Open Yap corpus (= ab2_mix_jre_oy_know's), share 0.35;
      aux_train_rows = IntersectRows([EpisodeEdgeRows(OY, 2%), LexiconRows(OY, explicit_v1, sexual, trainable)]),
      aux_train_row_weights = ThresholdRowWeights(MonologueRows(OY).out_stretch, 20, 0.3), aux_train_word_spans =
      FlaggedWordSpans(OY).
  [3] REAL CONFIG: its ``render_config`` (what ``write_config`` writes) carries every key with those paths and
      ``train_word_pad_frames: 1``; the launcher reads the three new keys and hands them to the mix loader.
  [4] probe arms (in memory): an explicit aux_data / train_word_spans is kept; each opt-out drops its keys; a knowledge
      opt-out without the Open Yap opt-out raises; the renderer refuses aux_train_word_spans without train_word_spans.
  [5] ab2_base_0929 has a spj2 judge and an Open Yap held-out summary.
  [6] CAN FAIL: the checks of [2] / [3] applied to ab2_wordmask (09-28b) fail.

Run from the setup root (~2 min, one config load, no GPU):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_policy_2026_09_29.py
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
spj, oy_track = mod(".track_persona_spjudge"), mod(".track_openyap")
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


B, N = runs.POLICY_2026_09_28B, runs.POLICY_2026_09_29
data = ab2h._STATE["train_data"]
BASE_B, BASE_N = ab2h.AB2H_BASELINE_0928B, ab2h.AB2H_BASELINE_0929
OY = tp.PPX_OPENYAP_DEFAULT.get("data")
NEW_KEYS = sorted(
    [
        "train_word_spans",
        "train_word_weight",
        "train_word_pad_frames",
        "aux_data",
        "aux_share",
        "aux_train_rows",
        "aux_train_row_weights",
        "aux_train_word_spans",
    ]
)

print("[1] epochs, pins, baselines")
check(B < N == runs.POLICY_LATEST and N in runs.KNOWN_POLICIES, "09-28b < 09-29 = POLICY_LATEST, known")
check(
    runs.RULE_PPX_WORDMASK == N == runs.RULE_PPX_OPENYAP_MIX
    and runs.PPX_WORD_WEIGHT == 0.0
    and runs.PPX_WORD_PAD_FRAMES == 1
    and runs.PPX_OPENYAP_AUX_SHARE == 0.35,
    "rules: word mask weight 0 pad 1, Open Yap share 0.35, both at 09-29",
)
at_n = sorted(t for t, p in tp.PPX_POLICY.items() if p == N)
#: the one-change arms declared at 09-29 so far (2026-09-29: knowledge-only, known-only knowledge mix; 2026-10-12: the
#: user-text head, track_persona_usertext); a new one is added here
AT_N_ARMS = {"ab2_knowonly_3p", "ab2_know_known", "ab2_usertext"}
check(at_n == sorted({BASE_N} | AT_N_ARMS), f"the arms at 09-29 = ab2_base_0929 + {sorted(AT_N_ARMS)} ({at_n})")
check(
    ab2h.ab2h_policy("some_new_arm") == N
    and ab2h.ab2h_baseline("some_new_arm") == BASE_N
    and ab2h.ab2h_policy(BASE_N) == N
    and ab2h.ab2h_baseline(BASE_N) == BASE_B,
    "a new 2-hour arm: 09-29, baseline ab2_base_0929 (itself at 09-29, rated against ab2_base_0928b)",
)
check(
    all(ab2h.ab2h_policy(n) == B and tp.PPX_POLICY[n] == B for n in ab2h.AB2H_AT_0928B),
    f"the 09-28b arms stay at 09-28b ({sorted(ab2h.AB2H_AT_0928B)})",
)
check(OY is not None, "the Open Yap corpus is registered as PPX_OPENYAP_DEFAULT")


def epoch_errors(run) -> list:
    """What must hold for an arm at 09-29 on p4's corpus; returns the failed items."""
    hp, base = run.hparams, tp.PPX_ARMS[BASE_B].hparams
    diff = sorted(k for k in set(hp) | set(base) if hp.get(k) != base.get(k))
    errs = [] if diff == NEW_KEYS else [f"differs from ab2_base_0928b in {diff}"]
    ws = creator(hp.get("train_word_spans"))
    if ws is not tp.ppx_word_spans(data) or hp.get("train_word_weight") != 0.0 or hp.get("train_word_pad_frames") != 1:
        errs.append("word mask is not ppx_word_spans(p4) at weight 0, pad 1")
    if ws is not creator(tp.PPX_ARMS["ab2_wordmask"].hparams["train_word_spans"]):
        errs.append("word spans are not ab2_wordmask's job")
    if OY is None or hp.get("aux_data") is None or hp["aux_data"].get_path() != OY.get_path():
        errs.append("aux_data is not the Open Yap corpus")
    elif hp["aux_data"].get_path() != tp.PPX_ARMS["ab2_mix_jre_oy_know"].hparams["aux_data"].get_path():
        errs.append("aux_data is not ab2_mix_jre_oy_know's corpus")
    if hp.get("aux_share") != 0.35:
        errs.append(f"aux_share {hp.get('aux_share')}")
    ir = creator(hp.get("aux_train_rows"))
    if type(ir).__name__ != "IntersectRows" or len(ir.rows) != 2:
        errs.append("aux_train_rows is not an IntersectRows of two keep-lists")
    else:
        edge, lex = creator(ir.rows[0]), creator(ir.rows[1])
        if not (type(edge).__name__ == "EpisodeEdgeRows" and edge.data.get_path() == OY.get_path()):
            errs.append("aux keep-list 1 is not EpisodeEdgeRows(OY)")
        if not (
            type(lex).__name__ == "LexiconRows"
            and lex.data.get_path() == OY.get_path()
            and lex.categories == ("sexual",)
            and lex.sides == "trainable"
        ):
            errs.append("aux keep-list 2 is not LexiconRows(OY, sexual, trainable)")
    tw = creator(hp.get("aux_train_row_weights"))
    if not (
        type(tw).__name__ == "ThresholdRowWeights"
        and tw.above == 20.0
        and tw.weight == 0.3
        and creator(tw.values).data.get_path() == OY.get_path()
    ):
        errs.append("aux_train_row_weights is not ThresholdRowWeights(MonologueRows(OY), 20, 0.3)")
    aws = creator(hp.get("aux_train_word_spans"))
    if not (type(aws).__name__ == "FlaggedWordSpans" and aws.data.get_path() == OY.get_path()):
        errs.append("aux_train_word_spans is not FlaggedWordSpans(OY)")
    return errs


def render(run):
    return yaml.safe_load(run.adapter.render_config(run, run.adapter.batch_size, run.max_steps))


def config_errors(run) -> list:
    cfg, hp, errs = render(run), run.hparams, []
    for k in ("train_word_spans", "aux_data", "aux_train_rows", "aux_train_row_weights", "aux_train_word_spans"):
        if hp.get(k) is None or cfg.get(k) != hp[k].get():
            errs.append(f"config {k} {cfg.get(k)!r}")
    if cfg.get("train_word_weight") != 0.0 or cfg.get("train_word_pad_frames") != 1 or cfg.get("aux_share") != 0.35:
        errs.append(
            f"config weight / pad / share {cfg.get('train_word_weight')}, {cfg.get('train_word_pad_frames')}, "
            f"{cfg.get('aux_share')}"
        )
    return errs


print("[2] ab2_base_0929's hparams")
base_run = tp.PPX_ARMS[BASE_N]
e = epoch_errors(base_run)
check(not e, f"ab2_base_0929 = ab2_base_0928b + exactly the 09-29 keys ({e})")

print("[3] the real rendered config and the launcher")
e = config_errors(base_run)
check(not e, f"ab2_base_0929's rendered config carries every 09-29 key ({e})")
cfg0928b = render(tp.PPX_ARMS[BASE_B])
check(
    not any(k in cfg0928b for k in ("train_word_pad_frames", "aux_train_row_weights", "aux_train_word_spans")),
    "ab2_base_0928b's config carries none of the new keys (rendered only when set)",
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
    {"train_word_pad_frames", "aux_train_row_weights", "aux_train_word_spans"} <= got,
    "the launcher reads train_word_pad_frames / aux_train_row_weights / aux_train_word_spans",
)
consts = {n.value for n in ast.walk(main) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
check(
    {"aux_row_weights", "aux_word_spans", "train_word_pad_frames"} <= consts,
    "... and hands them to the loaders as aux_row_weights / aux_word_spans / train_word_pad_frames",
)

print("[4] probe arms at 09-29")
HP = ab2h.AB2H_HPARAMS


def arm(tag, extra=None, policy=N):
    return tp._ppx_arm(f"_guard29_{tag}", data, extra or {}, hparams=HP, steps=100, benchmark=False, policy=policy)


check(not epoch_errors(arm("n")), "a probe arm at 09-29 gets the defaults")
hp = arm("own", {"aux_data": "/oy", "aux_share": 0.2, "train_word_spans": "/s.json", "train_word_weight": 0.1}).hparams
check(
    hp["aux_data"] == "/oy"
    and "aux_train_rows" not in hp
    and hp["train_word_spans"] == "/s.json"
    and hp["train_word_weight"] == 0.1
    and "train_word_pad_frames" not in hp,
    "explicit aux_data / train_word_spans are kept (no default added on top)",
)
tp.PPX_WORDMASK_OPT_OUT["_guard29_nowm"] = "guard"
hp = arm("nowm").hparams
check(
    not any(k in hp for k in ("train_word_spans", "train_word_pad_frames", "aux_train_word_spans"))
    and hp.get("aux_data") is not None,
    "word-mask opt-out: no spans on either corpus, the stratum stays",
)
tp.PPX_OPENYAP_MIX_OPT_OUT["_guard29_nooy"] = "guard"
hp = arm("nooy").hparams
check(
    not any(k.startswith("aux_") for k in hp) and hp.get("train_word_pad_frames") == 1,
    "Open Yap opt-out: no aux keys, the word mask stays",
)
tp.PPX_KNOWLEDGE_OPT_OUT["_guard29_nokn"] = "guard"
check(bool(raises(lambda: arm("nokn"))), "a knowledge opt-out without the Open Yap opt-out raises")
tp.PPX_WORDMASK_OPT_OUT["_guard29_bad"] = "guard"
bad = arm("bad", {"aux_train_word_spans": base_run.hparams["aux_train_word_spans"]})
check(bool(raises(lambda: render(bad))), "the renderer refuses aux_train_word_spans without train_word_spans")

print("[5] evals")
check(BASE_N in spj.SPJ2_JUDGES, "ab2_base_0929 has a spj2 judge")
check(
    any(k.endswith(f"persona/heldout_loss/{BASE_N}/summary_openyap.json") for k in tk.sis_graph.targets_dict)
    or any(f"heldout_loss/{BASE_N}/" in k and "openyap" in k for k in tk.sis_graph.targets_dict),
    "ab2_base_0929 has the Open Yap held-out loss",
)

print("[6] CAN FAIL: the 09-29 checks on ab2_wordmask (09-28b)")
old = tp.PPX_ARMS["ab2_wordmask"]
check(bool(epoch_errors(old)), f"ab2_wordmask fails the epoch checks ({epoch_errors(old)[:2]})")
check(bool(config_errors(old)), f"ab2_wordmask's config fails the config checks ({config_errors(old)[:2]})")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
