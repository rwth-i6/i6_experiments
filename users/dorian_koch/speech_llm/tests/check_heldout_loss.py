"""Guard: the PersonaPlex held-out loss (``moshi_family.personaplex.heldout_loss`` / ``heldout_set``, Sisyphus jobs in
``persona_heldout.py``) is the TRAINING loss, on a fixed set, in eval mode (user, 2026-09-25).

A held-out number that is computed differently from the logged training loss is worse than none: the train vs
held-out gap it shows would be an artifact. Each check drives the real code:

  [1] defaults: every config key ``heldout_loss.loss_settings`` reads has the SAME default as
      ``finetune_launcher.main``'s ``cfg.get`` (AST of the real launcher), so a key absent from a run's config means
      what it meant in training; ``heldout_set.PERSONA_MINIMAL`` == ``persona_prompts.PERSONA_MINIMAL``.
  [2] parity on TRAINING windows: 4 real rows of p4's training table (``build_train_sample``) and 4 held-out rows
      (``build_heldout``), a tiny real PersonaPlex LM (+ LoRA, non-zero B, fp32, CPU). Training side = the REAL
      training loader (``build_data_loader``, batch 1) + the launcher's REAL ``loss_step`` (its source cut out of
      ``finetune_launcher.py`` and executed with the variables ``main()`` binds from p5's / p4's real config.yaml),
      model in TRAIN mode. Job side = ``heldout_loss.evaluate_set``. Loss and every component equal (rel 1e-6), with
      and without ``depformer_skip_masked``; the rows carry EPAD targets (else the EPAD weight could not matter).
  [3] eval mode: a forward-pre-hook spy sees ``training=False`` on every module during ``evaluate_set``.
      CAN FAIL: the module with ``core.eval()`` -> ``core.train()`` is flagged by the spy.
  [4] CAN FAIL: a ``loss_settings`` that drops ``text_epad_weight`` (EPAD back at the PAD weight), and one that
      ignores ``loss_norm``, both break the parity of [2].
  [5] fixed-set determinism: ``build_train_sample`` twice with seed 0 -> the same ids and codes, seed 1 -> other ids;
      ``build_heldout`` (n=400, the real pool): 400 distinct ids, none in p4's training table, no episode of p2's or
      p3's corpus, every row with a voice clip, reproducible; ``evaluate_set`` twice under different global RNG
      states -> bit-identical numbers; ``summarize`` reproduces its bootstrap CI.
  [6] ``persona_heldout._consolidated_complete`` (how an eval waits for a checkpoint of a run that is still
      training): weights without ``config.json`` -> not available, with it -> available.

moshi_family venv, CPU, ~3 min (the real window tables, the HF-cached mimi + tokenizer):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd /home/tt201262/setups/2026-01-speech-llm && HF_HUB_OFFLINE=1 \
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_heldout_loss.py'
"""

import ast
import os
import sys
import tempfile
import textwrap
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from moshi_family.build import LoraConfig, apply_lora_to  # noqa: E402
from moshi_family.personaplex import heldout_loss as HL  # noqa: E402
from moshi_family.personaplex import heldout_set as HS  # noqa: E402
from moshi_family.personaplex.models import lm as ppx_lm  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.train_data import build_data_loader, personaplex_loss  # noqa: E402
from moshi_family.train_data_common import TEXT_EPAD_ID  # noqa: E402

SETUP = os.getcwd()
LIB = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex", "moshi_family")
LAUNCHER = os.path.join(LIB, "personaplex", "finetune_launcher.py")
HL_SRC = os.path.join(LIB, "personaplex", "heldout_loss.py")
I6 = os.path.join(SETUP, "recipe", "i6_experiments", "users", "dorian_koch", "speech_llm")
OUT = os.path.join(SETUP, "output")
WORK = os.path.join(SETUP, "work", "i6_experiments", "users", "dorian_koch", "speech_llm", "finetune")
CONFIGS = {
    "p5": os.path.join(WORK, "SpeechFinetune.SWboYEq1ncUk", "output", "config.yaml"),
    "p4": os.path.join(WORK, "SpeechFinetune.xXTXBmQ1i3gM", "output", "config.yaml"),
}

fails = []


def ck(cond, label, extra=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not cond:
        fails.append(label)


# ---------------------------------------------------------------------------------------------------------------
print("[1] config defaults == the launcher's; the minimal prompt == persona_prompts'")
tree = ast.parse(open(LAUNCHER).read())
main_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
launcher_defaults = {}
for n in ast.walk(main_fn):
    if (
        isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "get"
        and getattr(n.func.value, "id", None) == "cfg"
    ):
        key = n.args[0].value
        try:
            launcher_defaults[key] = ast.literal_eval(n.args[1]) if len(n.args) > 1 else None
        except ValueError:  # an expression default (e.g. gradient_checkpointing's); not a key read here
            launcher_defaults[key] = f"<expr {ast.unparse(n.args[1])}>"
for key, d in HL.LAUNCHER_DEFAULTS.items():
    ck(key in launcher_defaults, f"launcher reads {key!r}")
    ck(
        launcher_defaults.get(key, "<unread>") == d,
        f"default of {key!r}",
        f"launcher {launcher_defaults.get(key)!r} / heldout {d!r}",
    )
pp = ast.parse(open(os.path.join(I6, "persona_prompts.py")).read())
pm = next(
    ast.literal_eval(n.value)
    for n in pp.body
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PERSONA_MINIMAL"
)
ck(pm == HS.PERSONA_MINIMAL, "PERSONA_MINIMAL identical", repr(pm))

# ---------------------------------------------------------------------------------------------------------------
print("[5a] fixed sets on the real data: reproducible, disjoint from training")
tmp = tempfile.mkdtemp(prefix="check_heldout_", dir=os.environ.get("TMPDIR"))
ts0, s0 = HS.build_train_sample(os.path.join(OUT, "persona_v4", "p1_data"), n=4, seed=0)
ts0b, _ = HS.build_train_sample(os.path.join(OUT, "persona_v4", "p1_data"), n=4, seed=0)
ts1, _ = HS.build_train_sample(os.path.join(OUT, "persona_v4", "p1_data"), n=4, seed=1)
ck(
    list(ts0["id"]) == list(ts0b["id"]) and list(ts0["codes_user"]) == list(ts0b["codes_user"]),
    "train_sample: seed 0 twice -> same rows",
)
ck(set(list(ts0["id"])) != set(list(ts1["id"])), "train_sample: seed 1 -> other rows (non-vacuous)")
held_args = (
    os.path.join(OUT, "podcast_train", "jre_full_w20_60_t3_s30"),
    os.path.join(OUT, "persona_v4", "window_context_both"),
    os.path.join(OUT, "persona_v4", "voice_codes"),
    [os.path.join(OUT, "persona_v2", "window_context"), os.path.join(OUT, "persona_v3", "window_context")],
    [os.path.join(OUT, "persona_v4", "p1_data")],
)
hd, hsum = HS.build_heldout(*held_args, n=400, seed=0)
hd_again, _ = HS.build_heldout(*held_args, n=400, seed=0)
from datasets import load_from_disk  # noqa: E402

train_ids = set(load_from_disk(os.path.join(OUT, "persona_v4", "p1_data")).select_columns(["id"])["id"])
excl_eps = set()
for p in held_args[3]:
    excl_eps |= set(load_from_disk(p).select_columns(["episode_id"])["episode_id"])
ids = list(hd["id"])
ck(hsum["pool"] == 835, "held-out pool = 835 windows (the user-reliance pool)", str(hsum["pool"]))
ck(len(ids) == 400 and len(set(ids)) == 400, "400 distinct held-out windows")
ck(not (set(ids) & train_ids), "no held-out id in p4's (= p5's) training table")
ck(not (set(list(hd["episode_id"])) & excl_eps), "no held-out episode in p2's / p3's corpus")
ck(all(len(v) > 0 for v in list(hd["voice_codes"])), "every held-out row has a voice clip")
ck(ids == list(hd_again["id"]), "held-out: same seed -> same 400 ids")
ck(set(list(ts0["id"])) <= train_ids, "train_sample rows are training rows")
held4 = hd.select(range(4))
set_dirs = {"train_sample": os.path.join(tmp, "train_sample"), "heldout": os.path.join(tmp, "heldout")}
ts0.save_to_disk(set_dirs["train_sample"])
held4.save_to_disk(set_dirs["heldout"])

# ---------------------------------------------------------------------------------------------------------------
print("[2] parity with the launcher's real loss_step on the real training loader (tiny LM, CPU, fp32)")
torch.manual_seed(0)
kw = dict(loaders._lm_kwargs)
kw.update(
    dim=64,
    num_heads=4,
    num_layers=2,
    hidden_scale=2.0,
    context=1024,
    depformer_dim=32,
    depformer_dim_feedforward=64,
    depformer_num_heads=2,
    depformer_num_layers=2,
    dep_q=16,
)
core = ppx_lm.LMModel(device="cpu", dtype=torch.float32, **kw)
apply_lora_to(core, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
with torch.no_grad():
    for n_, p in core.named_parameters():
        if "lora_B" in n_:
            p.normal_(0, 0.05)

# The launcher's loss_step, verbatim: its source cut out of main() and executed with main()'s variables.
ls_node = next(n for n in ast.walk(main_fn) if isinstance(n, ast.FunctionDef) and n.name == "loss_step")
LOSS_STEP_SRC = textwrap.dedent(ast.get_source_segment(open(LAUNCHER).read(), ls_node))
from moshi_family.personaplex.finetune_launcher import _ForwardTrain  # noqa: E402


def launcher_loss_step(cfg: dict, model):
    """``main()``'s variables from ``cfg`` (as main() reads them), then its ``loss_step``."""
    ns = {
        "model": model,
        "device": "cpu",
        "dep_q": core.dep_q,
        "personaplex_loss": personaplex_loss,
        "TEXT_EPAD_ID": TEXT_EPAD_ID,
        "loss_norm": str(cfg.get("loss_norm", "count")),
        "emit_epad": cfg.get("emit_epad"),
        "text_epad_weight": None if cfg.get("text_epad_weight") is None else float(cfg["text_epad_weight"]),
        "floor_weight": None,  # the held-out rows carry no floor weight (documented difference)
        "depformer_skip_masked": bool(cfg.get("depformer_skip_masked", False)),
    }
    exec(LOSS_STEP_SRC, ns)
    return ns["loss_step"]


def training_rows(cfg: dict, set_dir: str, tok) -> list[tuple]:
    """``(codes, total, comps)`` per row via the training loader + the launcher's loss_step, model in TRAIN mode."""
    core.train()
    step = launcher_loss_step(cfg, _ForwardTrain(core))
    out = []
    loader = build_data_loader(
        set_dir,
        tok,
        batch_size=1,
        seed=0,
        system_prompt_key="context",
        voice_codes_key="voice_codes",
        swap_channels=False,
        stored_user_augment=False,
        infinite=False,
    )
    for batch in loader:
        with torch.no_grad():
            total, comps = step(batch)
        out.append((batch[0][0], float(total), comps))
    core.eval()
    return out


def compare(job_rows, train_rows, items) -> tuple[int, float]:
    """(matched rows, max relative difference over the loss and every component)."""
    worst, matched = 0.0, 0
    for codes, total, comps in train_rows:
        j = next(i for i, (c, _m) in enumerate(items) if c.shape == codes.shape and torch.equal(c, codes))
        r = job_rows[j]
        matched += 1
        for k, v in {"loss": total, **comps}.items():
            if v is None or r.get(k) is None:
                continue
            worst = max(worst, abs(r[k] - v) / max(1e-6, abs(v)))
    return matched, worst


results = {}
for cfg_name, cfg_path in CONFIGS.items():
    cfg = yaml.safe_load(open(cfg_path))
    settings = HL.loss_settings(cfg)
    ck(
        settings["text_epad_weight"] == 1.0 and settings["loss_norm"] == "weight",
        f"{cfg_name}: EPAD 1.0, loss_norm weight",
    )
    tok = HL.make_tokenizer({**settings, "hf_repo_id": "nvidia/personaplex-7b-v1"}, "cpu")
    for skip in (False, True):
        s2 = {**settings, "depformer_skip_masked": skip}
        cfg2 = {**cfg, "depformer_skip_masked": skip}
        for set_name, set_dir in set_dirs.items():
            ids_, build = HL.set_items(set_dir, tok)
            items = [build(i) for i in range(len(ids_))]
            n_epad = sum(int(((c[0] == TEXT_EPAD_ID) & m).sum()) for c, m in items)
            job = HL.evaluate_set(core, set_dir, tok, s2, device="cpu")
            tr = training_rows(cfg2, set_dir, tok)
            matched, worst = compare(job, tr, items)
            ck(
                matched == len(job) == len(ids_) and worst < 1e-6,
                f"{cfg_name} skip_masked={skip} {set_name}: job == training loss_step on {matched} windows",
                f"max rel diff {worst:.2e}, {n_epad} EPAD targets",
            )
            ck(n_epad > 0, f"{cfg_name} {set_name}: EPAD targets present (non-vacuous)")
            results[(cfg_name, skip, set_name)] = (job, tr, items)


# ---------------------------------------------------------------------------------------------------------------
def mutant(old: str, new: str, label: str):
    src = open(HL_SRC).read()
    assert old in src, f"mutation anchor gone for {label!r}: {old!r}"
    mod = types.ModuleType("heldout_loss_mutant")
    exec(compile(src.replace(old, new, 1), HL_SRC, "exec"), mod.__dict__)
    return mod


print("[3] eval mode during evaluate_set (spy), and the train-mode mutant is caught")


def spy_training(mod) -> bool:
    """True if any module ran a forward in training mode during mod.evaluate_set."""
    seen = []
    hooks = [m.register_forward_pre_hook(lambda m_, _a: seen.append(m_.training)) for m in core.modules()]
    try:
        cfg = yaml.safe_load(open(CONFIGS["p5"]))
        s = mod.loss_settings(cfg)
        tok = mod.make_tokenizer(s, "cpu")
        core.train()  # the caller's state must not matter
        mod.evaluate_set(core, set_dirs["heldout"], tok, s, device="cpu", limit=1)
    finally:
        for h in hooks:
            h.remove()
    assert seen, "spy saw no forward"
    return any(seen)


ck(not spy_training(HL), "evaluate_set runs every module in eval mode")
ck(
    spy_training(
        mutant(
            "    core.eval()\n    step = make_loss_step", "    core.train()\n    step = make_loss_step", "train mode"
        )
    ),
    "CAN FAIL: a train-mode evaluate_set is flagged",
)

print("[4] CAN FAIL: another EPAD weight / loss_norm breaks the parity")
cfg = yaml.safe_load(open(CONFIGS["p5"]))
for old, new, label in (
    (
        '    if s["text_epad_weight"] is not None:\n',
        '    s["text_epad_weight"] = None\n    if s["text_epad_weight"] is not None:\n',
        "EPAD at the PAD weight",
    ),
    ('    s["loss_norm"] = str(s["loss_norm"])\n', '    s["loss_norm"] = "count"\n', "loss_norm ignored"),
):
    mod = mutant(old, new, label)
    s = mod.loss_settings(cfg)
    tok = mod.make_tokenizer(s, "cpu")
    job = mod.evaluate_set(core, set_dirs["train_sample"], tok, s, device="cpu")
    _, tr, items = results[("p5", True, "train_sample")]
    _, worst = compare(job, tr, items)
    ck(worst > 1e-3, f"mutant ({label}) is caught", f"max rel diff {worst:.2e}")

print("[5b] evaluate_set and summarize are deterministic")
s = HL.loss_settings(cfg)
tok = HL.make_tokenizer(s, "cpu")
torch.manual_seed(123)
np.random.seed(7)
a = HL.evaluate_set(core, set_dirs["heldout"], tok, s, device="cpu")
torch.manual_seed(999)
np.random.seed(99)
_ = torch.rand(10)
b = HL.evaluate_set(core, set_dirs["heldout"], tok, s, device="cpu")
ck(a == b, "evaluate_set twice under different global RNG states: identical numbers")
ck(HL.summarize(a, seed=0) == HL.summarize(b, seed=0), "summarize reproduces its bootstrap CI")
m = HL.summarize(a, seed=0)["loss"]
ck(m["ci95"][0] <= m["mean"] <= m["ci95"][1], "the mean lies inside its CI", str(m))

# ---------------------------------------------------------------------------------------------------------------
print("[6] a checkpoint path is available only once consolidated/config.json is written")
# The job module imports the whole i6 recipe tree (returnn, ...), which this venv lacks: run the function's own source.
_ph_src = open(os.path.join(I6, "persona_heldout.py")).read()
_fn = next(n for n in ast.parse(_ph_src).body if isinstance(n, ast.FunctionDef) and n.name == "_consolidated_complete")
_ns = {"os": os}
exec(ast.get_source_segment(_ph_src, _fn), _ns)
_consolidated_complete = _ns["_consolidated_complete"]


class _P:
    def __init__(self, p):
        self.p = p

    def get_path(self):
        return self.p


d = os.path.join(tmp, "checkpoint_000010", "consolidated")
os.makedirs(d)
w = os.path.join(d, "model.safetensors")
ck(not _consolidated_complete(_P(w)), "nothing written -> not available")
open(w, "wb").close()
ck(not _consolidated_complete(_P(w)), "weights without config.json -> not available")
open(os.path.join(d, "config.json"), "w").write("{}")
ck(_consolidated_complete(_P(w)), "weights + config.json -> available")

print()
if fails:
    print(f"FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("all held-out loss checks passed")
