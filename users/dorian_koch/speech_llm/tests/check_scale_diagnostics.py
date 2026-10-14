"""Guard: the input-embedding-norm and text-head log Z metrics (``train_data_common.ScaleDiagnostics``, 2026-10-14).

What they are for: another group saw a tied text embedding grow during finetuning until, after the first RMSNorm, the
summed audio embedding was only a small perturbation and the model ignored audio. ``metrics.train.jsonl`` now carries,
per logged step, ``emb_text_l2`` / ``emb_audio_l2`` (mean L2 norm over the batch's (b, t) positions of the text-token
input embedding and of the SUMMED audio-codebook embeddings), their ratio ``emb_text_audio_ratio``, and the text head's
``text_logz_mean`` (mean logsumexp over the CE positions) / ``text_logit_max``. A metric that silently measured the
wrong tensor would be worse than none, so every check drives the REAL code:

  [1] base Moshi (``models/lm.py``, tiny, fp32, CPU, ALL params trained so the embeddings move): the REAL
      ``train_loop.run_training`` (grad_accum 2, log_every 1, 3 steps) over a loss_step shaped like the Moshi
      launcher's (``lm(codes)`` + ``moshi_loss(return_components=True)``), plus a decoy no_grad forward on a DIFFERENT
      batch (the KL teacher's shape: it must not be counted). Every logged record's five keys == an independent
      reference computed on the same weights and batches: the embedding modules called directly on the token inputs
      a pre-hook spy saw, ``torch.logsumexp`` / ``max`` over ``out.text_logits`` at ``text_mask & loss_mask``. The
      batches carry zero-token (-1) inputs in both streams, so the position mask matters.
  [2] the training math is untouched: the same run with ``model=None`` (no hooks) gives bit-identical losses and final
      weights.
  [3] PersonaPlex (``personaplex/models/lm.py`` ``embed_codes``, through ``_ForwardTrain``): the in-loop held-out
      eval's loss step (``heldout_loss.make_loss_step``) under the eval collector (no_grad) == the reference;
      ``heldout_loss.METRICS`` carries the five keys and ``summarize`` reads them.
  [4] wiring (source): every launcher that calls ``run_training(`` passes ``model=``; the RL trainer wraps its
      pi_theta forward in ``scale_diagnostics``; ``duplex_ce_loss`` feeds the active collector.
  [5] CAN FAIL: four in-memory mutants of the REAL ``ScaleDiagnostics`` source (each old string asserted present) must
      each break [1]: audio "sum" = the last codebook only; log Z without the CE term; the no_grad decoy counted;
      the zero-token position mask dropped.

moshi_family venv, CPU, ~20 s:
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd /home/tt201262/setups/2026-01-speech-llm && \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_scale_diagnostics.py'
"""

import ast
import glob
import inspect
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
FULL_DUPLEX = SETUP / "recipe" / "speech_llm" / "full_duplex"
sys.path.insert(0, str(FULL_DUPLEX))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402

from moshi_family import train_data_common as tdc  # noqa: E402
from moshi_family.models import loaders as moshi_loaders  # noqa: E402
from moshi_family.models.lm import LMModel as MoshiLM  # noqa: E402
from moshi_family.moshi_train_data import moshi_loss  # noqa: E402
from moshi_family.personaplex import heldout_loss as HL  # noqa: E402
from moshi_family.personaplex.finetune_launcher import _ForwardTrain  # noqa: E402
from moshi_family.personaplex.models import loaders as ppx_loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMModel as PpxLM  # noqa: E402
from moshi_family.train_loop import TrainConfig, run_training  # noqa: E402

MOSHI_FAMILY = FULL_DUPLEX / "moshi_family"
TINY = dict(
    dim=32,
    num_heads=2,
    num_layers=2,
    hidden_scale=2.0,
    text_card=100,
    context=64,
    depformer_dim=16,
    depformer_dim_feedforward=32,
    depformer_num_heads=2,
    depformer_num_layers=1,
)
KEYS = tdc.SCALE_DIAG_KEYS
REL = 1e-5

fails = []


def ck(cond, label, extra=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not cond:
        fails.append(label)


def close(a, b, rel=REL):
    return a is not None and b is not None and abs(a - b) <= rel * max(1.0, abs(b))


def make_batch(seed, n_q, T=20, B=2):
    """Random codes with zero-token (-1) inputs in the text row and in every audio row (so the mask matters)."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 50, (B, n_q, T), generator=g)
    codes[:, 0] = torch.randint(0, 100, (B, T), generator=g)
    codes[0, 0, 3:6] = -1  # text zero tokens
    codes[1, 1:, 9:12] = -1  # every audio codebook zero at these frames
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, :2] = False  # a loss-masked prefix (PersonaPlex's prompt frames): not a CE position
    return codes, mask


class InputSpy:
    """Forward-pre-hooks recording the token tensors the embedding modules are CALLED with (independent of the
    diagnostics' own forward hooks)."""

    def __init__(self, core):
        self.core = core
        self.text, self.audio = [], {}
        self.h = [core.text_emb.register_forward_pre_hook(lambda m, a: self.text.append(a[0].clone()))]
        for k, e in enumerate(core.emb):
            self.h.append(e.register_forward_pre_hook(lambda m, a, k=k: self.audio.setdefault(k, []).append(a[0])))

    def close(self):
        for h in self.h:
            h.remove()


@torch.no_grad()
def reference(core, call, codes, mask):
    """Independent numbers for one forward: ``call(codes) -> LMOutput`` run once under a spy, the embedding modules
    re-applied DIRECTLY to the tokens it saw, logsumexp / max over the masked text logits."""
    spy = InputSpy(core)
    try:
        out = call(codes)
    finally:
        spy.close()
    assert len(spy.text) == 1, len(spy.text)
    tok = spy.text[0]
    yt = core.text_emb(tok).float()
    ks = sorted(spy.audio)
    ya = sum(core.emb[k](spy.audio[k][0]).float() for k in ks)
    pos = (tok != -1) & torch.stack([spy.audio[k][0] != -1 for k in ks]).any(0)
    tn = yt.norm(dim=-1)[pos]
    an = ya.norm(dim=-1)[pos]
    lg = out.text_logits[:, 0].float()
    valid = out.text_mask[:, 0] & mask
    return {
        "t_sum": float(tn.sum()),
        "a_sum": float(an.sum()),
        "n_pos": int(pos.sum()),
        "lz_sum": float(torch.logsumexp(lg[valid], dim=-1).sum()),
        "n_ce": int(valid.sum()),
        "max": float(lg[valid].max()),
    }


def combine(refs):
    t, a, n = sum(r["t_sum"] for r in refs), sum(r["a_sum"] for r in refs), sum(r["n_pos"] for r in refs)
    lz, nc = sum(r["lz_sum"] for r in refs), sum(r["n_ce"] for r in refs)
    return {
        "emb_text_l2": t / n,
        "emb_audio_l2": a / n,
        "emb_text_audio_ratio": t / a,
        "text_logz_mean": lz / nc,
        "text_logit_max": max(r["max"] for r in refs),
    }


def moshi_run(*, with_model=True, steps=3, grad_accum=2):
    """[1]: the real run_training on a tiny base-Moshi LM. -> (records, per-step reference dicts, final params)."""
    torch.manual_seed(0)
    kw = dict(moshi_loaders._lm_kwargs)
    kw.update(TINY)
    core = MoshiLM(device="cpu", dtype=torch.float32, **kw)
    core.train()
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-2)
    n_q = core.num_codebooks
    batches = [make_batch(100 + i, n_q) for i in range(steps * grad_accum)]
    decoy = make_batch(999, n_q)[0]
    decoy[:, 1:] = torch.randint(0, 3, decoy[:, 1:].shape)  # a different input: a counted decoy moves the means
    refs, calls = {}, {"n": 0}

    def loss_step(batch):
        codes, m = batch
        step = calls["n"] // grad_accum + 1
        calls["n"] += 1
        refs.setdefault(step, []).append(reference(core, core, codes, m))
        out = core(codes)  # the launcher's `lm(model_in)`
        loss, parts = moshi_loss(out, codes, m, dep_q=core.dep_q, return_components=True)
        with torch.no_grad():  # the KL teacher's shape: a grad-free forward through the same modules
            core(decoy)
        return loss, parts

    with tempfile.TemporaryDirectory(prefix="check_scale_diag_") as d:
        run_training(
            optimizer=opt,
            base_lrs=[3e-2],
            group_names=("all",),
            trainable_params=params,
            batch_iter=iter(batches),
            loss_step=loss_step,
            save_fn=lambda step, final: None,
            cfg=TrainConfig(
                max_steps=steps, grad_accum=grad_accum, warmup_steps=1, grad_clip=1.0, save_every=0, log_every=1
            ),
            out_dir=d,
            log=lambda m: None,
            model=core if with_model else None,
        )
        recs = [json.loads(x) for x in (Path(d) / "metrics.train.jsonl").read_text().splitlines() if x.strip()]
    return recs, {s: combine(r) for s, r in refs.items()}, [p.detach().clone() for p in core.parameters()]


def problems_moshi():
    """[1] -> list of mismatches (empty = pass)."""
    recs, refs, _ = moshi_run()
    out = []
    if len(recs) != 3:
        return [f"{len(recs)} records, expected 3"]
    for r in recs:
        exp = refs[r["step"]]
        for k in KEYS:
            if not close(r.get(k), exp[k]):
                out.append(f"step {r['step']} {k}: logged {r.get(k)} reference {exp[k]}")
    return out


# ---------------------------------------------------------------------------------------------------------------
print("[1] base Moshi through the real run_training: the five keys == the independent reference, every step")
recs, refs, w_with = moshi_run()
ck(len(recs) == 3, "3 metrics records (log_every 1)", str(len(recs)))
for r in recs:
    exp = refs[r["step"]]
    for k in KEYS:
        ck(close(r.get(k), exp[k]), f"step {r['step']} {k}", f"logged {r.get(k)!r} reference {exp[k]!r}")
ratios = [r["emb_text_audio_ratio"] for r in recs]
ck(
    len(set(round(x, 9) for x in ratios)) == 3,
    "the ratio moves between steps (current weights, not a constant)",
    str(ratios),
)
ck(all(isinstance(r.get(k), float) for r in recs for k in KEYS), "all five are plain floats (scalars only)")
ck("text_loss" in recs[0] and "grad_norm" in recs[0], "the existing keys are still there")

print("[2] the hooks do not change the training math")
recs0, _, w_without = moshi_run(with_model=False)
ck([r["loss"] for r in recs0] == [r["loss"] for r in recs], "losses bit-identical with model=None")
ck(all(torch.equal(a, b) for a, b in zip(w_with, w_without)), "final weights bit-identical with model=None")
ck(all("emb_text_l2" not in r and "text_logz_mean" in r for r in recs0), "model=None: logit keys only")

# ---------------------------------------------------------------------------------------------------------------
print("[3] PersonaPlex embed_codes + the held-out eval's loss step, eval collector under no_grad")
torch.manual_seed(1)
kw = dict(ppx_loaders._lm_kwargs)
kw.update(TINY, dep_q=16)
pcore = PpxLM(device="cpu", dtype=torch.float32, **kw)
pcore.eval()
fwd = _ForwardTrain(pcore)
step_fn = HL.make_loss_step(fwd, device="cpu", dep_q=16, settings=HL.loss_settings({}))
pcodes, pmask = make_batch(7, pcore.num_codebooks)
exp = combine([reference(pcore, fwd, pcodes, pmask)])
diag = tdc.ScaleDiagnostics()
with torch.no_grad(), tdc.scale_diagnostics(diag, model=pcore):
    _total, comps = step_fn((pcodes, pmask))
got = diag.scalars()
for k in KEYS:
    ck(close(got.get(k), exp[k]), f"ppx {k}", f"got {got.get(k)!r} reference {exp[k]!r}")
ck(all(k in HL.METRICS for k in KEYS), "heldout_loss.METRICS carries the five keys")
ck(HL.METRICS[: HL.METRICS.index(KEYS[0])][-1] == "text_pad_loss_share", "appended after the old metrics (seeds kept)")
summ = HL.summarize([{"n_frames": 10, **got}, {"n_frames": 5, **got}])
ck(all(close(summ[k]["mean"], got[k]) for k in KEYS), "summarize reads them")
ck(tdc.active_scale_diagnostics() is None, "no collector left active after the scope")
ck(not any(e._forward_hooks for e in [pcore.text_emb, *pcore.emb]), "no hook left on the embeddings after the scope")

# ---------------------------------------------------------------------------------------------------------------
print("[4] wiring: every run_training launcher passes model=; RL and the CE loss feed the collector")
for path in sorted(glob.glob(str(MOSHI_FAMILY / "**" / "*launcher.py"), recursive=True)):
    tree = ast.parse(open(path).read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_training"]
    if calls:
        ck(
            all(any(k.arg == "model" for k in c.keywords) for c in calls),
            f"{Path(path).relative_to(MOSHI_FAMILY)} run_training(model=...)",
        )
rl_src = (MOSHI_FAMILY / "rl" / "trainer.py").read_text()
ck('phase("fwd_theta"), scale_diagnostics(self._scale_diag, model=self.core)' in rl_src, "RL pi_theta forward wrapped")
ck("_diag.add_text_logits(" in inspect.getsource(tdc.duplex_ce_loss), "duplex_ce_loss feeds the active collector")

# ---------------------------------------------------------------------------------------------------------------
print("[5] CAN FAIL: mutants of the real ScaleDiagnostics break [1]")
CLS_SRC = textwrap.dedent(inspect.getsource(tdc.ScaleDiagnostics))
MUTANTS = {
    "audio sum = last codebook only": (
        "self._audio, self._audio_ok = self._audio + y, self._audio_ok | ok",
        "self._audio, self._audio_ok = y, self._audio_ok | ok",
    ),
    "log Z without the CE term": ("lz = ce.detach().float() + logits", "lz = 0.0 * ce + logits"),
    "no_grad decoy counted": ("return not self.require_grad or torch.is_grad_enabled()", "return True"),
    "zero-token mask dropped": (
        "pos = ((tokens != zero_idx) & audio_ok).float()",
        "pos = torch.ones_like(tokens, dtype=torch.float32)",
    ),
}
orig_cls = tdc.ScaleDiagnostics
for name, (old, new) in MUTANTS.items():
    assert CLS_SRC.count(old) == 1, (name, old)
    ns = dict(vars(tdc))
    exec(compile(CLS_SRC.replace(old, new), f"<mutant {name}>", "exec"), ns)
    tdc.ScaleDiagnostics = ns["ScaleDiagnostics"]  # run_training imports it at call time
    try:
        probs = problems_moshi()
    finally:
        tdc.ScaleDiagnostics = orig_cls
    ck(bool(probs), f"mutant flagged: {name}", probs[0] if probs else "NOT DETECTED")
ck(not problems_moshi(), "unmutated class passes again")

print()
if fails:
    print(f"FAIL ({len(fails)}): " + "; ".join(fails))
    sys.exit(1)
print("PASS")
