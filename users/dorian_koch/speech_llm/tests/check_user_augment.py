"""Guard: user-stream augmentation is OFF by default, and can never touch the assistant channel.

The augmentation follows the Moshi paper §4.4, which
applies it to the USER stream only. The invariant that matters most here is invisible in a loss
curve: if the ASSISTANT channel were augmented, we would be changing what the model is trained to
SAY rather than the condition it HEARS, and the run would look entirely healthy while training the
model to emit noisy, reverberant speech.

Also guards the thing that makes this safe to merge at all -- everything defaults to a strict no-op,
so adding it to the loader changes nothing until an arm asks for it -- and the two claims the stored
variants rest on: the reverb IS denoiser's RevEcho (fed the same random numbers as the reference, from
facebookresearch/denoiser at a pinned commit), and conditioning on "some effect fired" keeps the paper's
joint distribution of effects. Section 11 (2026-09-24): the converter clip ``UserAugmentCfg.clip_peak`` -- None
bit-identical to the pre-knob source, 1.0 clips the FINAL waveform (mutants that skip it or clip right
after the gain fail), the ingest's draw records, and the recipe/launcher wiring (RULE_AUG_CLIP).

  CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/.../tests/check_user_augment.py
"""

import sys
from pathlib import Path

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))

from moshi_family.train_data_common import (  # noqa: E402
    UserAugmentCfg,
    augment_user_stream,
    revecho_impulse,
    sample_augment,
)

failures = []


def check(cond, msg, extra=""):
    print(f"{'[ok]' if cond else '[FAIL]'} {msg}{'  ' + extra if extra and not cond else ''}")
    if not cond:
        failures.append(msg)


SR = 24000


def sig(seed=0, n=SR * 3):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


# --- 1. Default is a STRICT no-op -----------------------------------------------------------
user, asst = sig(1), sig(2)
off = UserAugmentCfg()
check(not off.enabled(), "a default UserAugmentCfg reports itself disabled")
out = augment_user_stream(user, asst, SR, np.random.default_rng(0), off)
check(out is user, "disabled augmentation returns the SAME array, not a copy (a strict no-op)")

# --- 2. The ASSISTANT channel is never modified ----------------------------------------------
# The invariant a loss curve cannot show. Every family is exercised, including the echo -- which is
# the one that READS the assistant channel and so is the one that could plausibly write to it.
on_all = UserAugmentCfg(gain_prob=1.0, echo_reverb_prob=1.0)
asst_before = asst.copy()
user_before = user.copy()
got = augment_user_stream(user, asst, SR, np.random.default_rng(0), on_all)
check(np.array_equal(asst, asst_before), "the ASSISTANT channel is byte-identical after augmentation")
check(np.array_equal(user, user_before), "the input user array is not mutated in place")
check(got.dtype == np.float32 and got.shape == user.shape, "output keeps dtype float32 and shape")

# --- 3. Enabled augmentation actually does something (non-vacuity) ---------------------------
check(not np.allclose(got, user), "enabled augmentation actually changes the user signal")

gain_only = UserAugmentCfg(gain_prob=1.0)
g = augment_user_stream(user, asst, SR, np.random.default_rng(7), gain_only)
# A pure gain is a scalar multiple; anything else means the wrong operator ran.
ratio = g[np.abs(user) > 1e-3] / user[np.abs(user) > 1e-3]
check(float(np.std(ratio)) < 1e-4, "gain-only is a single scalar multiple of the input")

echo_only = UserAugmentCfg(echo_reverb_prob=1.0)
e = augment_user_stream(user, asst, SR, np.random.default_rng(7), echo_only)
check(not np.allclose(e, user), "echo+reverb changes the signal")
# Both operators are causal and DELAYED, so the signal before the earliest delay must be untouched.
# Derive the bound from the config rather than hardcoding it: reverb's first echo lands after
# first_delay minus its jitter (9 ms by default), EARLIER than the echo's 100 ms minimum -- a hardcoded
# 50 ms failed here for that reason, and a guard that needs editing whenever a default moves is a
# guard that will be silenced rather than fixed.
_lead_ms = min(echo_only.reverb_first_delay_min * (1 - echo_only.reverb_jitter) * 1000.0, echo_only.echo_delay_min_ms)
head = int((_lead_ms / 1000.0) * 0.8 * SR)
check(
    np.allclose(e[:head], user[:head], atol=1e-5),
    f"the first {_lead_ms * 0.8:.0f} ms is unchanged -- echo/reverb are causal, delayed operators",
)

# --- 4. Probabilities are respected ----------------------------------------------------------
never = UserAugmentCfg(gain_prob=0.0, echo_reverb_prob=0.0)
check(
    augment_user_stream(user, asst, SR, np.random.default_rng(0), never) is user,
    "all-zero probabilities is the same no-op as the default",
)

# --- 5. Determinism: same seed -> same output; different seed -> different -------------------
a1 = augment_user_stream(user, asst, SR, np.random.default_rng(42), on_all)
a2 = augment_user_stream(user, asst, SR, np.random.default_rng(42), on_all)
a3 = augment_user_stream(user, asst, SR, np.random.default_rng(43), on_all)
check(np.array_equal(a1, a2), "same rng seed reproduces the augmentation exactly")
check(not np.array_equal(a1, a3), "a different seed gives a different augmentation (not a constant)")

# --- 6. Noise without a staged corpus FAILS LOUD ---------------------------------------------
# A silent skip here would render a config saying "30% noise" that applies none, and any null result
# would be attributed to the augmentation rather than to its absence.
try:
    augment_user_stream(user, asst, SR, np.random.default_rng(0), UserAugmentCfg(noise_prob=0.5))
    check(False, "noise_prob without a noise_pool must raise")
except ValueError:
    check(True, "noise_prob without a staged noise corpus raises instead of silently doing nothing")

# --- 7. The recipe-facing config emits NOTHING at its default --------------------------------
# This is what keeps every finished arm's hash -- verified empty across all 2192 jobs on 2026-09-18.
from speech_llm.full_duplex.sis_recipe.doriank.train_config import AudioAugment  # noqa: E402

check(AudioAugment().to_hparams() == {}, "AudioAugment() lowers to NO hparams keys")
check(AudioAugment(gain_prob=0.5).to_hparams() == {"aug_gain_prob": 0.5}, "a set probability lowers to exactly one key")

# --- 8. The loader default keeps augmentation off ---------------------------------------------
from moshi_family.moshi_train_data import MoshiDataConfig  # noqa: E402

check(not MoshiDataConfig().user_augment.enabled(), "MoshiDataConfig() leaves user augmentation disabled")

# --- 9. The reverb IS denoiser's RevEcho ---------------------------------------------------------
# The reference `RevEcho._reverb` shifts the signal once per echo and draws its jitter from Python's
# `random`; ours collects the echoes as taps and convolves once. Fed the SAME random numbers, they must
# give the same reverb. The reference is fetched at a pinned commit (CC-BY-NC, so not vendored).
import importlib  # noqa: E402
import tempfile  # noqa: E402
import urllib.request  # noqa: E402

import scipy.signal  # noqa: E402
import torch  # noqa: E402

DENOISER_COMMIT = "8afd7c166699bb3c8b2d95b6dd706f71e1075df0"
_pkg = Path(tempfile.gettempdir()) / f"denoiser_{DENOISER_COMMIT[:8]}" / "denoiser"
if not (_pkg / "augment.py").exists():
    _pkg.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/facebookresearch/denoiser/{DENOISER_COMMIT}/denoiser/augment.py",
        _pkg / "augment.py",
    )
    (_pkg / "__init__.py").write_text("")
    (_pkg / "dsp.py").write_text("")  # augment.py imports it; RevEcho does not use it
sys.path.insert(0, str(_pkg.parent))
ref = importlib.import_module("denoiser.augment")


class _Replay:
    """Hands out one fixed sequence of U(-1, 1) numbers, to the reference (as `random`) and to ours."""

    def __init__(self, xs):
        self._it = iter(xs)

    def uniform(self, a, b):
        assert (a, b) == (-1, 1), (a, b)
        return next(self._it)


jit = np.random.default_rng(5).uniform(-1, 1, 20000).tolist()
x = sig(3, SR * 2)
for initial, first_delay, rt60 in ((0.25, 0.02, 0.9), (0.1, 0.012, 1.3)):
    ref.random = _Replay(jit)
    want = (
        ref.RevEcho(repeat=3, jitter=0.1, sample_rate=SR)
        ._reverb(torch.from_numpy(x.copy())[None, None], initial, first_delay, rt60)[0, 0]
        .numpy()
    )
    ir = revecho_impulse(
        SR, _Replay(jit), initial=initial, first_delay=first_delay, rt60=rt60, repeat=3, jitter=0.1, length=x.shape[0]
    )
    got = scipy.signal.oaconvolve(x, ir)[: x.shape[0]]
    err = float(np.abs(got - want).max())
    check(err < 1e-4, f"revecho_impulse == denoiser RevEcho._reverb (rt60 {rt60}, max |diff| {err:.1e})")

# --- 10. Stored variants: conditioning on "some effect" keeps the paper's joint distribution --------
paper = UserAugmentCfg(gain_prob=0.5, noise_prob=0.3, echo_reverb_prob=0.3)
rng = np.random.default_rng(0)
free = [sample_augment(rng, paper).effects() for _ in range(20000)]
cond = [sample_augment(rng, paper, require_effect=True).effects() for _ in range(20000)]
p_clean = sum(not e for e in free) / len(free)
check(
    abs(p_clean - paper.clean_prob()) < 0.01,
    f"unconditioned draws are clean {p_clean:.3f} ~ clean_prob {paper.clean_prob():.3f}",
)
check(all(cond), "require_effect never returns a clean draw")
for name, p in (("gain", 0.5), ("noise", 0.3), ("echo_reverb", 0.3)):
    share = sum(name in e for e in cond) / len(cond)
    want = p / (1 - paper.clean_prob())
    check(abs(share - want) < 0.015, f"P({name} | some effect) {share:.3f} ~ {want:.3f}")
try:
    sample_augment(rng, UserAugmentCfg(), require_effect=True)
    check(False, "require_effect with all probabilities 0 must raise")
except ValueError:
    check(True, "require_effect with all probabilities 0 raises instead of looping")

# --- 11. The converter clip (UserAugmentCfg.clip_peak; 2026-09-24) -----------------------------
# Augmentation clips the gained values to +-1.0, as happens in real data. A real
# microphone chain saturates at the converter, on the SUM it picked up (voice at its gain + room noise +
# loudspeaker echo + reverb), so the clip must be the LAST step. Checked here: [a] clip_peak=None is
# bit-identical to the implementation before the knob existed (its real source at a pinned commit), and
# records nothing; [b] clip_peak=1.0 bounds every output, really clips a +15 dB draw of a -24 LUFS
# speech-like channel (non-vacuous), equals np.clip of the unclipped chain (clip is last) and leaves
# in-range audio alone (not a coin); [c] mutants of the real source -- no clip, clip right after the gain
# -- fail [b]; [d] the ingest's draw records carry the clip; [e] the recipe/launcher wiring: hash-silent
# at None, on by default only for augmenting runs at RULE_AUG_CLIP, PODCAST_INGEST frozen.
import dataclasses  # noqa: E402
import inspect  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import textwrap  # noqa: E402
import types  # noqa: E402

import pyloudnorm  # noqa: E402

from moshi_family import podcast_mimi_encode as pme  # noqa: E402
from moshi_family import train_data_common as tdc  # noqa: E402

print("[11] converter clip (clip_peak)")
FD = SETUP / "recipe" / "speech_llm" / "full_duplex"
PRE_CLIP_COMMIT = "de5b68a2"  # full_duplex HEAD before clip_peak existed
_old_src = subprocess.run(
    ["git", "-C", str(FD), "show", f"{PRE_CLIP_COMMIT}:./moshi_family/train_data_common.py"],
    capture_output=True,
    text=True,
    check=True,
).stdout
pre = sys.modules["train_data_common_pre_clip"] = types.ModuleType("train_data_common_pre_clip")
exec(compile(_old_src, "<train_data_common @ pre-clip>", "exec"), pre.__dict__)
check(
    not hasattr(pre.UserAugmentCfg(), "clip_peak"), f"the reference really is the pre-knob source ({PRE_CLIP_COMMIT})"
)


def speechlike(seed, sec=8.0):
    """Harmonic voice with a gliding f0, syllable envelope and pauses, at -24 LUFS (the ingest's target)."""
    r = np.random.default_rng(seed)
    t = np.arange(int(SR * sec)) / SR
    phase = 2 * np.pi * np.cumsum(120 + 30 * np.sin(2 * np.pi * 0.7 * t)) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 30))
    env = np.clip(np.sin(2 * np.pi * 3.0 * t + r.uniform(0, 6.28)), 0, None) ** 2
    gate = np.repeat(r.random(int(sec * 2) + 1) < 0.8, SR // 2)[: t.size]
    x = voiced * env * gate + 0.02 * r.standard_normal(t.size) * env
    meter = pyloudnorm.Meter(SR)
    return pyloudnorm.normalize.loudness(x, meter.integrated_loudness(x), -24.0).astype(np.float32)


class _Pool:
    """The NoisePool surface build_noise_track uses (len, lengths, read_many), in memory."""

    def __init__(self):
        r = np.random.default_rng(9)
        self.clips = [(0.1 * r.standard_normal(SR * (2 + i))).astype(np.float32) for i in range(4)]

    def __len__(self):
        return len(self.clips)

    def lengths(self):
        return np.array([c.size for c in self.clips])

    def read_many(self, idx):
        return [self.clips[i] for i in idx]


POOL = _Pool()
va, vb = speechlike(11), speechlike(12)
lufs = pyloudnorm.Meter(SR).integrated_loudness(va.astype(np.float64))
print(f"  speech-like channel: {lufs:.2f} LUFS, peak {20 * np.log10(np.abs(va).max()):.1f} dBFS")

# [a] None == the pre-knob implementation, bit for bit, over paper-rate and all-on draws with noise.
same, n_draws = True, 0
for fields_ in (
    dict(gain_prob=0.5, noise_prob=0.3, echo_reverb_prob=0.3),
    dict(gain_prob=1.0, noise_prob=1.0, echo_reverb_prob=1.0),
):
    new_cfg, old_cfg = tdc.UserAugmentCfg(**fields_), pre.UserAugmentCfg(**fields_)
    for s in range(12):
        d_new = tdc.sample_augment(np.random.default_rng(s), new_cfg, require_effect=True)
        d_old = pre.sample_augment(np.random.default_rng(s), old_cfg, require_effect=True)
        st = {}
        y_new = tdc.apply_augment(va, vb, SR, d_new, new_cfg, noise_pool=POOL, stats=st)
        y_old = pre.apply_augment(va, vb, SR, d_old, old_cfg, noise_pool=POOL)
        same &= dataclasses.asdict(d_new) == dataclasses.asdict(d_old) and np.array_equal(y_new, y_old) and st == {}
        z_new = tdc.augment_user_stream(va, vb, SR, np.random.default_rng(s), new_cfg, POOL)
        z_old = pre.augment_user_stream(va, vb, SR, np.random.default_rng(s), old_cfg, POOL)
        same &= np.array_equal(z_new, z_old)
        n_draws += 1
check(
    same,
    f"clip_peak=None: draws, apply_augment and augment_user_stream bit-identical to {PRE_CLIP_COMMIT} ({n_draws} draws), no stats",
)

# [b] clip_peak=1.0
CLIP = tdc.UserAugmentCfg(gain_prob=0.5, noise_prob=0.3, echo_reverb_prob=0.3, clip_peak=1.0)
NOCLIP = dataclasses.replace(CLIP, clip_peak=None)
G15 = tdc.AugmentDraw(seed=1, gain_db=15.0)
FULL = tdc.AugmentDraw(
    seed=2, gain_db=15.0, noise_db=0.0, echo_scale=0.2, echo_delay_ms=200.0,
    reverb_initial=0.3, reverb_first_delay=0.02, reverb_rt60=1.0,
)  # fmt: skip
ECHO_ONLY = tdc.AugmentDraw(
    seed=3, echo_scale=0.1, echo_delay_ms=150.0, reverb_initial=0.1, reverb_first_delay=0.02, reverb_rt60=0.5
)


def clip_checks(fn, label, verbose=True):
    """The properties a converter clip must have, for an apply_augment-shaped ``fn`` -> list of failures."""
    bad = []
    st = {}
    y = fn(va, vb, SR, G15, CLIP, noise_pool=POOL, stats=st)
    raw = pre.apply_augment(va, vb, SR, G15, pre.UserAugmentCfg(), noise_pool=POOL)
    if not (float(np.abs(raw).max()) > 1.0):
        bad.append("vacuous: the +15 dB draw does not exceed full scale before the clip")
    if not (float(np.abs(y).max()) <= 1.0):
        bad.append(f"+15 dB draw: max|y| {float(np.abs(y).max()):.2f} > 1")
    if not (st.get("clipped") is True and st.get("clipped_share", 0) > 0 and st.get("peak_before_clip", 0) > 1):
        bad.append(f"+15 dB draw: stats do not record the clip ({st})")
    for d in (G15, FULL, ECHO_ONLY):
        y = fn(va, vb, SR, d, CLIP, noise_pool=POOL)
        want = np.clip(tdc.apply_augment(va, vb, SR, d, NOCLIP, noise_pool=POOL), -1.0, 1.0)
        if not np.array_equal(y, want):
            bad.append(f"{d.effects()}: output != np.clip(unclipped chain) -- the clip is not the LAST step")
        if not float(np.abs(y).max()) <= 1.0:
            bad.append(f"{d.effects()}: max|y| > 1")
    st = {}
    y = fn(va, vb, SR, ECHO_ONLY, CLIP, noise_pool=POOL, stats=st)
    if not (
        np.array_equal(y, tdc.apply_augment(va, vb, SR, ECHO_ONLY, NOCLIP, noise_pool=POOL))
        and st.get("clipped") is False
    ):
        bad.append(f"an in-range draw was touched or recorded as clipped ({st})")
    if verbose:
        share = {}
        fn(va, vb, SR, G15, CLIP, noise_pool=POOL, stats=share)
        print(f"  {label}: +15 dB draw peak {share.get('peak_before_clip', float('nan')):.2f} before, "
              f"{share.get('clipped_share', 0):.1%} of samples clipped")  # fmt: skip
    return bad


bad = clip_checks(tdc.apply_augment, "real")
check(
    not bad,
    "clip_peak=1.0: bounded, non-vacuous (+15 dB on -24 LUFS clips), == np.clip(chain), in-range untouched",
    "; ".join(bad),
)
rng = np.random.default_rng(0)
worst = max(
    float(
        np.abs(
            tdc.apply_augment(va, vb, SR, tdc.sample_augment(rng, CLIP, require_effect=True), CLIP, noise_pool=POOL)
        ).max()
    )
    for _ in range(40)
)
check(worst <= 1.0, f"40 paper-rate draws: max|y| {worst:.4f} <= 1.0")


# [c] mutants of the REAL source must fail the same checks
def mutated(name, *subs):
    """Function ``name`` of train_data_common re-compiled from its real source with ``(old, new)`` swaps."""
    src = textwrap.dedent(inspect.getsource(getattr(tdc, name)))
    for old, new in subs:
        assert old in src, f"mutation anchor not found: {old!r}"
        src = src.replace(old, new)
    g = dict(vars(tdc))
    exec(compile(src, f"<mutated {name}>", "exec"), g)
    return g[name]


_FINAL = "if cfg.clip_peak is not None:\n        # The converter"
_OFF = "if False:\n        # The converter"
_GAIN = "out = out * np.float32(10.0 ** (draw.gain_db / 20.0))"
_GAIN_CLIP = _GAIN + "\n        out = np.clip(out, -np.float32(cfg.clip_peak), np.float32(cfg.clip_peak))"
no_clip = mutated("apply_augment", (_FINAL, _OFF))
early = mutated("apply_augment", (_FINAL, _OFF), (_GAIN, _GAIN_CLIP))
for label, fn in (("no clip at all", no_clip), ("clip right after the gain (before noise/echo)", early)):
    got = clip_checks(fn, label, verbose=False)
    check(bool(got), f"can fail: mutant '{label}' is caught", got[0] if got else "NOT CAUGHT")

# [d] the ingest's draw records (podcast_mimi_encode.user_stream_variants, stub encode)
seen = []


def _encode(a, b):
    seen.append(max(float(np.abs(a).max()), float(np.abs(b).max())))
    return np.zeros((8, 3), np.int16), np.zeros((8, 3), np.int16)


_aug = {"gain_prob": 1.0, "gain_min_db": 5.0, "echo_reverb_prob": 0.3}  # loud draws, so some must clip
_, _, m0 = pme.user_stream_variants(
    va, vb, encode=_encode, cfg={"variants": 3, "seed": 0, "augment": _aug}, item_id="g"
)
fields0 = {f.name for f in dataclasses.fields(tdc.AugmentDraw)}
check(
    "clip_peak" not in m0 and all(set(d) == fields0 for c in "ab" for d in m0[c]),
    "ingest without clip_peak: draw records exactly as before (AugmentDraw fields only)",
)
seen.clear()
_, _, m1 = pme.user_stream_variants(
    va, vb, encode=_encode, cfg={"variants": 3, "seed": 0, "augment": {**_aug, "clip_peak": 1.0}}, item_id="g"
)
recs = [d for c in "ab" for d in m1[c]]
check(
    m1.get("clip_peak") == 1.0
    and all({"clipped", "clipped_share", "peak_before_clip"} <= set(d) for d in recs)
    and all(d["clipped"] == (d["peak_before_clip"] > 1.0) for d in recs)
    and any(d["clipped"] for d in recs)
    and max(seen) <= 1.0,
    "ingest with clip_peak=1.0: every draw records clipped/clipped_share/peak_before_clip, some clip, encoder sees <= 1.0",
    str([(round(d["gain_db"], 1), d["clipped"], round(d["clipped_share"], 4)) for d in recs]),
)

# [e] recipe + launcher wiring
import sisyphus.job  # noqa: E402
import yaml  # noqa: E402
from sisyphus import tk  # noqa: E402

from speech_llm.full_duplex.sis_recipe.doriank import runs, synthetic_train_data  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank.train_config import MOSHI_LIB, Corpus  # noqa: E402

os.environ.setdefault("CUDA_HOME", "/usr")
check(
    AudioAugment(gain_prob=0.5).to_hparams() == {"aug_gain_prob": 0.5}
    and AudioAugment(clip_peak=1.0).to_hparams() == {"aug_clip_peak": 1.0},
    "AudioAugment: clip_peak=None emits nothing, 1.0 emits exactly aug_clip_peak",
)
check(
    runs.RULE_AUG_CLIP == runs.POLICY_2026_09_24
    and runs.POLICY_LATEST >= runs.RULE_AUG_CLIP
    and runs.AUG_CLIP_PEAK == 1.0,
    "RULE_AUG_CLIP = 2026-09-24 (a literal epoch), included in POLICY_LATEST, ceiling 1.0",
)
_CORPUS = Corpus(mix=tk.Path("/dev/null/corpus", hash_overwrite="aug_clip_guard_corpus"), duration_sec=60)


def _build(policy, extra=None):
    sisyphus.job.created_jobs.clear()
    return runs.train(
        "aug_clip_guard", model=MOSHI_LIB, corpus=_CORPUS, policy=policy, steps=100,
        extra_hparams=extra, evals=False, register=False,
    ).job  # fmt: skip


def _cfg(job):
    return yaml.safe_load(job.adapter.render_config(job, job.adapter.batch_size, 100))


_aug_hp = AudioAugment(gain_prob=0.5, echo_reverb_prob=0.3).to_hparams()
j_new, j_old = _build(runs.POLICY_2026_09_24, _aug_hp), _build(runs.POLICY_2026_09_23, _aug_hp)
check(
    j_new.hparams.get("aug_clip_peak") == 1.0 and _cfg(j_new)["aug_clip_peak"] == 1.0,
    "an augmenting arm at 2026-09-24 gets aug_clip_peak 1.0, rendered",
)
check(
    "aug_clip_peak" not in j_old.hparams and _cfg(j_old)["aug_clip_peak"] is None,
    "the same arm at 2026-09-23 carries no key and renders null (no clip)",
)
j_plain_new, j_plain_old = _build(runs.POLICY_2026_09_24), _build(runs.POLICY_2026_09_23)
check(
    "aug_clip_peak" not in j_plain_new.hparams and j_plain_new._sis_id() == j_plain_old._sis_id(),
    "a NON-augmenting arm gets no key: 2026-09-24 hashes like 2026-09-23",
)
check(
    _build(runs.POLICY_2026_09_24, {**_aug_hp, "aug_clip_peak": 1e9}).hparams["aug_clip_peak"] == 1e9,
    "an arm that states aug_clip_peak keeps it (ablation possible)",
)
_launcher = (FD / "moshi_family" / "moshi_finetune_launcher.py").read_text()
check(
    'cfg.get("aug_clip_peak")' in _launcher and "clip_peak=aug_clip_peak" in _launcher,
    "base-Moshi launcher reads aug_clip_peak and passes it to UserAugmentCfg",
)
_pi, _pi_new = synthetic_train_data.PODCAST_INGEST, synthetic_train_data.PODCAST_INGEST_2026_09_24
check(
    "clip_peak" not in _pi["user_augment"]["augment"]
    and _pi_new["user_augment"]["augment"] == {**_pi["user_augment"]["augment"], "clip_peak": 1.0}
    and {k: v for k, v in _pi_new.items() if k != "user_augment"}
    == {k: v for k, v in _pi.items() if k != "user_augment"}
    and {k: v for k, v in _pi_new["user_augment"].items() if k != "augment"}
    == {k: v for k, v in _pi["user_augment"].items() if k != "augment"},
    "PODCAST_INGEST frozen (no clip); PODCAST_INGEST_2026_09_24 = it + clip_peak 1.0, nothing else",
)

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("all user-augmentation checks passed")
