"""Guard: user-stream augmentation is OFF by default, and can never touch the assistant channel.

Backlog A25. The augmentation follows the Moshi paper §4.4 (quoted in `paper_recipes.md`), which
applies it to the USER stream only. The invariant that matters most here is invisible in a loss
curve: if the ASSISTANT channel were augmented, we would be changing what the model is trained to
SAY rather than the condition it HEARS, and the run would look entirely healthy while training the
model to emit noisy, reverberant speech.

Also guards the thing that makes this safe to merge at all -- everything defaults to a strict no-op,
so adding it to the loader changes nothing until an arm asks for it -- and the two claims the stored
variants rest on: the reverb IS denoiser's RevEcho (fed the same random numbers as the reference, from
facebookresearch/denoiser at a pinned commit), and conditioning on "some effect fired" keeps the paper's
joint distribution of effects.

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


def check(cond, msg):
    print(f"{'[ok]' if cond else '[FAIL]'} {msg}")
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

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("all user-augmentation checks passed")
