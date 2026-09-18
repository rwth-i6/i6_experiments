"""Guard: user-stream augmentation is OFF by default, and can never touch the assistant channel.

Backlog A25. The augmentation follows the Moshi paper §4.4 (quoted in `paper_recipes.md`), which
applies it to the USER stream only. The invariant that matters most here is invisible in a loss
curve: if the ASSISTANT channel were augmented, we would be changing what the model is trained to
SAY rather than the condition it HEARS, and the run would look entirely healthy while training the
model to emit noisy, reverberant speech.

Also guards the thing that makes this safe to merge at all -- everything defaults to a strict no-op,
so adding it to the loader changes nothing until an arm asks for it.

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
# Derive the bound from the config rather than hardcoding it: reverb's first repeat lands at
# reverb_delay_ms (40 ms by default), which is EARLIER than the echo's 100 ms minimum -- a hardcoded
# 50 ms failed here for that reason, and a guard that needs editing whenever a default moves is a
# guard that will be silenced rather than fixed.
_lead_ms = min(echo_only.reverb_delay_ms, echo_only.echo_delay_min_ms)
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

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("all user-augmentation checks passed")
