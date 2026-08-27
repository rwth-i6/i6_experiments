"""
v14 adds weight sparsity / pruning
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from typing import Callable, Optional, Union, Dict, Literal, List

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
try:
    from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
    from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
except ModuleNotFoundError:
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings


@dataclass
class ThresholdPruningConfig:
    """Prune weights whose absolute value is below a fixed threshold."""
    start_epoch: int
    threshold: float

    def apply(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        return weight * (weight.abs() >= self.threshold).to(weight.dtype)


@dataclass
class PercentilePruningConfig:
    """Prune the bottom `percentile` fraction of weights by absolute value (value in [0, 1])."""
    start_epoch: int
    percentile: float

    def apply(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        cutoff = torch.quantile(weight.abs(), self.percentile)
        return weight * (weight.abs() >= cutoff).to(weight.dtype)


WeightPruningConfig = Union[ThresholdPruningConfig, PercentilePruningConfig]


@dataclass
class GaussianWeightNoiseConfig:
    """Per-bit Gaussian noise simulating memristor read uncertainty."""
    dev: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        # Fused implementation (2026-07-28): the sum of the P-1 independent per-bit Gaussians
        # (std dev*sqrt(2)*2^i each) is itself Gaussian, so draw it once with
        # sigma = dev*sqrt(2*sum_i 4^i). Distribution-identical to the per-bit loop below
        # (KS-verified, sigma matches to 4 digits), but without the P-1 full-size
        # CPU->device std/mean materializations per call that cost +180%/+295% train step
        # time at w4/w8. Previous per-bit implementation, kept for history:
        # for i in range(weight_bit_prec - 1):
        #     mean = 2 * (-weight_quantizer.zero_point).expand(weight.shape).to(weight.device).to(torch.float32)
        #     std = (torch.tensor(self.dev) * (2 ** i)).expand(weight.shape).to(weight.device).to(torch.float32)
        #     std = 2**0.5 * std
        #     noise = torch.normal(mean=mean, std=std).to(weight.device) * weight_quantizer.scale
        #     weight = weight + noise
        # return weight
        n_bits = weight_bit_prec - 1
        sigma = self.dev * (2.0 * sum(4**i for i in range(n_bits))) ** 0.5
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        zero_point = weight_quantizer.zero_point.to(weight.device).to(torch.float32)
        noise = torch.randn(weight.shape, dtype=torch.float32, device=weight.device) * sigma
        # per-bit mean term 2*(-zero_point), summed over the n_bits draws
        # (identically 0 for per_tensor_symmetric qint8)
        noise = (noise + 2.0 * n_bits * (-zero_point)) * scale
        return weight + noise


@dataclass
class BitFlipWeightNoiseConfig:
    """Per-bit random bit-flip noise simulating memristor programming errors.

    Each bit of the quantized weight is independently flipped (0→1 or 1→0)
    with probability `p`, modelling a device being programmed into the wrong state.
    """
    p: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        raise RuntimeError(
            "BitFlipWeightNoiseConfig is deprecated and must not be used: its apply() detaches the "
            "autograd graph (round() + int32 cast), so the quantized weights receive zero gradient "
            "and training cannot converge. Use BitFlipSTEWeightNoiseConfig (straight-through) instead."
        )
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        zero_point = weight_quantizer.zero_point.to(weight.device).to(torch.int32)
        n = weight_bit_prec
        bit_mask = (1 << n) - 1
        # Re-quantize to signed integer, then extract the n-bit pattern
        q_int = (weight.to(torch.float32) / scale).round().to(torch.int32) + zero_point
        q_bits = q_int & bit_mask
        # Flip each bit independently with probability p
        for i in range(n):
            flip_mask = torch.bernoulli(
                torch.full(weight.shape, self.p, dtype=torch.float32, device=weight.device)
            ).to(torch.int32)
            q_bits = q_bits ^ (flip_mask * (1 << i))
        q_bits = q_bits & bit_mask
        # Convert n-bit unsigned back to signed (two's complement)
        sign_bit_val = 1 << (n - 1)
        q_signed = (q_bits ^ sign_bit_val) - sign_bit_val
        # Dequantize
        return (q_signed - zero_point).to(weight.dtype) * scale


@dataclass
class GaussianWeightLevelNoiseConfig:
    """Whole-weight (not per-bit) additive Gaussian noise.

    A single zero-mean Gaussian perturbation is added to each quantized weight, with standard
    deviation `weight_dev` measured in LSB units (i.e. multiplied by the per-tensor quantization
    scale). This is the weight-level analogue of GaussianWeightNoiseConfig: one draw at the weight
    level instead of a sum of per-bit contributions.
    """
    weight_dev: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        std = torch.full(weight.shape, self.weight_dev, dtype=torch.float32, device=weight.device)
        noise = torch.normal(mean=torch.zeros_like(std), std=std) * scale
        return weight + noise


@dataclass
class UniformBitNoiseConfig:
    """Per-bit additive uniform noise (uniform analogue of the per-bit Gaussian).

    For each bit i a uniform perturbation drawn from U(-bit_amplitude, bit_amplitude) is scaled by
    the bit significance 2**i and by the quantization scale, then added to the weight.
    """
    bit_amplitude: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        for i in range(weight_bit_prec - 1):
            noise = torch.empty(weight.shape, dtype=torch.float32, device=weight.device).uniform_(
                -self.bit_amplitude, self.bit_amplitude
            ) * (2 ** i) * scale
            weight = weight + noise
        return weight


@dataclass
class UniformWeightLevelNoiseConfig:
    """Whole-weight additive uniform noise.

    A single perturbation drawn from U(-weight_amplitude, weight_amplitude), measured in LSB units
    (multiplied by the per-tensor quantization scale), is added to each quantized weight.
    """
    weight_amplitude: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        noise = torch.empty(weight.shape, dtype=torch.float32, device=weight.device).uniform_(
            -self.weight_amplitude, self.weight_amplitude
        ) * scale
        return weight + noise


@dataclass
class RelativeGaussianWeightNoiseConfig:
    """Multiplicative (relative) Gaussian weight noise: per-weight std proportional to |weight|.

    Models memristor read noise that scales with the programmed conductance. `rel_dev` is the
    relative standard deviation (a fraction of each weight's magnitude). Zero-mean and analog;
    unlike the LSB-scaled variants this is independent of the quantization scale.
    """
    rel_dev: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        std = self.rel_dev * weight.abs().to(torch.float32)
        noise = torch.normal(mean=torch.zeros_like(std), std=std)
        return weight + noise


@dataclass
class BitMixingWeightNoiseConfig:
    """Deterministic per-bit 'bit-mixing' perturbation.

    Each bit of the quantized weight is pulled toward the opposite state by a fixed fraction
    `mix`: a set bit contributes (1 - mix) * 2**bit instead of 2**bit, and an unset bit contributes
    mix * 2**bit instead of 0. Applied in two's-complement space (the sign bit uses weight
    -2**(n-1)), so mix=0 is the identity. Models incomplete programming / conductance relaxation
    toward the mid-state. Injected as an additive perturbation with a straight-through gradient
    (like the Gaussian variants), so the clean weights keep learning.
    """
    mix: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        zero_point = weight_quantizer.zero_point.to(weight.device)
        n = weight_bit_prec
        bit_mask = (1 << n) - 1
        q_bits = ((weight.to(torch.float32) / scale).round().to(torch.int32) + zero_point.to(torch.int32)) & bit_mask
        mixed = torch.zeros_like(weight, dtype=torch.float32)
        for i in range(n):
            b = ((q_bits >> i) & 1).to(torch.float32)
            m = b * (1.0 - self.mix) + (1.0 - b) * self.mix  # 1 -> 1-mix, 0 -> mix
            w_i = -(2 ** (n - 1)) if i == n - 1 else (2 ** i)  # two's-complement bit weight
            mixed = mixed + m * w_i
        perturbed = ((mixed - zero_point.to(torch.float32)) * scale).to(weight.dtype)
        return weight + (perturbed - weight).detach()


@dataclass
class BitFlipSTEWeightNoiseConfig:
    """Straight-through bit-flip noise (fixed replacement for BitFlipWeightNoiseConfig).

    Same stochastic per-bit flips as BitFlipWeightNoiseConfig (each of the n bits flips
    independently with probability `flip_p` in two's-complement space), but injected as an additive
    perturbation with a straight-through gradient (weight + (flipped - weight).detach()), so the
    quantized weights keep receiving gradients and noise-aware training can actually converge.
    """
    flip_p: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        zero_point = weight_quantizer.zero_point.to(weight.device).to(torch.int32)
        n = weight_bit_prec
        bit_mask = (1 << n) - 1
        q_int = (weight.to(torch.float32) / scale).round().to(torch.int32) + zero_point
        q_bits = q_int & bit_mask
        for i in range(n):
            flip_mask = torch.bernoulli(
                torch.full(weight.shape, self.flip_p, dtype=torch.float32, device=weight.device)
            ).to(torch.int32)
            q_bits = q_bits ^ (flip_mask * (1 << i))
        q_bits = q_bits & bit_mask
        sign_bit_val = 1 << (n - 1)
        q_signed = (q_bits ^ sign_bit_val) - sign_bit_val
        flipped = ((q_signed - zero_point).to(torch.float32) * scale).to(weight.dtype)
        return weight + (flipped - weight).detach()


@dataclass
class StudentTWeightNoiseConfig:
    """
    Heavy-tailed closed-form weight noise: Student-t, fitted to the Synaptogen device noise.

    The device error is Gaussian in magnitude but not in shape -- measured excess kurtosis
    +3.19 (w8) / +3.06 (w4) versus 0 for a Gaussian, i.e. ~1400x more mass beyond 5 sigma
    (observations/synaptogen_explicit_noise_benchmark.md). A Student-t matches that with one
    extra parameter: fitting by moments gives nu ~ 5.9 consistently across tensors and bit
    widths, and drops KL(device || model) from 0.0270 (best Gaussian) to 0.0002.

    Parametrisation: `dev` has the *same meaning* as GaussianWeightNoiseConfig.dev, i.e. the
    total injected sigma is dev*sqrt(2*sum_i 4^i), so StudentT(dev=d, nu) and gauss(dev=d) are
    magnitude-matched and differ only in tail shape. The t scale is set to
    s = sigma*sqrt((nu-2)/nu) so that Var = sigma^2 exactly. nu -> infinity recovers the
    Gaussian, so gauss is the nested null of this config.

    What it does NOT reproduce: the device's heteroscedasticity (sigma grows ~8% with weight
    magnitude) -- that feature is carried by AsymmetricBitNoiseConfig instead, and measurably
    contributes no kurtosis. Together: gauss (magnitude) < student-t (+ tails) <
    synpool (+ heteroscedasticity, exact marginals).

    Options:
        dev: amplitude, as in GaussianWeightNoiseConfig (0.035 ~ the device magnitude,
            0.05 ~ the empirical hardware-WER optimum)
        nu: degrees of freedom; 5.9 = the device fit, larger = closer to Gaussian
            (must be > 2 for the variance to exist; kurtosis exists only for nu > 4)
        start_epoch: as in the other noise configs
    """

    dev: float
    nu: float
    start_epoch: int

    def __post_init__(self):
        assert self.nu > 2.0, f"nu must be > 2 for a finite variance, got {self.nu}"

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx

        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        sigma = self.dev * (2.0 * sum(4**i for i in range(weight_bit_prec - 1))) ** 0.5
        key = (str(weight.device), weight_bit_prec)
        cache = getattr(self, "_dist_cache", None)
        if cache is None:
            cache = self._dist_cache = {}
        if key not in cache:
            # scale chosen so that Var(t) = nu/(nu-2) * s^2 == sigma^2
            s = sigma * ((self.nu - 2.0) / self.nu) ** 0.5
            cache[key] = torch.distributions.StudentT(
                df=torch.tensor(self.nu, dtype=torch.float32, device=weight.device),
                scale=torch.tensor(s, dtype=torch.float32, device=weight.device),
            )
        noise = cache[key].sample(weight.shape)
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        return weight + (noise * scale).to(weight.dtype)


@dataclass
class AsymmetricBitNoiseConfig:
    """
    Zero-mean per-bit Gaussian noise whose *variance* depends on the bit state
    (heteroscedastic): programmed ("ON") bits fluctuate more than unprogrammed ("OFF")
    ones. This is the stochastic descendant of BitMixingWeightNoiseConfig: it keeps
    bitmix's state-dependence but moves it from the *mean* into the *variance*, which is
    what makes it a noise rather than a deterministic reparametrization the optimizer can
    pre-distort away (analysis F6). Note a zero-mean *random mix fraction* would not be a
    new method at all: sign(1-2b) applied to a symmetric variable is distribution-neutral,
    so it reduces exactly to UniformBitNoiseConfig / GaussianWeightNoiseConfig.

    Perturbation per weight: sum_i N(0, sigma_i) * scale over the i = 0..P-2 *magnitude*
    bits, with sigma_i = sqrt(2)*dev * 2^i * (on_scale if bit i is 1 else off_scale).
    Bits are taken from the sign-magnitude decomposition (|round(w/scale)|), i.e. the
    decomposition the hardware actually uses (P-1 magnitude bits, each a differential cell
    pair, sign given by which array holds the bit) -- unlike bitmix/bitflipste, which
    perturb two's-complement bits including a sign bit that corresponds to no physical
    cell. The weight sign is irrelevant here because the noise is symmetric.

    Implemented as a single fused draw: the per-bit Gaussians are independent and
    zero-mean, so their sum is Gaussian with the summed variance (same identity as the
    fused GaussianWeightNoiseConfig, one randn instead of P-1).

    Options:
        dev: base amplitude, identical in meaning to GaussianWeightNoiseConfig.dev
            (0.05 = the empirical hardware-WER optimum, sigma ~ 0.3*rms(w))
        on_scale / off_scale: variance scales for programmed / unprogrammed bits. With
            normalize=True only their *ratio* matters. Reference points: 1.34 / 1.0 is the
            ratio measured on the Synaptogen device model (differential-pair readback sigma
            0.0651 for a programmed bit vs 0.0485 for an unprogrammed one; 1.62 at single
            cell level), 1.0 / 1.0 reduces exactly to GaussianWeightNoiseConfig(dev) and
            0.0 off_scale gives "only programmed cells fluctuate".
        normalize: rescale so the total injected noise power equals the homoscedastic
            (on_scale == off_scale) case at the same dev. Keeps magnitude fixed across the
            ratio sweep, so the experiment isolates noise *allocation* from noise
            *magnitude* (analysis F1: magnitude dominates everything else).
        start_epoch: as in the other noise configs
    """

    dev: float
    on_scale: float
    off_scale: float
    normalize: bool
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx

        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        w32 = weight.to(torch.float32)
        # sign-magnitude decomposition: the P-1 magnitude bits are the physical cells.
        # int cast also detaches, so the noise stays purely additive (gradient = identity).
        mag = torch.round(w32 / scale).abs().to(torch.int64)
        var = torch.zeros_like(w32)
        for i in range(weight_bit_prec - 1):
            b = ((mag >> i) & 1).to(torch.float32)
            s = b * self.on_scale + (1.0 - b) * self.off_scale
            var = var + (2.0 * self.dev**2 * 4.0**i) * (s * s)
        if self.normalize:
            var_ref = 2.0 * self.dev**2 * sum(4.0**i for i in range(weight_bit_prec - 1))
            var = var * (var_ref / var.mean().clamp_min(1e-12))
        noise = torch.randn(weight.shape, dtype=torch.float32, device=weight.device) * var.sqrt()
        return weight + (noise * scale).to(weight.dtype)


@dataclass
class SynaptogenPoolNoiseConfig:
    """
    Device-matched weight noise sampled from the Synaptogen memristor model
    (design + benchmark: observations/synaptogen_explicit_noise_benchmark.md).

    A pool of `pool_size` paired differential readback errors per bit state is
    Monte-Carlo'd once through the exact eval programming pipeline
    (TiledMemristorLinear.init_from_linear_quant semantics: differential pairs,
    `num_cycles`*15 burn-in pulses + RESET + SET, readback at 0.6 V x 8020) — every
    pool entry comes from a distinct freshly drawn device pair, so the pool is a
    device *ensemble*, not one device. Per forward, each (weight, bit-slice) draws
    one pool entry (iid fresh-devices regime; marginals exactly device-matched incl.
    HRS tails, bit=0 leak noise and the ~+1% programming gain), reconstruction
    sum_i 2^i * delta_i, injected via STE. Magnitude: sigma ~ 0.22*rms(w) at w8,
    ~0.24 at w4.

    The lazy pool build (first apply() call, ~10-30 s CPU, 8 MB resident) needs
    `synaptogen_ml` importable inside the job: trainings must pass
    `import_memristor` in train_args (serialize_training), forward jobs use the
    usual eval_model flag.

    Options:
        pool_size: pool entries per bit state (1e6 ample: sigma estimate error ~0.07%)
        num_cycles: burn-in depth, matches num_cycles_init of the eval programming
        read_noise: "none" | "correct" (physical constants) | "v3" (reproduces the
            inflated readout-noise constants of the new_v3 eval pin); read noise is
            second-order vs programming noise in every mode
        strength: multiplier on the injected perturbation (1.0 = device-matched;
            use for magnitude sweeps around the device point)
        parametric: sample per-bit-state Gaussians with moments measured from the
            pool instead of gathering raw entries (drops the mild non-Gaussian
            tails; the pool is still built once for the moments)
        pool_seed: seed for the device sampling (np.random.seed alone does NOT seed
            the synaptogen generator; the module rng is rebound during the build and
            restored afterwards)
        refresh_per_epoch: rebuild the pool each (sub-)epoch with seed
            pool_seed+epoch to refresh the empirical support; off = one pool for
            the whole training (deterministic across restarts either way)
        start_epoch: as in the other noise configs
    """

    pool_size: int
    num_cycles: int
    read_noise: str
    strength: float
    parametric: bool
    pool_seed: int
    refresh_per_epoch: bool
    start_epoch: int

    # plain class attribute (unannotated -> not a dataclass field, stays out of asdict)
    _POOL_CHUNKS = 16  # independent burn-in sequences mixed into the pool

    def _program_fresh(self, syn, np, n, bits):
        """One fresh cell array through the eval programming pipeline; returns the
        per-cell readback in weight units."""
        u_read, i_scale = np.float32(0.6), np.float32(8020.0)
        cells = syn.CellArrayCPU(n)
        for _ in range(self.num_cycles * 15):
            cells.applyVoltage(np.random.uniform(-2.0, 2.0))
        cells.applyVoltage(2.0)
        cells.applyVoltage(bits * -2.0)
        current = cells.I(u_read).astype(np.float32)
        if self.read_noise != "none":
            if self.read_noise == "correct":
                e_, bw = 1.602176634e-19, 1e8
            elif self.read_noise == "v3":
                e_, bw = float(np.exp(1)), 1e-8
            else:
                raise ValueError(f"unknown read_noise mode {self.read_noise!r}")
            kbt = 1.380649e-23 * 300.0
            sigma = np.sqrt(4.0 * kbt * bw * np.abs(current / float(u_read)) + 2.0 * e_ * np.abs(current) * bw)
            current = current + np.random.standard_normal(n).astype(np.float32) * sigma.astype(np.float32)
        return current * i_scale

    def _build_pool(self, seed):
        import time
        from functools import partial
        import numpy as np
        import synaptogen_ml.synaptogen as syn

        old_rng = (syn.rng, syn.randn, syn.rand)
        np_state = np.random.get_state()
        t0 = time.monotonic()
        try:
            rng = np.random.default_rng(seed)
            syn.rng = rng
            syn.randn = partial(rng.standard_normal, dtype=np.float32)
            syn.rand = partial(rng.random, dtype=np.float32)
            np.random.seed(seed)  # the burn-in voltage scalars use the legacy global
            per = max(self.pool_size // self._POOL_CHUNKS, 1)
            p10, p00 = [], []
            for _ in range(self._POOL_CHUNKS):
                n = 2 * per
                bits = np.zeros(n, dtype=np.float32)
                bits[:per] = 1.0
                x = self._program_fresh(syn, np, n, bits)
                y = self._program_fresh(syn, np, n, np.zeros(n, dtype=np.float32))
                p10.append(x[:per] - y[:per])
                p00.append(x[per:] - y[per:])
            pair10 = np.concatenate(p10)
            pair00 = np.concatenate(p00)
        finally:
            syn.rng, syn.randn, syn.rand = old_rng
            np.random.set_state(np_state)
        pool_flat = torch.from_numpy(np.concatenate([pair00, pair10]).astype(np.float32))
        moments = (float(pair10.mean()), float(pair10.std()), float(pair00.mean()), float(pair00.std()))
        print(
            f"SynaptogenPoolNoise: built pool 2x{pair10.size} (seed {seed}) in {time.monotonic() - t0:.1f}s;"
            f" pair10 {moments[0]:.4f}+-{moments[1]:.4f}, pair00 {moments[2]:.4f}+-{moments[3]:.4f}"
        )
        return pool_flat, moments

    def _get_pool(self, seed, device):
        cache = getattr(self, "_pool_cache", None)
        if cache is None or cache[0] != seed:
            pool_flat, moments = self._build_pool(seed)
            cache = (seed, pool_flat, moments, {})  # single-entry cache: old pools are evicted
            self._pool_cache = cache
        dev_cache = cache[3]
        key = str(device)
        if key not in dev_cache:
            dev_cache[key] = cache[1].to(device)
        return dev_cache[key], cache[2]

    def apply(self, weight, weight_quantizer, weight_bit_prec, training):
        from returnn.torch.context import get_run_ctx

        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        seed = self.pool_seed
        if training and self.refresh_per_epoch:
            seed = self.pool_seed + get_run_ctx().epoch
        pool_flat, moments = self._get_pool(seed, weight.device)

        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        q = torch.round(weight.to(torch.float32) / scale)
        sign = q.sign().view(-1)
        mag = q.abs().view(-1).to(torch.int64)
        one = torch.ones((), dtype=torch.float32, device=weight.device)
        acc = torch.zeros_like(sign)
        k = pool_flat.numel() // 2
        mu10, sd10, mu00, sd00 = moments
        for i in range(weight_bit_prec - 1):
            bit = (mag >> i) & 1
            if self.parametric:
                noise = torch.randn(mag.numel(), dtype=torch.float32, device=weight.device)
                delta = torch.where(bit.bool(), noise * sd10 + mu10, noise * sd00 + mu00)
            else:
                idx = bit * k + torch.randint(0, k, (mag.numel(),), device=weight.device)
                delta = pool_flat[idx]
            acc = acc + (2.0**i) * delta * torch.where(bit.bool(), sign, one)
        w_eff = (acc.view(weight.shape) * scale).to(weight.dtype)
        return weight + self.strength * (w_eff - weight).detach()


WeightNoiseConfig = Union[
    GaussianWeightNoiseConfig,
    BitFlipWeightNoiseConfig,
    GaussianWeightLevelNoiseConfig,
    UniformBitNoiseConfig,
    UniformWeightLevelNoiseConfig,
    RelativeGaussianWeightNoiseConfig,
    BitMixingWeightNoiseConfig,
    BitFlipSTEWeightNoiseConfig,
    SynaptogenPoolNoiseConfig,
    AsymmetricBitNoiseConfig,
    StudentTWeightNoiseConfig,
]


@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config_mod(VGG4LayerActFrontendV1Config):
    activation: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]
    activation_str: str = ""

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation_str = d.pop("activation_str")
        if activation_str == "ReLU":
            from torch.nn import ReLU

            activation = ReLU()
        else:
            assert False, "Unsupported activation %s" % d["activation_str"]
        d["activation"] = activation
        return VGG4LayerActFrontendV1Config(**d)


@dataclass
class ConformerPositionwiseFeedForwardQuantV4Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    weight_bit_prec: Union[int, float]
    activation_bit_prec: Union[int, float]
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    moving_average: Optional[float]  # Moving average for input quantization
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    weight_noise: Optional[WeightNoiseConfig]
    correction_settings: Optional[CycleCorrectionSettings]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu

@dataclass
class ConformerPosEmbConfig(ModelConfiguration):
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float

@dataclass
class QuantizedConformerMHSARelPosV1Config(ModelConfiguration):

    input_dim: int
    num_att_heads: int
    with_bias: bool
    att_weights_dropout: float
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    dot_quant_dtype: torch.dtype
    dot_quant_method: str
    Av_quant_dtype: torch.dtype
    Av_quant_method: str
    bit_prec_W_i: Union[int, float]
    bit_prec_W_o: Union[int, float]
    bit_prec_learn_emb: Union[int, float]
    activation_bit_prec: Union[int, float]
    moving_average: Optional[float]  # Moving average for input quantization
    dropout: float
    quant_in_linear: bool
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    correction_settings: Optional[CycleCorrectionSettings]
    weight_noise: Optional[WeightNoiseConfig]
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"
        assert self.dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"


@dataclass
class ConformerConvolutionQuantV4Config(ModelConfiguration):
    """
    Attributes:
        channels: number of channels for conv layers
        kernel_size: kernel size of conv layers
        dropout: dropout probability
        activation: activation function applied after normalization
        norm: normalization layer with input of shape [N,C,T]
    """

    channels: int
    kernel_size: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    weight_bit_prec: Union[int, float]
    activation_bit_prec: Union[int, float]
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    moving_average: Optional[float]  # Moving average for input quantization
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    correction_settings: Optional[CycleCorrectionSettings]
    weight_noise: Optional[WeightNoiseConfig]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


@dataclass
class ConformerBlockQuantV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1 (first ff, or both if ff2_cfg is None)
        ff2_cfg: Optional separate config for the second ff module; if None, ff_cfg is reused
        mhsa_cfg: Configuration for ConformerMHSAV1
        conv_cfg: Configuration for ConformerConvolutionV1
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardQuantV4Config
    mhsa_cfg: QuantizedConformerMHSARelPosV1Config
    conv_cfg: ConformerConvolutionQuantV4Config
    ff2_cfg: Optional[ConformerPositionwiseFeedForwardQuantV4Config] = None
    modules: List[str] = field(default_factory=lambda: ["ff", "mhsa", "conv", "ff"])
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])


@dataclass
class ConformerEncoderQuantV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: Union[ConformerBlockQuantV1Config, List[ConformerBlockQuantV1Config]]


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return SpecaugConfig(**d)


@dataclass
class QuantModelTrainConfigV15:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    pos_emb_config: ConformerPosEmbConfig
    specauc_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    conv_kernel_size: int
    final_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    weight_quant_dtype: Union[torch.dtype, str]
    weight_quant_method: str
    activation_quant_dtype: Union[torch.dtype, str]
    activation_quant_method: str
    dot_quant_dtype: Union[torch.dtype, str]
    dot_quant_method: str
    Av_quant_dtype: Union[torch.dtype, str]
    Av_quant_method: str
    moving_average: Optional[float]  # default if enabled should be 0.01, if set enables moving average
    weight_bit_prec: Union[int, float, List[Union[int, float, Dict[str, Union[int, float]]]]]
    activation_bit_prec: Union[int, float]
    quantize_output: bool
    quant_in_linear: bool
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    correction_settings: Optional[CycleCorrectionSettings]
    weight_noise: Optional[WeightNoiseConfig]
    module_list: List[str]
    module_scales: List[float]
    aux_ctc_loss_layers: Optional[List[int]]
    aux_ctc_loss_scales: Optional[List[float]]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["converter_hardware_settings"] = DacAdcHardwareSettings(**d["converter_hardware_settings"]) if d["converter_hardware_settings"] is not None else None
        d["pos_enc_converter_hardware_settings"] = DacAdcHardwareSettings(**d["pos_enc_converter_hardware_settings"]) if d["pos_enc_converter_hardware_settings"] is not None else None
        d["correction_settings"] = CycleCorrectionSettings(**d["correction_settings"]) if d["correction_settings"] is not None else None
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        if d.get("weight_pruning") is not None:
            pruning_d = d["weight_pruning"]
            if "threshold" in pruning_d:
                d["weight_pruning"] = ThresholdPruningConfig(
                    start_epoch=pruning_d["start_epoch"],
                    threshold=pruning_d["threshold"],
                )
            elif "percentile" in pruning_d:
                d["weight_pruning"] = PercentilePruningConfig(
                    start_epoch=pruning_d["start_epoch"],
                    percentile=pruning_d["percentile"],
                )
            else:
                raise NotImplementedError(f"Cannot determine pruning type from keys: {list(pruning_d.keys())}")
        else:
            d["weight_pruning"] = None

        for name in ["weight_quant_dtype", "activation_quant_dtype", "dot_quant_dtype", "Av_quant_dtype"]:
            if d[name] == "qint8":
                weight_dtype = torch.qint8
            elif d[name] == "quint8":
                weight_dtype = torch.quint8
            else:
                raise NotImplementedError
            d[name] = weight_dtype
        if d.get("weight_noise") is not None:
            noise_d = d["weight_noise"]
            if "weight_dev" in noise_d:
                d["weight_noise"] = GaussianWeightLevelNoiseConfig(
                    weight_dev=noise_d["weight_dev"], start_epoch=noise_d["start_epoch"]
                )
            elif "bit_amplitude" in noise_d:
                d["weight_noise"] = UniformBitNoiseConfig(
                    bit_amplitude=noise_d["bit_amplitude"], start_epoch=noise_d["start_epoch"]
                )
            elif "weight_amplitude" in noise_d:
                d["weight_noise"] = UniformWeightLevelNoiseConfig(
                    weight_amplitude=noise_d["weight_amplitude"], start_epoch=noise_d["start_epoch"]
                )
            elif "rel_dev" in noise_d:
                d["weight_noise"] = RelativeGaussianWeightNoiseConfig(
                    rel_dev=noise_d["rel_dev"], start_epoch=noise_d["start_epoch"]
                )
            elif "mix" in noise_d:
                d["weight_noise"] = BitMixingWeightNoiseConfig(
                    mix=noise_d["mix"], start_epoch=noise_d["start_epoch"]
                )
            elif "flip_p" in noise_d:
                d["weight_noise"] = BitFlipSTEWeightNoiseConfig(
                    flip_p=noise_d["flip_p"], start_epoch=noise_d["start_epoch"]
                )
            elif "nu" in noise_d:
                # must precede the "dev" branch: this config also has a `dev` key
                d["weight_noise"] = StudentTWeightNoiseConfig(
                    dev=noise_d["dev"], nu=noise_d["nu"], start_epoch=noise_d["start_epoch"]
                )
            elif "on_scale" in noise_d:
                # must precede the "dev" branch: this config also has a `dev` key
                d["weight_noise"] = AsymmetricBitNoiseConfig(
                    dev=noise_d["dev"],
                    on_scale=noise_d["on_scale"],
                    off_scale=noise_d["off_scale"],
                    normalize=noise_d["normalize"],
                    start_epoch=noise_d["start_epoch"],
                )
            elif "pool_size" in noise_d:
                d["weight_noise"] = SynaptogenPoolNoiseConfig(
                    pool_size=noise_d["pool_size"],
                    num_cycles=noise_d["num_cycles"],
                    read_noise=noise_d["read_noise"],
                    strength=noise_d["strength"],
                    parametric=noise_d["parametric"],
                    pool_seed=noise_d["pool_seed"],
                    refresh_per_epoch=noise_d["refresh_per_epoch"],
                    start_epoch=noise_d["start_epoch"],
                )
            elif "dev" in noise_d:
                d["weight_noise"] = GaussianWeightNoiseConfig(dev=noise_d["dev"], start_epoch=noise_d["start_epoch"])
            elif "p" in noise_d:
                d["weight_noise"] = BitFlipWeightNoiseConfig(p=noise_d["p"], start_epoch=noise_d["start_epoch"])
            else:
                raise NotImplementedError(f"Cannot determine noise type from keys: {list(noise_d.keys())}")
        else:
            d["weight_noise"] = None
        return QuantModelTrainConfigV15(**d)

    def __post_init__(self):
        if isinstance(self.weight_bit_prec, list):
            assert len(self.weight_bit_prec) == self.num_layers, (
                f"weight_bit_prec list length {len(self.weight_bit_prec)} must match num_layers {self.num_layers}"
            )
            _valid_keys = {"ff", "ff1", "ff2", "mhsa", "conv"}
            for entry in self.weight_bit_prec:
                if isinstance(entry, dict):
                    assert set(entry.keys()) <= _valid_keys, (
                        f"weight_bit_prec dict keys must be a subset of {_valid_keys}, got {set(entry.keys())}"
                    )
                    assert "ff" in entry or ("ff1" in entry and "ff2" in entry), (
                        "weight_bit_prec dict must contain 'ff' or both 'ff1' and 'ff2'"
                    )
                    assert "mhsa" in entry, "weight_bit_prec dict must contain 'mhsa'"
                    assert "conv" in entry, "weight_bit_prec dict must contain 'conv'"
        for param in [self.weight_quant_dtype, self.activation_quant_dtype, self.dot_quant_dtype, self.Av_quant_dtype]:
            if param == "qint8":
                param = torch.qint8
            elif param == "quint8":
                param = torch.quint8
            elif any(param == x for x in [torch.quint8, torch.qint8]):
                continue
            else:
                raise NotImplementedError
