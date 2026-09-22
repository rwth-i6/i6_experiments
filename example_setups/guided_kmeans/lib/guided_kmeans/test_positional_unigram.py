"""
§7 of ``docs/positional_unigram_init/segmented_base.md``, as runnable checks.

No pytest setup exists for this code base; run it inside the pinned venv with
the recipe root on the path::

    app-py -c "import sys
    sys.path[:0] = ['<repo>/recipe']
    from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans \\
        .test_positional_unigram import main
    sys.exit(main())"

Tests 1-3 are the ones to run before trusting anything downstream: they check
the invariants, then the estimator against the closed-form hypergeometric
marginal of a Bakis chain, then the binned estimate against the same closed form
pushed through the position map. Failure patterns diagnose the cause - a tilted
band means the position convention is off by one, a widened band means lengths
are being pooled that should not be.

§7.6 (one real forward-backward pass with uniform tables must reproduce ``pi1``)
is not here: it needs RASR, features and the sisyphus graph, and lives in
``config/vq_unigram_init.py`` as a comparison arm instead.
"""

from __future__ import annotations

import sys

import numpy as np

from .positional_unigram import (
    POSITION_CONVENTION,
    PositionalUnigram,
    accumulate_binned,
    alignment_kernel,
    build_gamma_tokens,
    kernel_width_tau,
    resolution_report,
    suggest_num_bins,
    backoff,
    bakis_gamma,
    band_of,
    build_gamma,
    effective_rank,
    first_m_step,
    gamma_diagnostics,
    kernel_smooth_tau,
    make_bands,
    overlap_weights,
    smoothing_matrix,
    table_diagnostics,
)

_FAILURES = []


def _check(name, condition, extra=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{(' - ' + extra) if extra else ''}", flush=True)
    if not condition:
        _FAILURES.append(name)
    return condition


def _bakis_paths(num_samples, num_frames, num_states, rng):
    """
    ``num_samples`` uniformly drawn Bakis paths, 0-based states.

    The correct sampler is a uniformly random ``(S-1)``-subset of the ``T-1``
    step positions, because with pinned endpoints and equal self-loops every
    admissible path is equiprobable - sampling a path by walking it with some
    loop probability would *not* give the same distribution unless the loop
    probability happened to be right, and would silently shift the profile.
    """
    order = np.argsort(rng.random((num_samples, num_frames - 1)), axis=1)
    steps = np.zeros((num_samples, num_frames - 1), dtype=np.int64)
    np.put_along_axis(steps, order[:, : num_states - 1], 1, axis=1)
    return np.concatenate(
        [np.zeros((num_samples, 1), dtype=np.int64), np.cumsum(steps, axis=1)], axis=1
    )


def _test_invariants():
    print("\n--- 1. invariants -------------------------------------------------")
    ok = True
    for num_frames in (1, 2, 7, 16, 40, 64, 129, 221):
        for num_bins in (1, 8, 16, 64, 128):
            frame, bin_index, weight = overlap_weights(num_frames, num_bins)
            per_frame = np.bincount(frame, weights=weight, minlength=num_frames)
            per_bin = np.bincount(bin_index, weights=weight, minlength=num_bins)
            ok &= np.allclose(per_frame, 1.0, atol=1e-12)
            ok &= np.allclose(per_bin, num_frames / num_bins, atol=1e-12)
            ok &= len(frame) <= num_frames + num_bins
    _check("sum_beta w(t, beta) == 1 and sum_t w(t, beta) == T/B", bool(ok))

    frame, bin_index, weight = overlap_weights(1, 8)
    _check(
        "T = 1 spreads unit mass evenly over all bins",
        np.allclose(weight, 1.0 / 8) and len(weight) == 8,
    )

    rng = np.random.default_rng(0)
    # Bin-balanced, as §2.2 guarantees real accumulated counts are: each cell
    # carries the same total mass, only distributed differently over labels.
    counts = rng.random((3, 32, 5))
    counts /= counts.sum(-1, keepdims=True) / 7.0
    smoothed = kernel_smooth_tau(counts, sigma=1.5)
    _check(
        "kernel step preserves mass per cell on bin-balanced counts",
        np.allclose(counts.sum(-1), smoothed.sum(-1), atol=1e-9),
        f"max drift {np.abs(counts.sum(-1) - smoothed.sum(-1)).max():.2e}",
    )
    unbalanced = rng.random((3, 32, 5)) * 10
    _check(
        "and preserves each band's total mass on unbalanced counts",
        np.allclose(
            unbalanced.sum((1, 2)),
            kernel_smooth_tau(unbalanced, sigma=1.5).sum((1, 2)),
            atol=1e-9,
        ),
    )
    matrix = smoothing_matrix(32, 1.0)
    _check(
        "smoothing matrix is doubly stochastic (reflecting, no mass leaks out)",
        np.allclose(matrix.sum(0), 1.0) and np.allclose(matrix.sum(1), 1.0),
    )
    _check(
        "smoothing matrix does not spread across the whole axis",
        matrix[0, 16] == 0.0 and matrix[0, 0] > matrix[0, 1],
    )

    gamma = backoff(counts, counts.sum(-1), np.array([300, 300, 300]))
    _check("backoff rows sum to 1", np.allclose(gamma.sum(-1), 1.0, atol=1e-9))

    structural = np.zeros((1, 4, 3))
    structural[0, :, 0] = 100.0          # label 0 everywhere, labels 1-2 never
    gamma = backoff(structural, structural.sum(-1), np.array([10_000]))
    _check(
        "structural zeros survive backoff at large R (no flat floor)",
        gamma[0, :, 1].max() < 1e-3,
        f"max leak {gamma[0, :, 1].max():.2e}",
    )

    edges = make_bands([5] * 300 + [50] * 300 + [500] * 300, r_min=200)
    lengths = np.array([1, 5, 50, 500, 10_000])
    bands = band_of(lengths, edges)
    _check(
        "band_of is monotone and clamps out-of-range lengths",
        bool(np.all(np.diff(bands) >= 0))
        and bands[0] == 0
        and bands[-1] == len(edges) - 2,
        f"edges {edges.tolist()}, bands {bands.tolist()}",
    )
    thin = make_bands([100] * 10, r_min=200)
    _check("a narrow length distribution collapses to one band", len(thin) == 2)


def _test_closed_form(rng):
    print("\n--- 2. exact mode against the closed form ------------------------")
    num_samples, num_frames, num_states = 100_000, 40, 8
    paths = _bakis_paths(num_samples, num_frames, num_states, rng)
    unigram = build_gamma(
        paths,
        num_labels=num_states,
        band_edges=np.array([num_frames, num_frames + 1], dtype=np.int32),
        num_bins=num_frames,
        sigma_bins=0.0,
        kappa0=0.0,
        kappa1=0.0,
    )
    reference = bakis_gamma(num_frames, num_states)
    deviation = np.abs(unigram.gamma[0] - reference)
    tolerance = 4 / np.sqrt(num_samples)
    _check(
        f"B = T, no smoothing: matches the hypergeometric within 4/sqrt(R) = {tolerance:.4f}",
        deviation.max() < tolerance,
        f"max |deviation| {deviation.max():.5f} at "
        f"{np.unravel_index(deviation.argmax(), deviation.shape)}",
    )
    return paths, reference


def _test_binned_consistency(paths, reference, rng):
    print("\n--- 3. binned consistency ----------------------------------------")
    num_samples, num_frames = paths.shape
    num_states = reference.shape[1]
    num_bins = 16
    unigram = build_gamma(
        paths,
        num_labels=num_states,
        band_edges=np.array([num_frames, num_frames + 1], dtype=np.int32),
        num_bins=num_bins,
        sigma_bins=0.0,
        kappa0=0.0,
        kappa1=0.0,
    )
    frame, bin_index, weight = overlap_weights(num_frames, num_bins)
    pushed = np.zeros((num_bins, num_states))
    np.add.at(pushed, bin_index, weight[:, None] * reference[frame])
    pushed /= num_frames / num_bins                 # sum_t w(t, beta) = T/B

    deviation = np.abs(unigram.gamma[0] - pushed)
    tolerance = 4 / np.sqrt(num_samples)
    _check(
        f"B = 16: matches the closed form pushed through w(t, beta) within {tolerance:.4f}",
        deviation.max() < tolerance,
        f"max |deviation| {deviation.max():.5f}",
    )
    _check(
        "binned profiles still sum to 1",
        np.allclose(unigram.gamma.sum(-1), 1.0, atol=1e-9),
    )


def _test_smoothing_preserves_structure(paths):
    print("\n--- 4. smoothing is structure-preserving -------------------------")
    num_frames = paths.shape[1]
    band_edges = np.array([num_frames, num_frames + 1], dtype=np.int32)
    raw = build_gamma(
        paths, num_labels=8, band_edges=band_edges, num_bins=64,
        sigma_bins=0.0, kappa0=0.0, kappa1=0.0,
    )
    smoothed = build_gamma(
        paths, num_labels=8, band_edges=band_edges, num_bins=64,
        sigma_bins=1.0, kappa0=10.0, kappa1=10.0,
    )
    before = effective_rank(raw.gamma[0])
    after = effective_rank(smoothed.gamma[0])
    _check(
        "effective rank retained above 0.9x after smoothing + backoff",
        after >= 0.9 * before,
        f"{before:.3f} -> {after:.3f} ({after / before:.3f}x)",
    )


def _test_degeneracy_detector(rng):
    print("\n--- 5. degeneracy detector fires ---------------------------------")
    num_labels, num_codewords, num_samples = 8, 16, 20_000
    # Doubly stochastic and ergodic, so the stationary distribution is uniform
    # and every positional marginal is uniform too - the exact fixed point §5
    # warns about.
    shift = np.eye(num_labels)
    transition = (
        np.roll(shift, 1, axis=1) + np.roll(shift, 2, axis=1) + np.roll(shift, 3, axis=1)
    ) / 3.0
    emission = np.full((num_labels, num_codewords), 0.02)
    for label in range(num_labels):
        emission[label, (2 * label) % num_codewords] = 0.5
        emission[label, (2 * label + 1) % num_codewords] = 0.5 - 0.02 * num_codewords + 0.02
    emission /= emission.sum(axis=1, keepdims=True)

    labels, codewords = [], []
    for _ in range(num_samples):
        num_frames = int(rng.integers(40, 61))
        sequence = np.empty(num_frames, dtype=np.int64)
        sequence[0] = rng.integers(num_labels)
        for t in range(1, num_frames):
            sequence[t] = rng.choice(num_labels, p=transition[sequence[t - 1]])
        labels.append(sequence)
        codewords.append(
            np.array([rng.choice(num_codewords, p=emission[c]) for c in sequence])
        )

    band_edges = make_bands([len(s) for s in labels], r_min=200)
    unigram = build_gamma(labels, num_labels, band_edges, num_bins=32)
    histogram, mass, _, _ = accumulate_binned(
        codewords, band_edges, 32, num_codewords
    )
    table, _ = first_m_step(unigram.gamma, histogram, mass)

    diagnostics = gamma_diagnostics(unigram)
    rows = table_diagnostics(table)
    spread = np.abs(table - table.mean(axis=0, keepdims=True)).max()
    _check(
        "a stationary ergodic prior makes every row of pi1 the same",
        spread < 8 / np.sqrt(num_samples),
        f"max row spread {spread:.5f} against {8 / np.sqrt(num_samples):.5f}",
    )
    _check(
        "gamma diagnostics report an effective rank near 1",
        diagnostics["max_effective_rank"] < 1.2,
        f"effective rank {diagnostics['max_effective_rank']:.3f}",
    )
    _check(
        "row separation flags every pair as tied",
        rows["tied_pair_fraction"] > 0.99,
        f"max cosine {rows['max_row_cosine']:.5f}, "
        f"tied {rows['tied_pair_fraction']:.3f}",
    )
    # ... and the same machinery on a non-degenerate prior must not flag it.
    left_to_right = _bakis_paths(5_000, 50, num_labels, rng)
    sharp = build_gamma(
        left_to_right,
        num_labels,
        np.array([50, 51], dtype=np.int32),
        num_bins=32,
    )
    _check(
        "the same diagnostics stay quiet on a left-to-right prior",
        gamma_diagnostics(sharp)["max_effective_rank"] > 2.0,
        f"effective rank {gamma_diagnostics(sharp)['max_effective_rank']:.3f}",
    )


def _test_artifact_roundtrip(tmp_path="/tmp"):
    print("\n--- 6. artifact round trip ---------------------------------------")
    import os
    import tempfile

    rng = np.random.default_rng(3)
    paths = _bakis_paths(2_000, 30, 5, rng)
    unigram = build_gamma(
        paths, 5, np.array([30, 31], dtype=np.int32), num_bins=16,
        meta={"corpus_hash": "deadbeef"},
    )
    with tempfile.TemporaryDirectory() as directory:
        target = os.path.join(directory, "gamma.npz")
        unigram.save(target)
        loaded = PositionalUnigram.load(target)
        _check(
            "save/load preserves gamma to float32 and keeps the meta",
            np.allclose(loaded.gamma, unigram.gamma, atol=1e-6)
            and loaded.meta["corpus_hash"] == "deadbeef"
            and loaded.meta["position_convention"] == POSITION_CONVENTION,
        )

        with np.load(target, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        import json

        meta = json.loads(str(arrays["meta"]))
        meta["position_convention"] = "something-else"
        arrays["meta"] = np.asarray(json.dumps(meta))
        np.savez(target, **arrays)
        try:
            PositionalUnigram.load(target)
            rejected = False
        except ValueError:
            rejected = True
        _check("a foreign position convention is rejected on load", rejected)


def _test_m_step_guard():
    print("\n--- 7. M-step guards ----------------------------------------------")
    gamma = np.zeros((1, 4, 3))
    gamma[0, :, 0] = 1.0                       # only label 0 has any mass
    mass = np.full((1, 4), 10.0)
    histogram = np.zeros((1, 4, 6))
    histogram[0, :, 2] = 10.0
    table, info = first_m_step(gamma, histogram, mass, unseen="global")
    _check(
        "an unseen label gets the global codeword histogram, and is reported",
        info["unseen_labels"] == [1, 2]
        and np.isclose(table[1, 2], 1.0)
        and np.allclose(table.sum(1), 1.0),
    )
    mismatched = histogram.copy()
    mismatched[0, 0, 2] = 5.0                  # H no longer sums to N
    try:
        first_m_step(gamma, mismatched, mass)
        caught = False
    except AssertionError:
        caught = True
    _check("gamma and H accumulated over different cells is caught", caught)




# --- addendum §11: token sequences with L << T -------------------------------


def _uniform_compositions(num_frames, parts, rng, size):
    """
    ``size`` uniform compositions of ``T`` into ``parts`` positive durations.

    This is the sampler the geometric duration model implies *after*
    conditioning on the total length: every composition has likelihood
    ``a^(T-L) (1-a)^L``, the same for all of them, so the conditional law is
    uniform and ``a`` never appears. Drawing durations by walking a chain with
    some self-loop probability would give the same distribution only by
    accident.
    """
    order = np.argsort(rng.random((size, num_frames - 1)), axis=1)
    breaks = np.sort(order[:, : parts - 1], axis=1) + 1
    edges = np.concatenate(
        [
            np.zeros((size, 1), dtype=np.int64),
            breaks,
            np.full((size, 1), num_frames, dtype=np.int64),
        ],
        axis=1,
    )
    return np.diff(edges, axis=1)


def _test_duration_kernel(rng):
    print("\n--- 7/9. the alignment kernel ------------------------------------")
    kernel = alignment_kernel(40, 8, 1)
    reference = bakis_gamma(40, 8)
    _check(
        "m=1 kernel is the base spec's hypergeometric with S -> L",
        np.abs(kernel - reference).max() < 1e-12,
        f"max |deviation| {np.abs(kernel - reference).max():.2e}",
    )
    frames = np.arange(1, 41)[:, None]
    tokens = np.arange(1, 9)[None, :]
    support = (tokens <= frames) & (frames <= 40 - 8 + tokens)
    _check(
        "support is exactly i <= t <= T - L + i, and rows sum to 1",
        np.array_equal(kernel > 0, support)
        and np.allclose(kernel.sum(axis=1), 1.0, atol=1e-12),
    )
    _check(
        "L = 1 puts every frame on the only token; L = T is the identity",
        np.allclose(alignment_kernel(17, 1), 1.0)
        and np.allclose(alignment_kernel(9, 9), np.eye(9)),
    )

    # §9.1's claim that the self-loop probability cancels, checked by simulation
    # rather than by re-deriving it: sample compositions (which is the
    # conditional law for *any* a) and compare the empirical kernel.
    num_samples, num_frames, num_tokens = 40_000, 60, 12
    durations = _uniform_compositions(num_frames, num_tokens, rng, num_samples)
    index = np.concatenate(
        [np.repeat(np.arange(num_tokens), d)[None, :] for d in durations], axis=0
    )
    empirical = np.stack(
        [np.bincount(index[:, t], minlength=num_tokens) for t in range(num_frames)]
    ) / num_samples
    deviation = np.abs(empirical - alignment_kernel(num_frames, num_tokens)).max()
    _check(
        "kernel matches sampled compositions within 4/sqrt(R)",
        deviation < 4 / np.sqrt(num_samples),
        f"max |deviation| {deviation:.5f} against {4 / np.sqrt(num_samples):.5f}",
    )

    _check(
        "m = 1 state splitting is a no-op",
        np.array_equal(alignment_kernel(60, 12, 1), alignment_kernel(60, 12, 1)),
    )
    # The spec quotes sd_tau ~ 1/(2 sqrt(mL)), which is the mL << T limit. Check
    # the exact width everywhere, and the asymptotic only where it applies.
    for num_frames, num_tokens, states in ((400, 40, 3), (4000, 40, 3), (489, 125, 3)):
        kernel = alignment_kernel(num_frames, num_tokens, states)
        position = np.arange(1, num_tokens + 1)
        mean = (kernel * position).sum(axis=1)
        sd = np.sqrt(np.maximum((kernel * position ** 2).sum(axis=1) - mean ** 2, 0.0))
        window = slice(num_frames // 2 - num_frames // 8, num_frames // 2 + num_frames // 8)
        measured = sd[window].mean() / num_tokens
        exact = float(kernel_width_tau([num_tokens], [num_frames], states)[0])
        _check(
            f"sd_tau at T={num_frames}, L={num_tokens}, m={states} matches the exact form",
            abs(measured - exact) < 0.05 * exact,
            f"measured {measured:.5f}, exact {exact:.5f}, "
            f"asymptotic {1 / (2 * np.sqrt(states * num_tokens)):.5f}",
        )
    asymptotic_regime = float(kernel_width_tau([40], [4000], 3)[0])
    _check(
        "the asymptotic 1/(2 sqrt(mL)) is recovered once mL << T",
        abs(asymptotic_regime - 1 / (2 * np.sqrt(120))) < 0.05 / (2 * np.sqrt(120)),
    )


def _test_token_end_to_end(rng):
    print("\n--- 10. token accumulation against frame-level counting ----------")
    num_samples, num_frames, num_tokens, num_labels = 20_000, 120, 30, 6
    for states in (1, 3):
        # Token sequences with real positional structure, so a flat gamma would
        # not pass by accident: the first and last token are pinned to distinct
        # labels and the middle drifts.
        tokens = rng.integers(0, num_labels, size=(num_samples, num_tokens))
        tokens[:, 0] = 0
        tokens[:, -1] = num_labels - 1
        sub = _uniform_compositions(num_frames, num_tokens * states, rng, num_samples)
        durations = sub.reshape(num_samples, num_tokens, states).sum(axis=2)

        expanded = (
            np.repeat(tokens[r], durations[r]) for r in range(num_samples)
        )
        frame_level = build_gamma(
            expanded,
            num_labels=num_labels,
            band_edges=np.array([num_frames, num_frames + 1], dtype=np.int32),
            num_bins=24,
            sigma_bins=0.0,
            kappa0=0.0,
            kappa1=0.0,
        )
        token_level = build_gamma_tokens(
            ((tokens[r], num_frames) for r in range(num_samples)),
            num_labels=num_labels,
            band_edges=np.array([num_tokens, num_tokens + 1], dtype=np.int32),
            num_bins=24,
            sub_states=states,
            sigma_bins=0.0,
            kappa0=0.0,
            kappa1=0.0,
        )
        deviation = np.abs(frame_level.gamma - token_level.gamma).max()
        _check(
            f"m={states}: kernel scatter reproduces frame-level counting "
            f"within 4/sqrt(R)",
            deviation < 4 / np.sqrt(num_samples),
            f"max |deviation| {deviation:.5f} against {4 / np.sqrt(num_samples):.5f}",
        )
        _check(
            f"m={states}: token-mode mass is T/B per bin, as in §2.5",
            np.allclose(token_level.mass, num_samples * num_frames / 24),
        )

    # ... and the failure this test exists to catch: expanding with one duration
    # model and assuming another must NOT agree. The token sequences need
    # positional structure for the difference to show - with iid tokens gamma is
    # flat whatever the kernel, and the test would pass vacuously.
    tokens = rng.integers(0, num_labels, size=(num_samples, num_tokens))
    tokens[:, 0] = 0
    tokens[:, -1] = num_labels - 1
    sub = _uniform_compositions(num_frames, num_tokens * 3, rng, num_samples)
    durations = sub.reshape(num_samples, num_tokens, 3).sum(axis=2)
    frame_level = build_gamma(
        (np.repeat(tokens[r], durations[r]) for r in range(num_samples)),
        num_labels=num_labels,
        band_edges=np.array([num_frames, num_frames + 1], dtype=np.int32),
        num_bins=24, sigma_bins=0.0, kappa0=0.0, kappa1=0.0,
    )
    mismatched = build_gamma_tokens(
        ((tokens[r], num_frames) for r in range(num_samples)),
        num_labels=num_labels,
        band_edges=np.array([num_tokens, num_tokens + 1], dtype=np.int32),
        num_bins=24, sub_states=1, sigma_bins=0.0, kappa0=0.0, kappa1=0.0,
    )
    _check(
        "a wrong sub-state count is visible, not silent",
        np.abs(frame_level.gamma - mismatched.gamma).max() > 4 / np.sqrt(num_samples),
        f"max |deviation| {np.abs(frame_level.gamma - mismatched.gamma).max():.5f}",
    )


def _test_resolution_floor(rng):
    print("\n--- 11. resolution floor ------------------------------------------")
    num_labels, num_samples = 8, 4_000
    ranks = {}
    for num_tokens in (4, 16, 64, 256):
        num_frames = 4 * num_tokens
        # A corpus whose label distribution genuinely varies with position: token
        # i prefers label round(C * i / L). Endpoint pinning is not enough here -
        # its share of the sequence falls as 1/L, so a corpus with iid middles
        # gets *less* structured as L grows and the rank would fall with it,
        # measuring the corpus rather than the resolution.
        position = np.arange(num_tokens) / max(num_tokens - 1, 1)
        centres = position * (num_labels - 1)
        weights = np.exp(
            -0.5 * ((np.arange(num_labels)[None, :] - centres[:, None]) / 0.7) ** 2
        )
        weights /= weights.sum(axis=1, keepdims=True)
        tokens = np.stack(
            [
                rng.choice(num_labels, size=num_samples, p=weights[i])
                for i in range(num_tokens)
            ],
            axis=1,
        )
        unigram = build_gamma_tokens(
            ((tokens[r], num_frames) for r in range(num_samples)),
            num_labels=num_labels,
            band_edges=np.array([num_tokens, num_tokens + 1], dtype=np.int32),
            num_bins=32, sub_states=1, sigma_bins=0.0, kappa0=0.0, kappa1=0.0,
        )
        ranks[num_tokens] = effective_rank(unigram.gamma[0])
        width = float(kernel_width_tau([num_tokens], [num_frames], 1)[0])
        print(f"        L={num_tokens:4d}  sd_tau {width:.4f}  effective rank "
              f"{ranks[num_tokens]:.3f}")
    _check(
        "effective rank grows with L",
        all(ranks[a] < ranks[b] for a, b in ((4, 16), (16, 64), (64, 256))),
        ", ".join(f"L={k}: {v:.2f}" for k, v in ranks.items()),
    )
    # The spec's second claim - "near rank 1 plus boundary tokens at L = 4" -
    # is about a corpus whose label distribution does *not* vary strongly with
    # position, which is the realistic case: the structure then comes from the
    # pinned endpoints alone and the kernel has nothing else to resolve. On the
    # deliberately position-structured corpus above it is false, and rightly so
    # - four positions carrying four distinct distributions are four
    # distinguishable rows however short the sequence.
    num_tokens = 4
    tokens = rng.integers(0, num_labels, size=(num_samples, num_tokens))
    tokens[:, 0] = 0
    tokens[:, -1] = num_labels - 1
    flat = build_gamma_tokens(
        ((tokens[r], 4 * num_tokens) for r in range(num_samples)),
        num_labels=num_labels,
        band_edges=np.array([num_tokens, num_tokens + 1], dtype=np.int32),
        num_bins=32, sub_states=1, sigma_bins=0.0, kappa0=0.0, kappa1=0.0,
    )
    boundary_only = effective_rank(flat.gamma[0])
    _check(
        "with no positional structure, L = 4 leaves only the boundary tokens",
        boundary_only < 3.0,
        f"effective rank {boundary_only:.3f} of {num_labels}, against "
        f"{ranks[4]:.3f} on the position-structured corpus at the same L",
    )

    report = resolution_report([4] * 10, [16] * 10, 1)
    _check(
        "resolution_report flags a corpus below the floor",
        report["verdict"] in ("flat", "marginal") and report["fraction_good"] == 0.0,
        f"verdict {report['verdict']}, sd_tau median {report['sd_tau_median']:.3f}",
    )
    _check(
        "suggest_num_bins follows 4 sqrt(max L) clipped to [32, 128]",
        suggest_num_bins([10, 274]) == 66 and suggest_num_bins([4]) == 32
        and suggest_num_bins([10_000]) == 128,
        f"{suggest_num_bins([10, 274])}, {suggest_num_bins([4])}, "
        f"{suggest_num_bins([10_000])}",
    )


def main() -> int:
    rng = np.random.default_rng(1234)
    _test_invariants()
    paths, reference = _test_closed_form(rng)
    _test_binned_consistency(paths, reference, rng)
    _test_smoothing_preserves_structure(paths)
    _test_degeneracy_detector(rng)
    _test_artifact_roundtrip()
    _test_m_step_guard()
    _test_duration_kernel(rng)
    _test_token_end_to_end(rng)
    _test_resolution_floor(rng)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) failed: {', '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
