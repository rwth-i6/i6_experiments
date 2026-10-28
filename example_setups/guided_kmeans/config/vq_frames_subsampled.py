"""Frame-level VQ training on subsampled frames.

The runs of :mod:`.vq_frames_cheat_init` and :mod:`.vq_frames_uniform_init`, with
the silence-free frames pooled over time by a factor k in training, the initial
table, decoding and the FER. Everything else is unchanged.

Setting from the supervised probe (:mod:`.vq_frames_pooling_probe`): average pooling,
with overlapping (2k frames) and non-overlapping (k frames) windows. Supervised PER
against 13.3% at full rate:

* k=2: 15.6% (non-overlapping) / 16.0% (overlapping), best loop probability 0.25
* k=4: 35.9% (non-overlapping) / 41.9% (overlapping), best loop probability 0.05

The loop probabilities per factor are that optimum and one value on each side.
"""

from i6_experiments.example_setups.guided_kmeans.config.vq_frames_cheat_init import (
    cheat_init_table,
    run_frames,
)

FACTORS = [2, 4]
POOLING = "avg_time_np"
LOOP_PROBS = {
    2: [0.1, 0.25, 0.4],
    4: [0.01, 0.05, 0.1],
}

#: Runs continued past the default 50 epochs: (factor, window frames, init name) ->
#: {loop probability: epochs}. k=4 with non-overlapping windows from uniform init was
#: still improving at epoch 50, and loop 0.1 still at epoch 100.
EXTENDED_RUNS = {
    (4, 4, "uniform-init"): {0.01: 100, 0.05: 100, 0.1: 200},
}


def decode_epochs_for(num_epochs):
    """The usual decode epochs, every 10 epochs up to 100, then every 20."""
    return [0, 1, 2, 5] + list(range(10, min(num_epochs, 100) + 1, 10)) + list(range(120, num_epochs + 1, 20))

INITS = [
    # (init name, report description, make_table, sigma, seed)
    ("cheat-init", "a supervised table counted from the pooled frames", cheat_init_table, None, None),
    ("uniform-init", "an exactly uniform table (sigma=0)", lambda ls100, exp_dir, pooling_kwargs: None, 0.0, 42),
]


def run():
    for factor in FACTORS:
        # None: overlapping windows of 2 * factor frames (pipeline default); factor: non-overlapping
        for window in [None, factor]:
            window_frames = window if window is not None else 2 * factor
            for init_name, init_description, make_table, sigma, seed in INITS:
                extended = EXTENDED_RUNS.get((factor, window_frames, init_name))
                run_frames(
                    exp_dir=f"vq_frames_sub-{factor}_win-{window_frames}_{init_name.replace('-', '_')}",
                    init_name=init_name,
                    init_description=f"{init_description}, frames pooled by {factor} with windows of {window_frames}",
                    make_table=make_table,
                    sigma=sigma,
                    seed=seed,
                    loop_probs=LOOP_PROBS[factor],
                    subsampling=factor,
                    pooling_function=POOLING,
                    pooling_kernel_size=window,
                    num_epochs=extended,
                    decode_epochs=(
                        {p: decode_epochs_for(n) for p, n in extended.items()} if extended else None
                    ),
                )


def py():
    run()
