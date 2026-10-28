"""Frame-level (unsegmented) VQ training from an exact uniform initialization.

Identical to :mod:`.vq_frames_cheat_init` - same features, loop probabilities,
scales, epochs, search build, sentence-end score and decoding - except that the
initial table is exactly uniform, p(codeword | label) = 1/512 (a sigma=0 draw),
instead of a supervised table. The frame-level counterpart of the sigma=0.0
run in :mod:`.vq_unsupervised_long`.
"""

from i6_experiments.example_setups.guided_kmeans.config.vq_frames_cheat_init import run_frames

exp_dir = "vq_frames_uniform_init"


def run():
    run_frames(
        exp_dir=exp_dir,
        init_name="uniform-init",
        init_description="an exactly uniform table (sigma=0)",
        make_table=lambda ls100, exp_dir, pooling_kwargs: None,
        sigma=0.0,
        seed=42,
    )


def py():
    run()
