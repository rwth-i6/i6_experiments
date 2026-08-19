"""
A diagnostics pass on the *initialization* centroids - the cheating centroids
``chunked_cheated_cov.py`` starts from, before any clustering epoch has touched
them.

Records what the guiding search actually does frame by frame: the recognition
scores of every sequence, and the distance from every frame to the centroid it
was aligned with. Nothing here updates a model or touches a clustering run -
see ``setup/diagnostics.py`` for why this is a job of its own.

Whatever the search does wrong *here* it does with oracle clusters, which makes
this the one pass whose findings cannot be blamed on the clustering having
drifted. It also depends on nothing but input files, so it runs immediately,
regardless of what any clustering run is doing.

No epoch indexing is involved, deliberately. A run's models carry an offset
that is easy to get wrong - ``chunked_clustering`` keys ``out_centroids`` so
that entry ``e`` is the model *produced* by epoch ``e``, while epoch ``e``
*recognized* with ``out_centroids[e - 1]`` - and entry 0 is these very files.
Pointing at them directly removes the question. To inspect a trained model
later, re-derive the run with ``chunked_clustering(...)`` using arguments
identical to ``chunked_cheated_cov.py`` and pass ``out_centroids[e]`` /
``out_covs[e]`` below; identical arguments produce identical job hashes, so
that reuses the epoch jobs instead of recomputing them.

Loading the result::

    from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked \\
        import load_diagnostics
    diag = load_diagnostics("output/guided_kmeans/cov_cheating_seg/diagnostics/init")

    table = diag.sequence_table()           # per-sequence scores, per frame
    diag.frames_of(table["seq_tag"][worst]) # drill into one outlier

See ``Diagnostics.sequence_table`` for what the columns mean - in particular
why the recognition scores come in a ``_last`` and a ``_sum`` form, and why the
``_per_frame`` columns are the ones to histogram.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.diagnostics import clustering_diagnostics
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import (
    get_sampled_segments_file,
)

exp_dir = "cov_cheating_seg"

#: Restrict the pass to the sampled subset the decodes are scored on (~120
#: segments, minutes instead of half an hour). Good for iterating on what to
#: look at; too few sequences for a sequence-level score distribution, so turn
#: it off for the real analysis - the full corpus is ~30 min at num_chunks=30.
USE_SAMPLED_SUBSET = False

input_data = {
    "features": tk.Path(
        "/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/segmented_features_wav2vec2_ls100h.hdf"
    ),
    "cheating_centroids": tk.Path(
        "/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids.npy"
    ),
    "cheating_covs": tk.Path(
        "/u/mann/experiments/2026-06-09--guided-k-means/test/cheating_centroids_larissa/covs.npy"
    ),
}

rasr_path = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")


def run():
    # The search this pass performs has to be the one the run performs,
    # otherwise the dump describes a differently-configured recognition. These
    # therefore track chunked_cheated_cov.py.
    use_eow_phonemes = False
    use_pruning = False
    num_chunks = 30
    num_workers = 8
    lm_order = 3
    subsampling = None
    loop_probability, silence_loop_probability = 0.0, 0.0
    lm_scale = 40.0

    exp_name = (
        f"chunked_lm-{lm_order}-{lm_scale}"
        f"_loop-{loop_probability}-sil-loop-{silence_loop_probability}_ls-100-segmented"
    )

    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    recognition_config = create_recog_rasr_config(
        lm_scale=lm_scale,
        emission_scale=1.0,
        transition_scale=1.0,
        loop_probability=loop_probability,
        silence_loop_probability=silence_loop_probability,
        use_tree_search=False,
        max_beam_size=20000 if use_pruning else None,
        score_threshold=10000.0 if use_pruning else None,
        lm_order=lm_order,
        use_eow_phonemes=use_eow_phonemes,
    )

    diagnostics = clustering_diagnostics(
        features_hdf=input_data["features"],
        recognition_config=recognition_config,
        lexicon=lexicon,
        centroids=input_data["cheating_centroids"],
        covs=input_data["cheating_covs"],
        segments=get_sampled_segments_file(min_phoneme_count=5) if USE_SAMPLED_SUBSET else None,
        subsampling=subsampling,
        distance_scale=1.0,
        rasr_path=rasr_path,
        num_chunks=num_chunks,
        num_workers=num_workers,
        alias=f"guided_kmeans/{exp_dir}/diagnostics/{exp_name}_init",
    )
    tk.register_output(
        f"guided_kmeans/{exp_dir}/diagnostics/init", diagnostics.out_diagnostics
    )


def py():
    run()
