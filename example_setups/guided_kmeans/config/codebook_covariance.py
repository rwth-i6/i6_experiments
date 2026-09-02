"""The covariance arm: what shape to give the densities of a frozen codebook.

Shares stage 1 with :mod:`.codebook_mixture` - the same
:func:`~.codebook_mixture._find_codebook` call with the same arguments, so
sisyphus computes each codebook once and both configs consume it - and varies
only what happens between the partition and the mixture: how a covariance is
estimated for each cell of it.

That is the plan's step 2, deliberately split out of ``codebook_mixture`` so
the two questions do not confound each other. There, the covariance is the
corpus-wide one held fixed and the only variable is the weight
initialization; here the initialization is pinned and the covariance varies.

**The arithmetic that motivates the whole sweep.** A full ``[D, D]`` covariance
has D(D+1)/2 free parameters - 131,328 at D=512. ``ls-100-segmented`` supplies
3,659,991 vectors, one per pooled phoneme segment (the unpooled ``ls-100`` file
has 15,160,794, ~4.1 frames per segment - so these numbers are specific to the
segmented input this config uses):

    ============  ====================  ==============  ==================
    K             vectors per cluster   full: per par.  diagonal: per par.
    ============  ====================  ==============  ==================
    128                        28,594            0.22                55.8
    256                        14,297            0.11                27.9
    512                         7,148           0.054                14.0
    ============  ====================  ==============  ==================

So at every K there is *less than one vector of evidence per free parameter* of
a full covariance, and at K=512 it is one per eighteen. For reference the
existing 40-label full-covariance runs on this input sit at 0.70 - already
below one - so this is not a new problem so much as a further eighteen-fold
step along an axis that is already past its limit.

Note the distinction the diagnostics keep separate, because it is the one that
makes this arm subtle rather than merely broken. *Rank* needs only more vectors
than dimensions: at 7.1k vectors and D=512 every covariance here comes out
invertible, and ``num_singular`` will happily read 0. *Evidence* is the table
above, and it is catastrophic. A full covariance in this regime is therefore
not a matrix that fails loudly - it is one that inverts cleanly and means
nothing, which is exactly why the failure has to be forced into the open
rather than waited for.

The diagonal column is the same table for D free parameters instead of
D(D+1)/2. It stays above the conventional ten-per-parameter threshold at every
K, though only just at K=512 - which is worth knowing before reading too much
into that point.

**Why the failure has to be forced into the open.** It will not announce
itself. ``GaussianModelNumpy`` inverts with ``np.linalg.inv`` and casts to
float32, so a covariance with condition number past ~1e7 yields meaningless
scores rather than an exception, and a genuinely singular one gives a
``-inf`` log-determinant that turns every score into ``inf``. A run that simply
crashes is a weak result. Hence:

* ``ClusterCovarianceJob`` reports condition numbers, log-determinants,
  smallest eigenvalues, how many covariances came out singular, and
  ``frames_per_free_parameter`` - the last being the one that carries the
  argument, since the others can all look healthy while it is 0.054;
* the ``full`` arm is swept along ``ridge`` rather than run once. The point
  the sweep makes is that the ridge needed for numerical stability is large
  enough to swamp the estimate - i.e. what makes the model work is exactly
  what removes the per-cluster information from it. ``shared`` is that limit
  taken exactly, and is the control every other setting has to beat.

``diagonal`` is the arm expected to actually work: diagonal in the space the
corpus covariance whitens, which is semi-tied covariances with the transform
taken rather than estimated. D free parameters per cluster instead of
D(D+1)/2 - a factor of 257 at D=512 - which moves K=512 from 0.054 vectors per
parameter to 14.

Cost: each setting is a full guided run, so the sweep below is
``len(codebook_sizes) * len(covariance_settings)`` of them. Trim
``covariance_settings`` before ``codebook_sizes`` - the ridge points are the
cheapest to give up, since ``shared`` already marks the limit they approach.
"""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    ClusterCovarianceJob,
    GlobalCovarianceJob,
    RandomMixturesJob,
    chunked_clustering,
    mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.statistics_jobs import (
    MixtureDiagnosticsJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    LatexTableReport,
    clustering_statistics_per_epoch,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.codebook_mixture import _find_codebook

exp_dir = "codebook_covariance"
version = 1


def run():
    use_eow_phonemes = False
    num_clusters = 40 if not use_eow_phonemes else 79
    input_data_key = "ls-100-segmented"

    codebook_sizes = [128, 256, 512]
    # Must match codebook_mixture.run() exactly, or the codebooks are different
    # jobs and stage 1 gets computed twice.
    codebook_epochs = 10
    codebook_seed = 42
    weight_epochs = 10

    # (name, structure, ridge). The ridge points exist to show the full arm
    # being bought numerical stability at the price of the information it was
    # supposed to carry; "shared" is where that price is paid in full.
    covariance_settings = [
        ("full", "full", 0.0),
        ("full-ridge1e-2", "full", 1e-2),
        ("diagonal", "diagonal", 0.0),
        ("shared", "shared", 0.0),
    ]

    # Pinned, so the covariance is the only thing that varies. 0.1 because it is
    # the sharpest of the initializations codebook_mixture sweeps and so starts
    # furthest from the degenerate all-labels-alike solution (46% of the
    # attainable I(L;C) against 11% at the default of 1.0).
    concentration = 0.1
    weight_seed = 42

    # Matches codebook_mixture: the two arms are only comparable if they differ
    # in the covariance and nothing else, and a Viterbi arm would additionally
    # differ in what reaches the E-step (one path vs a distribution over them)
    # and in which statistics get recorded.
    use_forward_backward = True
    num_chunks = 20
    num_workers = 8
    lm_order = 3
    subsampling = None

    lm_scale = 30.0
    loop_prob = 0.0
    distance_scale = 1.0

    decode_lm_scale = 40.0
    decode_loop_prob = 0.4
    decode_distance_scale = 1.0

    train_beam_size = 100_000
    decode_beam_size = 100_000

    # Fewer than codebook_mixture decodes: this arm is read across settings at
    # the end rather than along its trajectory, and the per-epoch statistics
    # already say whether a run is going anywhere.
    decode_epochs = sorted({0, weight_epochs})

    features = input_data[input_data_key]["features"]
    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)
    global_cov = GlobalCovarianceJob(features, subsampling=subsampling).out_cov

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv"]["features"],
        sampling_method=SegmentFile(input_data["cv"]["segment_file"]),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "codebook", "covariance", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            # Forward-backward group, as in codebook_mixture.
            "log_likelihood", "posterior_entropy", "dead_clusters",
            # Named via the lexicon above; "silence" is the share of mass on
            # [SILENCE] and "l1" the distance to the phoneme unigram prior.
            "silence", "l1",
        ],
        sort_by=["codebook", "covariance"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Covariance structure over a frozen unguided codebook. Stage 1 and the "
            f"weight initialization are held fixed (Dirichlet alpha={concentration}, "
            f"seed {weight_seed}); only how each cell's covariance is estimated varies. "
            f"'full' is unconstrained and at these cluster counts has fewer frames per "
            f"cluster than a covariance has free parameters; 'diagonal' is diagonal in "
            f"the space the corpus covariance whitens (D parameters rather than "
            f"D(D+1)/2); 'shared' pools one covariance over the corpus and is the control. "
            f"Read alongside the per-setting covariance diagnostics, which carry the "
            f"condition numbers and singular counts behind these numbers. Guided by "
            f"forward-backward search, so the mixture weights are estimated under a "
            f"distribution over paths rather than a single alignment."
        ),
    )
    recog_results = []

    codebooks = {
        size: _find_codebook(
            features=features,
            num_densities=size,
            global_cov=global_cov,
            subsampling=subsampling,
            num_epochs=codebook_epochs,
            num_chunks=num_chunks,
            seed=codebook_seed,
        )
        for size in codebook_sizes
    }

    for size, (cov_name, structure, ridge) in product(codebook_sizes, covariance_settings):
        codebook = codebooks[size]

        cluster_covs = ClusterCovarianceJob(
            features_hdf=features,
            centroids=codebook.out_centroids[codebook_epochs],
            # The covariance stage 1 assigned under, so this job reproduces that
            # run's own partition rather than a near miss - and, for the diagonal
            # structure, supplies the transform. The space the clusters are
            # measured in should be the space they were found in.
            assignment_covs=codebook.out_covs[codebook_epochs],
            structure=structure,
            ridge=ridge,
            subsampling=subsampling,
        )
        cluster_covs.add_alias(f"guided_kmeans/{exp_dir}/covs/k-{size}_{cov_name}")
        tk.register_output(
            f"guided_kmeans/{exp_dir}/covariance_diagnostics/k-{size}_{cov_name}.json",
            cluster_covs.out_diagnostics,
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/covariance_counts/k-{size}_{cov_name}.npy",
            cluster_covs.out_counts,
        )

        _beam = f"{train_beam_size // 1000}k" if train_beam_size else "inf"
        exp_name = (
            f"k-{size}_cov-{cov_name}_alpha-{concentration}_seed-{weight_seed}"
            f"_lm-{lm_order}-{lm_scale}_loop-{loop_prob}_am-{distance_scale}_beam-{_beam}"
        )

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=lm_scale,
            loop_probability=loop_prob,
            silence_loop_probability=loop_prob,
            use_forward_backward_search=use_forward_backward,
            lm_order=lm_order,
            use_eow_phonemes=use_eow_phonemes,
            max_beam_size=train_beam_size,
        )

        flavor = mixture_flavor(
            centroids=codebook.out_centroids[codebook_epochs],
            covs=cluster_covs.out_covs,
            mixtures=RandomMixturesJob(
                num_clusters, size, concentration=concentration, seed=weight_seed
            ).out_mixtures,
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            # As in codebook_mixture: the codebook and its covariances are the
            # experiment's input, and only p(density|label) is learned. Without
            # this the covariances would be re-estimated from epoch 1 and the
            # setting under test would not survive its own first epoch.
            update_densities=False,
            mixture_floor=1e-6,
            num_workers=num_workers,
        )

        exp_result = chunked_clustering(
            num_epochs=weight_epochs,
            features_hdf=features,
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            flavor=flavor,
            subsampling=subsampling,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            rasr_path=(
                tools.RASR_PATH_FORWARD_BACKWARD if use_forward_backward else tools.RASR_PATH
            ),
            num_chunks=num_chunks,
            num_workers=num_workers,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
        )

        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json", exp_result.out_statistics
        )
        statistics = clustering_statistics_per_epoch(
            exp_result.out_epoch_statistics,
            name=exp_name,
            epoch_offset=1,
            # Required on the forward-backward path. Its counter reports
            # soft_cluster_frequencies - masses indexed by cluster, with no names
            # attached, because the counter is built from num_clusters alone.
            # Without the lexicon to name them, EpochStatisticsJob can produce
            # neither the phoneme distribution nor the prior distance.
            lexicon=lexicon,
        )
        diagnostics = {
            epoch: MixtureDiagnosticsJob(exp_result.out_artifacts["mixtures"][epoch])
            for epoch in decode_epochs
        }

        recognition_config_decode = create_recog_rasr_config(
            lm_scale=decode_lm_scale,
            emission_scale=1.0,
            transition_scale=None,
            loop_probability=decode_loop_prob,
            silence_loop_probability=decode_loop_prob,
            lm_order=lm_order,
            use_eow_phonemes=use_eow_phonemes,
            max_beam_size=decode_beam_size,
        )
        decode_name = exp_name + f"_dec-{decode_lm_scale}_dloop-{decode_loop_prob}"

        for recog_epoch in decode_epochs:
            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                model_dir=exp_result.out_models[recog_epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=decode_distance_scale,
                subsampling=subsampling,
                write_frame_labels=True,
            )
            res = decode_and_score(
                decode_name + f"_ep-{recog_epoch}",
                "cv",
                decode_config,
                cv_dataset_config,
                rasr_path=tools.RASR_PATH,
                device="cpu",
                corpus_key="train-other-960",
            )
            if res.frame_labels is not None:
                res.fer = FrameErrorRateJob(res.frame_labels, GMM_ALIGNMENT_CV, lexicon).out_fer
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_fer", res.fer
                )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/per/{decode_name}_ep-{recog_epoch}_per", res.per
            )
            recog_results.append(res)
            latex_report.add_row(
                result=res,
                params={"codebook": size, "covariance": cov_name},
                epoch=recog_epoch,
                statistics=statistics,
                values={
                    k: v
                    for k, v in (
                        ("mi", diagnostics[recog_epoch].out_mi),
                        ("fer", res.fer),
                    )
                    if v is not None
                },
            )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
