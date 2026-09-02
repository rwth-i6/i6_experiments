"""Two-stage codebook experiment: unguided k-means, then weights over a frozen codebook.

The question this run asks is whether an *unsupervised* partition of the encoder
space carries enough phonetic structure that a label can be defined purely as a
distribution over its cells - and it asks it without any alignment or
transcription entering the loop at any point.

Stage 1 - ``_find_codebook``
    Plain k-means at K in {128, 256, 512}, via
    :func:`...lib.guided_kmeans.chunked.unguided_flavor`. No lexicon, no
    language model, no search: every frame takes its nearest cluster and the
    means follow. Scoring is Mahalanobis under the *corpus* covariance held
    fixed, i.e. k-means in the globally whitened space - see ``global_cov``
    below for why the whitening is not optional and why the covariance is not
    re-estimated here.

Stage 2 is not a stage
    The covariance the codebook was found under is the covariance the mixture
    scores with, carried straight across from stage 1's output. Estimating a
    *per-cluster* full covariance is deliberately not done here, because there
    is nowhere near the evidence for one - see :mod:`.codebook_covariance`,
    which does it anyway and reports what happens.

    Note what the corpus actually supplies, since it is easy to get wrong by a
    factor of four: ``ls-100-segmented`` holds **3,659,991 vectors**, one per
    pooled phoneme segment, not one per 20 ms frame. (The unpooled
    ``ls-100`` file has 15,160,794, i.e. ~4.1 frames per segment; switching
    ``input_data_key`` changes every count below with it.) At D=512 a full
    covariance has 131,328 free parameters, so K=512 gives ~7.1k vectors per
    cluster - one vector per 18 parameters.

Stage 3 - the loop below
    A shared codebook (:class:`...chunked.models.GaussianMixtureModel`) whose
    densities never move: ``update_densities=False``, so the only parameters
    an epoch re-estimates are ``p(density | label)``. The guiding RASR search
    runs as usual, so the label posteriors change every epoch and the weights
    keep moving - what is frozen is the codebook they are defined over.

    The search is forward-backward (``use_forward_backward = True``), so what
    reaches the mixture E-step is a dense ``[T, L]`` posterior rather than a
    single path. That is the right object for a from-scratch run: with no
    reference to anchor one alignment, committing to the best path throws away
    the very uncertainty the weights should be estimated under, and every label
    a frame plausibly belongs to contributes to the counts in proportion to how
    plausible it is. The E-step takes either form unchanged - a Viterbi
    alignment is just the one-hot case - so the only things the flag moves are
    the recognizer class, the RASR binary and which statistics get recorded.

**The initialization is the experiment's weak point, by construction.** With a
shared codebook and identical weights every label scores identically, the first
search has nothing acoustic to separate labels by, and the run converges to
whatever the language model prefers - a fixed point the weights cannot climb
out of once the densities are frozen, because they are all that is left to
move. Symmetry therefore has to be broken in the weights, and with no reference
available the only honest way to break it is at random. Hence the sweep over
Dirichlet concentration: ``RandomMixturesJob``'s ``concentration`` decides how
hard each label commits to a few densities from the outset, and at K=512 the
default of 1.0 draws rows that all sit within O(1/K) of uniform - a very weak
break. Measured on the initializations themselves, as a fraction of the
attainable I(L;C):

    alpha = 1.0  ->  11%      alpha = 0.5  ->  19%      alpha = 0.1  ->  46%

so this is expected to matter more than the seed does. Both are swept.

**What convergence is read from.** No reference exists for these runs, so PER
is not available as a training signal and is not used as one. Two things are
watched instead, both free and both reference-free:

* ``mean_log_likelihood_per_frame`` and ``mean_posterior_entropy`` in each
  epoch's statistics. The first is the actual EM objective, so it is monotone
  under a correct update and its flattening is what convergence means here. The
  second is the collapse detector: if the search is not separating labels
  acoustically the per-frame label posteriors stay flat, and the entropy sits
  near ``log 40 = 3.69`` nats however well the likelihood is doing.

  (A Viterbi run reports the ``average_am_score`` / ``average_lm_score`` split
  instead, where the collapse shows up as a total score improving only through
  the LM term. Forward-backward reports neither - see
  :class:`...lib.guided_kmeans.statistics.FBStatisticsCounter`.)
* ``label_codeword_mi`` from
  :class:`...setup.statistics_jobs.MixtureDiagnosticsJob`, the mutual
  information between label and density under the model's own weights. It is
  exactly 0 for the degenerate solution, and its value at initialization is the
  baseline each run has to climb away from.

Decoding is therefore reserved for a few epochs rather than run on all of them:
the per-epoch statistics answer "is this converging", cheaply, and PER only has
to answer "to what" at the end. ``decode_epochs`` is the knob.
"""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    DuplicateCovsJob,
    GlobalCovarianceJob,
    RandomCentroidsJob,
    RandomMixturesJob,
    chunked_clustering,
    mixture_flavor,
    unguided_flavor,
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

exp_dir = "codebook_mixture"
version = 1


def _find_codebook(
    *, features, num_densities, global_cov, subsampling, num_epochs, num_chunks, seed
):
    """Stage 1: unguided k-means, returning the run that produced the codebook.

    Cheap relative to every other run in this setup, and for one reason: there
    is no RASR search, which is ~99.6% of a guided epoch's wall time. What is
    left is reading features and computing scores, so the epoch is I/O-bound
    and ``num_chunks`` parallelizes reads rather than searches.

    Note the arguments *not* passed: no ``recognition_config``, no ``lexicon``,
    no ``score_reference``. An unguided run has no label inventory, so there is
    nothing for its hypotheses to be scored against - the epoch job records no
    traceback statistics for it and ``out_hypotheses`` comes out empty. Its
    convergence is read from centroid movement, not from PER.
    """
    return chunked_clustering(
        num_epochs=num_epochs,
        features_hdf=features,
        num_clusters=num_densities,
        flavor=unguided_flavor(
            centroids=RandomCentroidsJob(features, num_densities, seed=seed).out_centroids,
            # Held fixed for the whole run. Without it the partition is squared
            # Euclidean, which at these feature dimensions splits along whichever
            # axes the encoder happened to give the largest scale - the partition
            # would then describe that scaling as much as it describes the data.
            covs=DuplicateCovsJob(global_cov, num_densities).out_covs,
            num_clusters=num_densities,
        ),
        subsampling=subsampling,
        num_chunks=num_chunks,
        alias_prefix=f"guided_kmeans/{exp_dir}/codebook-{num_densities}_seed-{seed}",
    )


def run():
    use_eow_phonemes = False
    num_clusters = 40 if not use_eow_phonemes else 79  # labels, i.e. score width
    input_data_key = "ls-100-segmented"

    codebook_sizes = [128, 256, 512]
    # Batch Lloyd's on ~18M frames; the objective flattens well before this, so
    # 10 is a ceiling rather than a target. Watch the centroid movement in the
    # epoch statistics and cut it if it has stopped moving.
    codebook_epochs = 10
    weight_epochs = 10

    use_forward_backward = True
    num_chunks = 20
    num_workers = 8
    lm_order = 3
    subsampling = None
    codebook_seed = 42

    # The two axes of the initialization sweep. Concentration is expected to
    # dominate (see the module docstring); the extra seeds exist to say by how
    # much, and are run at one setting rather than crossed with everything.
    concentrations = [0.1, 0.5, 1.0]
    weight_seeds = [42]
    seed_spread_at = (256, 0.1)
    seed_spread_seeds = [43, 44]

    lm_scale = [30.0]
    lm_spread_at = (*seed_spread_at, 42)
    lm_spread_scales = [1.0]
    loop_prob = 0.0
    distance_scale = 1.0

    decode_lm_scale = 40.0
    # Training is pre-segmented and so runs at 0.0; the cv set has ordinary
    # segmentation, where a label spans several frames and needs a self-loop.
    decode_loop_prob = 0.4
    decode_distance_scale = 1.0

    train_beam_size = 100_000
    decode_beam_size = 100_000

    # Epoch 0 is the initialization, whose PER is the floor every later epoch
    # has to beat; the last is the result. The middle one is there to catch a
    # run that peaks and then degrades, which the statistics alone would not
    # distinguish from one still improving.
    decode_epochs = sorted({0, weight_epochs // 2, weight_epochs})

    features = input_data[input_data_key]["features"]
    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)
    # Computed from the features this run uses, with the same segments and
    # subsampling, so it describes the frames the model actually sees.
    global_cov = GlobalCovarianceJob(features, subsampling=subsampling).out_cov

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv"]["features"],
        sampling_method=SegmentFile(input_data["cv"]["segment_file"]),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "codebook", "alpha", "seed", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            # The forward-backward group. A Viterbi run would want
            # am_score/transition_score/lm_score here instead; those keys do not
            # exist on this path, and asking for them renders blank rather than
            # raising - see latex_report's module docstring.
            "log_likelihood", "posterior_entropy", "dead_clusters",
            # Named via the lexicon above; "silence" is the share of mass on
            # [SILENCE] and "l1" the distance to the phoneme unigram prior.
            "silence", "l1",
        ],
        sort_by=["codebook", "alpha", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Mixture weights over a frozen unguided codebook: {codebook_epochs} epochs "
            f"of k-means at K in {codebook_sizes} under the corpus covariance, then "
            f"{weight_epochs} epochs training p(density|label) alone. Random Dirichlet "
            f"initialization at concentration alpha; no alignment or transcription enters "
            f"the loop. Training loop probability {loop_prob} at AM scale {distance_scale}, "
            f"LM scale {lm_scale}; decoded at LM scale {decode_lm_scale}, loop "
            f"{decode_loop_prob}. 'mi' is I(L;C) in nats under the model's own weights, "
            f"reference-free; 0 is the degenerate solution in which every label weights "
            f"the codebook identically. Statistics columns describe the guiding pass that "
            f"ran with that epoch's model, so the final epoch has none."
        ),
    )
    recog_results = []

    # One codebook per size, shared by every weight run over it: the stage-1
    # jobs do not depend on the initialization sweep, so sisyphus computes each
    # codebook once no matter how many weight runs consume it.
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
    for size, codebook in codebooks.items():
        tk.register_output(
            f"guided_kmeans/{exp_dir}/codebooks/k-{size}_centroids.npy",
            codebook.out_centroids[codebook_epochs],
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/codebooks/k-{size}_statistics.json",
            codebook.out_statistics,
        )

    settings = [
        (size, alpha, seed, lm)
        for (size, alpha), seed, lm in product(product(codebook_sizes, concentrations), weight_seeds, lm_scale)
    ] + [(*seed_spread_at, seed, 30.0) for seed in seed_spread_seeds] \
    + [(*lm_spread_at, lm) for lm in lm_spread_scales]

    for size, alpha, seed, lm_scale in settings:
        codebook = codebooks[size]
        _beam = f"{train_beam_size // 1000}k" if train_beam_size else "inf"
        exp_name = (
            f"k-{size}_alpha-{alpha}_seed-{seed}_lm-{lm_order}-{lm_scale}"
            f"_loop-{loop_prob}_am-{distance_scale}_beam-{_beam}"
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
            # The codebook stage 1 converged on, and the covariance it was found
            # under - taken from the run's own output rather than rebuilt from
            # DuplicateCovsJob so the two cannot drift apart if stage 1's
            # covariance handling ever changes.
            centroids=codebook.out_centroids[codebook_epochs],
            covs=codebook.out_covs[codebook_epochs],
            mixtures=RandomMixturesJob(
                num_clusters, size, concentration=alpha, seed=seed
            ).out_mixtures,
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            # The point of the whole config: the partition was decided in stage 1
            # and stays decided. Also drops the only O(D^2) statistic in the
            # pipeline, which is what makes K=512 affordable - the per-chunk
            # state is the [40, 512] weight statistic rather than ~1 GB of
            # second moments.
            update_densities=False,
            # Mandatory here rather than optional. With the densities frozen the
            # weights are the only parameters left, so a density a label loses
            # cannot be compensated for by the others moving; at the default of
            # 0.0 a zero weight is absorbing and the effective codebook can only
            # ever shrink. Watch `used_densities` in the diagnostics.
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

        # Every epoch, epoch 0 included: these are mini-tasks over a [40, K]
        # array, and the trajectory is the point - epoch 0 is the baseline the
        # rest has to climb away from, and a run whose I(L;C) is flat has
        # collapsed onto the language model whatever its PER says.
        diagnostics = {
            epoch: MixtureDiagnosticsJob(exp_result.out_artifacts["mixtures"][epoch])
            for epoch in range(0, weight_epochs + 1)
        }
        for epoch, job in diagnostics.items():
            tk.register_output(
                f"guided_kmeans/{exp_dir}/mixture_diagnostics/{exp_name}_ep-{epoch}.json",
                job.out_diagnostics,
            )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/mixtures/{exp_name}_final.npy",
            exp_result.out_artifacts["mixtures"][weight_epochs],
        )

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
                params={"codebook": size, "alpha": alpha, "seed": seed},
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
