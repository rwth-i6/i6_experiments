"""Chunked guided k-means: Gaussian *mixture* models, pre-segmented features, random init.

Showcases the two mixture layouts side by side, both starting from random
frames and the corpus covariance so the only difference is how densities are
shared:

``shared-128``
    One codebook of 128 densities that every label draws on, weighted per label
    (:class:`GaussianMixtureModel`). Densities are re-estimated from every
    label that uses them, so 128 of them are supported by the whole corpus.

``per-label-3``
    3 densities owned by each of the 40 labels, 120 in total
    (:class:`PerLabelMixtureModel`). Nothing is shared, so each density is
    estimated from one label's frames only - but a label's densities are free
    to specialize without pulling on any other label's.

Roughly matched in parameter count on purpose; what differs is where the
statistical strength comes from. Both reduce to the single-Gaussian setup in
``fb_cov.py`` at one density per label, which is the run to compare against.
``viterbi_mixture_cheat_init.py`` is the same per-label setup started from the
cheating centroids instead of random frames.

Viterbi search over pre-segmented features, so the training loop probability is
0.0 - a segment is one label. The decode is the ordinary unsegmented cv set at
0.4, which is what the rest of the setup reports and therefore what these
numbers can be compared against. Forward-backward is one flag away:
``use_forward_backward`` below drives the RASR config, the flavor and the
pipeline together, and picks the RASR binary with it.

Two mechanics worth copying out of here:

* the model is chosen by passing a *flavor* rather than by setting flags, which
  is how a run says which model, search and updating routine belong together;
* decoding takes ``exp_result.out_models[epoch]`` - the whole model directory -
  rather than individual arrays, so the decode side needs to know nothing about
  mixtures. ``DecodeConfig(centroids=..., covs=...)`` still works and is what
  the single-Gaussian configs use.

Not enabled here: per-epoch PER on the *training* corpus, via
``chunked_clustering(..., score_reference=TaggedCorpusToTxtJob(phoneme_corpus).out_txt)``.
It is a cheap convergence signal, but it needs the training reference to cover
exactly the segments in the feature file - check that before switching it on.
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
    RepeatCovsJob,
    SplitCentroidsJob,
    UniformMixturesJob,
    chunked_clustering,
    mixture_flavor,
    per_label_mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, SegmentFile
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    LatexTableReport,
    clustering_statistics_per_epoch,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob

exp_dir = "viterbi_mixture"
version = 1


def _shared_codebook(features, num_labels, num_densities, global_cov, seed):
    """Initial artifacts for a codebook of ``num_densities`` shared densities.

    Random frames as density means, the corpus-wide covariance duplicated across
    every density, and *random* weights per label.

    Random weights rather than uniform, which looks like the neutral choice and
    is not: with one shared codebook and identical weights, every label produces
    an identical score column, and the first recognition pass has nothing
    acoustic to distinguish labels by. Symmetry has to be broken somewhere, and
    the weights are the only place it can be for this layout.

    The shared covariance rather than the identity so the first epoch already
    scores in a whitened space - the partition then reflects the shape of the
    data instead of the encoder's arbitrary per-dimension scaling.
    ``IdentityCovsJob(num_densities, feature_dim=512)`` is the substitute if no
    covariance is wanted.
    """
    return dict(
        centroids=RandomCentroidsJob(features, num_densities, seed=seed).out_centroids,
        covs=DuplicateCovsJob(global_cov, num_densities).out_covs,
        mixtures=RandomMixturesJob(num_labels, num_densities, seed=seed).out_mixtures,
    )


def _per_label(features, num_labels, densities_per_label, global_cov, seed):
    """Initial artifacts for ``densities_per_label`` densities owned by each label.

    Uniform weights are genuinely neutral here - the densities already differ
    per label, so there is no symmetry left for the weights to break.

    One random frame per label, then split into copies displaced along the
    principal axis of that label's covariance. The split is what makes this
    worth doing rather than drawing ``L * n`` frames outright: it is the same
    move applied to a *converged* single-Gaussian run, which is the usual way
    into a mixture setup. Swap ``seed_centroids``/``seed_covs`` for a previous
    result's ``out_centroids[N]``/``out_covs[N]`` to start from one, and the
    split then separates each label along the direction its own frames actually
    spread in.

    Note how the two halves stay in step: SplitCentroidsJob and RepeatCovsJob
    both expand label ``l`` into adjacent slots at ``l * n``, so centroid and
    covariance line up density for density.
    """
    seed_centroids = RandomCentroidsJob(features, num_labels, seed=seed).out_centroids
    # One covariance per label. Here every label starts from the same
    # corpus-wide one, so the split direction is shared; from a converged run
    # each label brings its own and the directions differ.
    seed_covs = DuplicateCovsJob(global_cov, num_labels).out_covs
    return dict(
        centroids=SplitCentroidsJob(
            seed_centroids,
            densities_per_label,
            perturbation=0.2,   # the conventional value against a real covariance
            covs=seed_covs,
        ).out_centroids,
        covs=RepeatCovsJob(seed_covs, densities_per_label).out_covs,
        mixtures=UniformMixturesJob(num_labels, densities_per_label).out_mixtures,
    )


def run():
    use_eow_phonemes = False
    num_epochs = 10
    num_clusters = 40 if not use_eow_phonemes else 79   # labels, i.e. score width
    input_data_key = "ls-100-segmented"

    # One flag, in both the RASR config and the flavor: Viterbi hands the
    # accumulator one-hot label posteriors, forward-backward hands it dense
    # gammas, and the mixture E-step takes either without changing.
    use_forward_backward = False

    num_chunks = 20
    num_workers = 8
    lm_order = 3
    subsampling = None
    seed = 42

    lm_scales = [20.0, 30.0, 40.0]
    loop_probs = [0.0]
    distance_scales = [1.0]

    decode_lm_scale = 40.0
    # Training is pre-segmented and so runs at 0.0, but the decode is not: the
    # cv set has ordinary segmentation, where a label spans several frames and
    # needs a self-loop. Pairing 0.0 with it would be the mismatched
    # combination. Same split as viterbi_cov_cheat_seg.py.
    decode_loop_prob = 0.4
    decode_distance_scale = 1.0

    train_beam_size = 100_000
    decode_beam_size = 100_000

    features = input_data[input_data_key]["features"]
    # Computed from the features this run uses rather than taken from
    # constants, so the config depends on nothing precomputed by hand. Same
    # segments and subsampling as the clustering, or the covariance would
    # describe frames the model never sees. `constants.SHARED_COVS` is the
    # precomputed equivalent - reach it with SelectCovJob(SHARED_COVS).out_cov.
    global_cov = GlobalCovarianceJob(
        features, subsampling=subsampling
    ).out_cov
    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    # (name, flavor factory, initial artifacts). Adding a third layout is a
    # third entry - the loop below names no model class.
    # (name, flavor factory, initial artifacts, extra flavor arguments). Adding a
    # fourth layout is a fourth entry - the loop below names no model class.
    layouts = [
        (
            "shared-128",
            mixture_flavor,
            _shared_codebook(features, num_clusters, 128, global_cov, seed),
            {},
        ),
        (
            "per-label-3",
            per_label_mixture_flavor,
            _per_label(features, num_clusters, 3, global_cov, seed),
            {},
        ),
        # Ten densities per label, with one covariance shared across them. Pooling
        # is what makes this size affordable twice over: the covariance is
        # estimated from the label's whole mass rather than a tenth of it, and the
        # per-chunk second moment stays [L, D, D] instead of growing to
        # [10L, D, D] - 80 MB rather than 800 MB.
        (
            "per-label-10-pooled",
            per_label_mixture_flavor,
            _per_label(features, num_clusters, 10, global_cov, seed),
            {"pool_covariances": True},
        ),
    ]

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv"]["features"],
        sampling_method=SegmentFile(input_data["cv"]["segment_file"]),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "layout", "lm_scale", "epoch",
            "per", "del", "ins", "sub", "fer",
            "silence", "l1", "am_score", "transition_score", "lm_score",
        ],
        sort_by=["layout", "lm_scale"],
        # Every epoch that exists, rather than first-and-last: these runs are read
        # for their trajectory. Unfinished epochs are dropped rather than left
        # blank, because a table long enough to overflow a page is truncated - a
        # float cannot break - and the blank rows are what would push it over.
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Chunked k-means with Gaussian mixtures, pre-segmented, random init: "
            f"all epochs, shared codebook vs per-label densities. Training loop "
            f"probability 0.0 and AM scale {distance_scales[0]} throughout; decoded "
            f"at LM scale {decode_lm_scale}, loop {decode_loop_prob}. Statistics "
            f"columns describe the guiding pass that ran with that epoch's model, so "
            f"the final epoch has none."
        ),
    )
    recog_results = []

    for (
        (layout_name, flavor_factory, initial, flavor_kwargs),
        lm_scale,
        loop_prob,
        distance_scale,
    ) in product(layouts, lm_scales, loop_probs, distance_scales):
        _beam = f"{train_beam_size // 1000}k" if train_beam_size else "inf"
        exp_name = (
            f"{layout_name}_seed-{seed}_lm-{lm_order}-{lm_scale}"
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

        # The flavor carries the model, the accumulator and the search as one
        # decision. Note what is *not* here: no initial_centroids/initial_covs,
        # and nothing that names GaussianMixtureModel or the accumulator - the
        # only difference between the two layouts is which factory built this.
        flavor = flavor_factory(
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            # Densities that lose all their weight cannot recover it under the
            # default 0.0, so a mixture only ever shrinks. A small floor keeps
            # every density a reachable candidate, at the cost of no longer
            # being exactly textbook EM. Worth turning off once to see how many
            # densities the data actually supports.
            mixture_floor=1e-6,
            num_workers=num_workers,
            **flavor_kwargs,
            **initial,
        )

        exp_result = chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=features,
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            flavor=flavor,
            subsampling=subsampling,
            distance_scale=distance_scale,
            # Must agree with the RASR config above and with the flavor: this
            # picks the recognizer class, that picks the search RASR runs.
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
        # Per-epoch rather than from the merged file: the merge waits for the
        # last epoch, so the statistics columns would stay blank for the whole
        # run rather than filling in as it goes.
        statistics = clustering_statistics_per_epoch(
            exp_result.out_epoch_statistics, name=exp_name, epoch_offset=1
        )
        # The mixture weights are a per-epoch artifact like any other, and the
        # interesting one here: how much of the codebook each label ends up on.
        tk.register_output(
            f"guided_kmeans/{exp_dir}/mixtures/{exp_name}_final.npy",
            exp_result.out_artifacts["mixtures"][num_epochs],
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

        # From epoch 0, the initialization itself. MaterializeModelJob writes
        # the starting artifacts out as a model directory so it decodes like any
        # other epoch, and that number is the reference the rest of the run is
        # read against: it is the one model whose quality is known in advance,
        # so a bad epoch-0 PER points at the search or the decode setup while a
        # good one followed by bad epochs points at the update.
        for recog_epoch in range(0, num_epochs + 1):
            decode_config = DecodeConfig(
                # Passed for the interface's sake and ignored for scoring: with
                # model_dir set, the callback loads whichever class the manifest
                # names and gets the mixture weights along with the densities.
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
                params={"layout": layout_name, "lm_scale": lm_scale},
                epoch=recog_epoch,
                statistics=statistics,
                # The "fer" column reads from values, not from the result.
                values={"fer": res.fer} if res.fer is not None else {},
            )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_all_epochs_{version}.tex")


def py():
    run()
