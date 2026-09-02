"""Chunked guided k-means: per-label Gaussian mixtures, pre-segmented features, cheating init.

The cheating-init counterpart to ``viterbi_mixture.py``. Same search, same corpus,
same model - :class:`PerLabelMixtureModel` with 3 densities per label - and the
only difference is where the densities start:

``viterbi_mixture.py``      random frames, split along the *corpus* covariance
``this config``        the cheating centroids, split along each label's *own*
                       cheating covariance

so the pair measures what the mixture buys once the starting point is already
right, separately from what it buys as a way of finding one. Against
``viterbi_cov_cheat_init.py`` - the same cheating init with one density per
label - it measures what the extra densities buy on their own.

The initialization is three small jobs and worth reading as a unit::

    SplitCentroidsJob(cheating_centroids, 3, covs=cheating_covs)  -> [120, 512]
    RepeatCovsJob(cheating_covs, 3)                               -> [120, 512, 512]
    UniformMixturesJob(40, 3)                                     -> [40, 3]

Each label's three densities are placed along the principal axis of *that
label's* covariance, at -0.2, 0, +0.2 of its standard deviation along that
axis, and all three then start from that same covariance. So the split happens
in the direction the label's own frames actually spread in, which is the whole
reason to pass a covariance rather than jitter isotropically - and the three
densities begin as an honest decomposition of the single Gaussian
``viterbi_cov_cheat_init.py`` would have used, rather than as three guesses.

Sharing the covariance across the three is deliberate: splitting the mean and
the covariance at once would leave each density with a third of the evidence
and a full [512, 512] to estimate from it. The covariances separate on their
own from the first M-step.

Only the per-label layout appears here. A shared codebook has no natural
cheating init - the cheating centroids are per label, and a codebook is by
definition not.
"""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    INITIALIZATIONS,
    GMM_ALIGNMENT_CV,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    RepeatCovsJob,
    SplitCentroidsJob,
    UniformMixturesJob,
    chunked_clustering,
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

exp_dir = "viterbi_mixture_cheat_init"
version = 1


def _cheating_per_label(densities_per_label):
    """Initial artifacts for the per-label mixture, from the cheating model.

    ``SplitCentroidsJob`` and ``RepeatCovsJob`` expand label ``l`` into the same
    adjacent slots at ``l * n``, so centroid and covariance line up density for
    density - that agreement is what the layout depends on and what the tests
    pin down.
    """
    centroids = INITIALIZATIONS["cheating_centroids"]    # [L, D]
    covs = INITIALIZATIONS["cheating_covs"]              # [L, D, D]
    num_labels = 40
    return dict(
        centroids=SplitCentroidsJob(
            centroids,
            densities_per_label,
            perturbation=0.2,   # the conventional value against a real covariance
            covs=covs,
        ).out_centroids,
        covs=RepeatCovsJob(covs, densities_per_label).out_covs,
        mixtures=UniformMixturesJob(num_labels, densities_per_label).out_mixtures,
    )


def run():
    use_eow_phonemes = False
    num_epochs = 10
    num_clusters = 40 if not use_eow_phonemes else 79   # labels, i.e. score width
    input_data_key = "ls-100-segmented"

    # Kept in step with viterbi_mixture.py so the two are comparable; see there for
    # why this is one flag across the RASR config, the flavor and the pipeline.
    use_forward_backward = False

    num_chunks = 20
    num_workers = 8
    lm_order = 3
    subsampling = None

    lm_scales = [20.0, 30.0, 40.0]
    loop_probs = [0.0]              # pre-segmented: a segment is one label
    distance_scales = [1.0]

    decode_lm_scale = 40.0
    # Training is pre-segmented (loop 0.0); the cv decode set is not, so it
    # needs a self-loop. See viterbi_mixture.py.
    decode_loop_prob = 0.4
    decode_distance_scale = 1.0

    train_beam_size = 100_000
    decode_beam_size = 100_000

    features = input_data[input_data_key]["features"]
    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    # (name, densities per label, extra flavor arguments). Ten densities per label
    # only becomes affordable with a pooled covariance - see viterbi_mixture.py -
    # and pooling also gives that covariance the label's whole mass instead of a
    # tenth of it, which is the shortage the 3-density run runs into.
    arms = [
        ("per-label-3", 3, {}),
        ("per-label-10-pooled", 10, {"pool_covariances": True}),
    ]

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv"]["features"],
        sampling_method=SegmentFile(input_data["cv"]["segment_file"]),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "arm", "lm_scale", "epoch",
            "per", "del", "ins", "sub", "fer",
            "silence", "l1", "am_score", "transition_score", "lm_score",
        ],
        sort_by=["arm", "lm_scale"],
        # Every epoch that exists, rather than first-and-last: these runs are read
        # for their trajectory. Unfinished epochs are dropped rather than left
        # blank, because a table long enough to overflow a page is truncated - a
        # float cannot break - and the blank rows are what would push it over.
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Chunked k-means with per-label densities, pre-segmented, "
            f"cheating init: all epochs. Training loop probability 0.0 and AM scale "
            f"{distance_scales[0]} throughout; decoded at LM scale {decode_lm_scale}, "
            f"loop {decode_loop_prob}. Statistics columns describe the guiding pass "
            f"that ran with that epoch's model, so the final epoch has none."
        ),
    )
    recog_results = []

    for (arm_name, densities_per_label, flavor_kwargs), lm_scale, loop_prob, distance_scale in (
        product(arms, lm_scales, loop_probs, distance_scales)
    ):
        initial = _cheating_per_label(densities_per_label)
        _beam = f"{train_beam_size // 1000}k" if train_beam_size else "inf"
        exp_name = (
            f"{arm_name}_lm-{lm_order}-{lm_scale}"
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

        flavor = per_label_mixture_flavor(
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            # See viterbi_mixture.py: 0.0 is textbook EM but lets a density's weight
            # reach zero and stay there. Kept identical to the random-init run
            # so the two differ only in where they start.
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
        # Started uniform at 1/3 each, so how far these move says how much of
        # the split the data actually wanted.
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
                params={"arm": arm_name, "lm_scale": lm_scale},
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
