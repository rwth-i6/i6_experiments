"""Unsupervised training of a discrete HMM over a frozen codebook.

The unsupervised counterpart of :mod:`.vq_supervised`, and the first
unsupervised config in this setup with a measured ceiling to be judged against.
Everything except the table is an input: the codebook is fixed, quantization is
plain L2, and an epoch does nothing but recognize, count ``p(codeword | label)``
and normalize.

What the supervised run established
    Counting that table from a reference alignment gives **82.4% held-out
    segment accuracy** and decodes to **14.4% PER** at LM scale 1.0. So the
    codebook, the model, the search and the decode configuration are all sound,
    and any failure here is in the unsupervised estimation alone. That is a
    much sharper target than the mixture runs ever had.

    The same sweep also fixed the scale: 14.4% at LM scale 1.0 against 39% at
    5.0 and 74% at 20.0. Note *why* scale 0.0 is worst of all (98%): with
    ``transition_scale=None`` the transition scale follows the LM scale, so 0.0
    removes the transition costs too and the search inserts freely. The useful
    range is narrow and near 1.0.

Why the initialization is the experiment
    With the codebook frozen the table is the whole model, so a table that
    starts near-uniform gives every label the same score column and hands the
    first counting step an alignment the language model produced on its own.
    The mixture runs died exactly there. :class:`...NormalTableJob` draws each
    entry from a normal and renormalizes each row, which is the initialization
    the reference setup uses; ``sigma`` is the coefficient of variation that
    survives the renormalization, and it is swept rather than guessed. Small
    sigma is the *weak* break: at C=512, sigma near 1.0 corresponds to a
    Dirichlet(1.0) draw and sigma near 3.1 to Dirichlet(0.1).

Two corpora
    ``cv-nosil``
        This setup's own cv features, segmented on the reference alignment with
        the silence segments removed - 326,171 vectors. Small and quick, and
        the only silence-free set we can build: silence removal needs a frame
        alignment, and this setup has one for cv and for a 120-sequence debug
        subset, but **not for ls-100**. Training and decoding both happen on cv
        here, which is legitimate for an unsupervised run (no labels are used
        either way) but means these numbers are not held out.
    ``ls100-nosil``
        This setup's own ls-100h features, segmented on the 960h frame
        alignment with silence removed - ~3.4M vectors. The alignment covers
        all 28,234 sequences frame for frame; what differs is the corpus prefix
        in the tags (``train-clean-100`` against ``train-other-960``), which
        gives zero exact matches until it is stripped.
    ``ls960-segments``
        A colleague's LibriSpeech-960 GMM segment representations, 32,666,275
        vectors already segmented and silence-free. This is the corpus the
        reference result was obtained on, and the one to judge a reproduction
        by. Roughly 100x the cv set, so it carries the epoch cost.

Two language models
    This setup's count trigram, and a colleague's convolutional LM distilled to
    ARPA. They differ mainly in the tail - the latter puts markedly less mass on
    unlikely trigrams, so it is sharper - and since the search is what turns
    acoustic scores into the counts this whole model is made of, a sharper LM is
    not a neutral change. Both are run.
"""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512,
    GMM_ALIGNMENT_LS960_FRAME,
    COLLEAGUE_SEGMENT_FEATURES_LS960,
    PHONEME_LM_ZIJIAN_3GRAM,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    NormalTableJob,
    chunked_clustering,
    vq_flavor,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
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

exp_dir = "vq_unsupervised"
version = 1


# --- shared run construction ------------------------------------------------
# Extracted so that :mod:`.vq_unsupervised_long` builds *bit-identical* jobs for
# the epochs both configs cover. Continuing a run is only free if every epoch's
# spec matches exactly (see chunked_clustering's docstring), and the cheapest way
# to guarantee that is for the two configs to run the same code rather than to
# keep two copies of the same argument list in step by hand.

#: Everything held fixed across every VQ run. Changing anything here changes the
#: job hashes and orphans the epochs already computed.
USE_EOW_PHONEMES = False
NUM_LABELS = 40 if not USE_EOW_PHONEMES else 79
NUM_CODEWORDS = 512
LM_ORDER = 3
USE_FORWARD_BACKWARD = True
NUM_WORKERS = 8
SUBSAMPLING = None
LM_SCALE = 1.0
DISTANCE_SCALE = 1.0
LOOP_PROB = 0.0
BEAM_SIZE = 100_000
TABLE_FLOOR = 1e-2


def build_vq_training(
    *,
    features,
    lm_path,
    sigma,
    seed,
    num_epochs,
    num_chunks,
    lexicon,
    alias_prefix,
    num_workers=NUM_WORKERS,
    rqmt=None,
):
    """One unsupervised VQ run, as (recognition_config, ChunkedClusteringExpResult).

    **Every scheduling knob here is excluded from the job hash**, which is what
    makes a continuation free to be tuned. ``num_chunks`` is unhashed because
    merging counts is associative, so the partition cannot change the result;
    ``num_workers`` sits in the recognizer spec's ``unhashed_kwargs``; ``rqmt``
    is in ``GuidedClusteringEpochJob.hash``'s exclusion set. ``num_epochs`` is
    not part of an epoch job's identity either - asking for 100 epochs produces
    the same first ten jobs as asking for ten - and ``alias_prefix`` only names
    aliases. So the epochs already computed survive any of these changing.

    Sizing them is a scheduling problem, not a modelling one, and the numbers
    that matter were measured on this cluster rather than guessed:

    * the QOS caps this user at **1100 CPUs** on ``cpu_modern``; nothing else
      binds, and pending tasks report ``QOSMaxCpuPerUserLimit``;
    * **Slurm rounds an allocation up to an even core count.** Requesting
      ``cpu=9`` (``num_workers=8``) yields ``AllocCPUS=10``, so one core per
      task is paid for and never used - 10% of the quota. ``num_workers=9``
      makes the request 10, which is what gets allocated either way;
    * a task's peak RSS is **5 GB**, so the default ``mem=16`` reserves three
      times what it uses. At 110 concurrent tasks that is 1.76 TB against a
      2.2 TB QOS memory cap - not binding today, but the first thing that would
      bind if task count rose;
    * chunks are balanced to ~0.3% by :func:`...chunked.features.plan_chunks`,
      and a measured epoch array runs at 89-90% utilisation with a max/median
      of 1.1x. **There is no straggler problem**; what wastes the quota is the
      gap between one epoch's array finishing and the next being scheduled.

    That last point is why ``num_chunks`` is the knob that matters. An epoch job
    of 100 tasks takes ~1000 of the 1100 CPUs, so only one epoch job runs at a
    time and every run's per-epoch overhead is paid serially. Sizing an epoch
    job to ``1100 / (number of concurrent runs)`` instead lets the runs overlap,
    so one run's scheduling gap is another run's compute.

    :param num_workers: RASR search processes per task. The task requests
        ``num_workers + 1`` CPUs, so this also sets the scheduling granularity;
        larger values amortize that ``+1`` better but make each task chunkier.
    :param rqmt: overrides for the epoch job's requirements, e.g. ``{"mem": 8}``
    """
    recognition_config = create_recog_rasr_config(
        lm_scale=LM_SCALE,
        emission_scale=1.0,
        transition_scale=LM_SCALE,
        loop_probability=LOOP_PROB,
        silence_loop_probability=LOOP_PROB,
        use_forward_backward_search=USE_FORWARD_BACKWARD,
        lm_order=LM_ORDER,
        use_eow_phonemes=USE_EOW_PHONEMES,
        max_beam_size=BEAM_SIZE,
        lm_path=lm_path,
    )
    flavor = vq_flavor(
        centroids=COLLEAGUE_CENTROIDS_K512,
        table=NormalTableJob(NUM_LABELS, NUM_CODEWORDS, sigma=sigma, seed=seed).out_table,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=NUM_LABELS,
        distance_scale=DISTANCE_SCALE,
        use_forward_backward=USE_FORWARD_BACKWARD,
        table_floor=TABLE_FLOOR,
        num_workers=num_workers,
    )
    exp_result = chunked_clustering(
        num_epochs=num_epochs,
        features_hdf=features,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=NUM_LABELS,
        flavor=flavor,
        subsampling=SUBSAMPLING,
        distance_scale=DISTANCE_SCALE,
        use_forward_backward=USE_FORWARD_BACKWARD,
        rasr_path=(
            tools.RASR_PATH_FORWARD_BACKWARD if USE_FORWARD_BACKWARD else tools.RASR_PATH
        ),
        num_chunks=num_chunks,
        num_workers=num_workers,
        rqmt=rqmt,
        alias_prefix=alias_prefix,
    )
    return recognition_config, exp_result


def build_decode_config(lm_path, decode_lm_scale, decode_loop_prob):
    """The decode-side RASR config, shared for the same reason."""
    return create_recog_rasr_config(
        lm_scale=decode_lm_scale,
        emission_scale=1.0,
        transition_scale=None,
        loop_probability=decode_loop_prob,
        silence_loop_probability=decode_loop_prob,
        lm_order=LM_ORDER,
        use_eow_phonemes=USE_EOW_PHONEMES,
        max_beam_size=BEAM_SIZE,
        lm_path=lm_path,
    )


def silence_free_ls100_features():
    """ls-100h segmented on the 960h frame alignment, silence dropped."""
    job = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="mean",
        rqmt={"cpu": 2, "mem": 16, "time": 8},
    )
    return job


def silence_free_cv_features():
    """cv segmented on its own reference alignment, silence dropped."""
    return SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="mean",
    )



def run():
    use_eow_phonemes = False
    num_labels = 40 if not use_eow_phonemes else 79
    num_codewords = 512
    num_epochs = 10
    lm_order = 3

    use_forward_backward = True
    num_workers = 8
    subsampling = None

    # Fixed at what the supervised sweep found, so the initialization is the
    # only thing varying. Deliberately not swept again here: 1.0 beat 5.0 by 25
    # points and 20.0 by 60, and the transition scale follows it.
    lm_scale = 1.0
    distance_scale = 1.0
    loop_prob = 0.0
    decode_lm_scale = 1.0
    decode_loop_prob = 0.0
    beam_size = 100_000

    # A zero entry is +inf and absorbing under a frozen codebook, and a codeword
    # no label admits leaves a frame unsearchable. The supervised sweep found
    # PER flat across floors (14.357 / 14.356 / 14.350 at 1e-3 / 1e-2 / 1e-1),
    # so one value is enough here and 1e-2 is the middle of that range.
    table_floor = 1e-2

    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    cv_features = silence_free_cv_features()
    cv_features.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

    # ls-100h, segmented on the 960h frame alignment with silence dropped. The
    # alignment covers all 28,234 sequences frame for frame; only the corpus
    # prefix differs (train-clean-100 against train-other-960) and the job
    # normalizes that away. ~3.4M segments once silence is gone.
    ls100_features = silence_free_ls100_features()
    ls100_features.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_nosil")
    tk.register_output(
        f"guided_kmeans/{exp_dir}/features/ls100_nosil_statistics.json",
        ls100_features.out_statistics,
    )

    # (name, features, num_chunks, sigmas, seeds). The chunk counts follow the
    # corpus sizes: cv is 326k vectors over 2786 sequences, the 960h set is
    # 32.7M over 278k, and RASR search is what an epoch costs.
    corpora = [
        # ("cv-nosil", cv_features.out_features, 20, [0.1, 1.0], [42, 43, 44]),
        ("ls100-nosil", ls100_features.out_features, 50, [0.1, 1.0], [42, 43]),
        # ("ls960-segments", COLLEAGUE_SEGMENT_FEATURES_LS960, 200, [1.0], [42, 43]),
    ]
    language_models = [("ours-3gram", None), ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM)]

    # Decoding always happens on the silence-free cv set, whichever corpus the
    # table was estimated on - otherwise the two arms are not comparable, and
    # the 960h set has no reference to score against anyway.
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features.out_features,
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "corpus", "lm", "sigma", "seed", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["corpus", "lm", "sigma", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised discrete-HMM training over a frozen 512-entry codebook: "
            f"{num_epochs} epochs of counting p(codeword|label) under a "
            f"forward-backward search, decoded on silence-free cv at LM scale "
            f"{decode_lm_scale}. The same table counted from a reference alignment "
            f"gives 82.4\\% held-out segment accuracy and 14.4\\% PER, which is the "
            f"ceiling these runs are to be read against. 'mi' is I(label; codeword) "
            f"under the model's own table, reference-free and 0 for the degenerate "
            f"all-labels-alike solution."
        ),
    )
    recog_results = []

    for (corpus_name, features, num_chunks, sigmas, seeds), (lm_name, lm_path) in product(
        corpora, language_models
    ):
        for sigma, seed in product(sigmas, seeds):
            exp_name = f"{corpus_name}_{lm_name}_sigma-{sigma}_seed-{seed}"

            recognition_config, exp_result = build_vq_training(
                features=features,
                lm_path=lm_path,
                sigma=sigma,
                seed=seed,
                num_epochs=num_epochs,
                num_chunks=num_chunks,
                lexicon=lexicon,
                alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
            )

            tk.register_output(
                f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json",
                exp_result.out_statistics,
            )
            statistics = clustering_statistics_per_epoch(
                exp_result.out_epoch_statistics,
                name=exp_name,
                epoch_offset=1,
                lexicon=lexicon,
            )
            # The table is shaped exactly like a mixture weight matrix - label by
            # codeword, rows summing to 1 - so the mixture diagnostics apply to
            # it unchanged, and I(label; codeword) means the same thing.
            diagnostics = {
                epoch: MixtureDiagnosticsJob(exp_result.out_artifacts["table"][epoch])
                for epoch in range(0, num_epochs + 1)
            }
            for epoch, job in diagnostics.items():
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/table_diagnostics/{exp_name}_ep-{epoch}.json",
                    job.out_diagnostics,
                )

            recognition_config_decode = build_decode_config(
                lm_path, decode_lm_scale, decode_loop_prob
            )
            for recog_epoch in (0, num_epochs // 2, num_epochs):
                decode_config = DecodeConfig(
                    centroids=COLLEAGUE_CENTROIDS_K512,
                    model_dir=exp_result.out_models[recog_epoch],
                    recog_rasr_config=recognition_config_decode,
                    distance_scale=distance_scale,
                    subsampling=subsampling,
                    write_frame_labels=True,
                )
                decode_name = f"{exp_name}_ep-{recog_epoch}"
                res = decode_and_score(
                    decode_name,
                    "cv",
                    decode_config,
                    cv_dataset,
                    rasr_path=tools.RASR_PATH,
                    device="cpu",
                    corpus_key="train-other-960",
                )
                if res.frame_labels is not None:
                    res.fer = FrameErrorRateJob(
                        res.frame_labels, GMM_ALIGNMENT_CV, lexicon
                    ).out_fer
                    tk.register_output(
                        f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer
                    )
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per
                )
                recog_results.append(res)
                latex_report.add_row(
                    result=res,
                    params={
                        "corpus": corpus_name, "lm": lm_name,
                        "sigma": sigma, "seed": seed,
                    },
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
