"""Initializing the VQ table with the analytic first EM step.

The unsupervised VQ runs start from a random table because the table is the
whole model and a uniform one leaves every label scoring alike
(:class:`...chunked_clustering.NormalTableJob`). But a uniform table is also the
one case where the first E-step is free: the emission product over a label
sequence of fixed length is ``K^-T`` for every sequence, so it cancels out of the
sequence posterior and leaves

    p(c_1^T | x_1^T) = p(c_1^T)          exactly, given |c| = T
    gamma_t(c)       = p(c_t = c)        the positional unigram of the label prior

No acoustics, no search. So the first re-estimation can be computed in closed
form from label sequences plus one pass of quantization over the features, and
what would have been the most expensive epoch of a run - a full RASR
forward-backward over the corpus - becomes a GEMM. The method, its smoothing and
its diagnostics are specified in ``docs/positional_unigram_init/``.

Why the segmented arm and not the frame-level one
    ``|c| = T`` has to hold. Here it does by construction: one mean-pooled vector
    per phone segment, ``LOOP_PROB = 0.0`` and ``skip="infinity"``, so the search
    can neither stay on a label nor skip one and every frame consumes exactly one
    label token. At frame level a label spans several frames and the duration
    model, not the label prior, would decide the profile.

Search settings
    The 5-gram count LM at beam 1000, taken from :mod:`.vq_gmm_tau` where they
    were established, rather than the 3-gram at beam 100,000 that
    :mod:`.vq_unsupervised` swept. Both are hashed, so these runs share no epoch
    job with the 3-gram ones, and the tight beam is what makes 40 epochs
    affordable. It also removes the second language model: the Zijian model is a
    trigram ARPA and cannot stand in for a 5-gram.

    Only the analytic arm runs. The random-init comparison under these settings
    has been run elsewhere; re-deriving it here would cost a full sweep to
    reproduce numbers that exist.

Where the label sequences come from
    The corpus's own transcriptions, phonemized through the lexicon. Text only,
    no alignment - the same information class as the phoneme LM the search
    already uses. ``SegmentedFeaturesFromAlignmentJob.out_labels`` would be the
    oracle version of the same input and is one keyword away
    (``positional_unigram_table(labels=...)``), which is the A/B worth running if
    the text estimate looks weak.

What to look at first, before any of this decodes
    ``gamma_diagnostics``. The matrix carries no acoustic information, so it
    predicts whether uniform initialization can work at all, and both halves of
    that prediction were measured on cv before this config was written:

    **gamma has structure, at the utterance edges.** Three bands, effective rank
    7.19 / 3.28 / 3.29 out of 40, with the L1 deviation from the band mean at
    0.34 in the first bin and 0.19 at the interior maximum (band 1). It is
    linguistic: ``DH`` is the 4th most likely label at ``tau = 0`` and out of the
    top five by ``tau = 1``, while ``S``, ``D`` and ``T`` rise towards the end.
    Short utterances carry more of it than long ones (7.19 against 3.3).

    **The table is near-degenerate anyway.** Pushed through the M-step on a sixth
    of cv: max row cosine 0.99996, 80% of label pairs above 0.99, effective rank
    2.09, and ``I(label; codeword) = 0.0016`` nats against 2.27 for the
    supervised table. The M-step averages the positional tilt away - §5's exact
    fixed point, arrived at by measurement rather than by argument.

    That is a statement about the label prior, not about this code, and it does
    not make the method pointless: a real FB epoch from a uniform table computes
    the *same* near-degenerate table, which is what §0 says and what the
    cross-check below tests. What this buys is that epoch, not a better start. If
    a better start is wanted, §5 names the remedies - an epsilon symmetry break
    on top of ``pi1``, or shorter segments, which the band-0 numbers support.
"""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    NormalTableJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.positional_unigram import (
    TableComparisonJob,
    positional_unigram_table,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
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
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans import tools

# Everything about the run itself is shared with the random-init runs, so that
# the initialization is the only thing that differs and the two arms are
# comparable row for row.
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    NUM_CODEWORDS,
    NUM_LABELS,
    build_decode_config,
    build_vq_training,
    silence_free_cv_features,
    silence_free_ls100_features,
)

exp_dir = "vq_unigram_init"
version = 1

NUM_EPOCHS = 40
DECODE_LM_SCALE = 1.0
DECODE_LOOP_PROB = 0.0
DISTANCE_SCALE = 1.0

#: Search settings carried over from :mod:`.vq_gmm_tau`, which is where they were
#: established - a 5-gram rather than the 3-gram ``vq_unsupervised`` swept, and a
#: beam three orders of magnitude tighter. Both are hashed: they change what the
#: search computes, so these runs share no epoch job with the 3-gram ones.
#:
#: The beam is what makes 40 epochs affordable here. Note it applies to training
#: *and* decoding, which is deliberate - a table estimated under one beam and
#: decoded under another is being read at a different operating point than it was
#: fitted at.
LM_ORDER = 5
BEAM_SIZE = 1_000

#: §2.4. Bin width has to sit well below the profile width in ``tau``, which is
#: of order ``1/sqrt(T)``; at the measured mean ``T ~ 125`` that is 1/64 against
#: 1/11. Oversizing B is cheap, undersizing destroys structure irrecoverably.
NUM_BINS = 64

SEED = 42

#: §7.6, off by default. Turning it on adds one *extra* arm per corpus - a single
#: epoch from a genuinely uniform table (``sigma = 0`` makes every entry of
#: ``NormalTableJob``'s draw equal) - whose re-estimated table has to reproduce
#: the analytic one, because under a uniform table those are the same
#: computation. It is the strongest available check that this initialization is
#: what it claims to be, and the only thing in this config that trains a table
#: initialized any other way.
#:
#: It runs on cv only - the check is about correctness, not scale - so it also
#: needs the cv arm uncommented in ``corpora`` below. With cv commented out this
#: flag does nothing whichever way it is set.
RUN_FB_CROSSCHECK = False


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    cv_features = silence_free_cv_features()
    cv_features.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")
    ls100_features = silence_free_ls100_features()
    ls100_features.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_nosil")

    # Decoding always happens on the silence-free cv set, whichever corpus the
    # table was estimated on - otherwise the two arms are not comparable.
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features.out_features,
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
    )

    # (name, job, bliss corpus the tags belong to, chunks for the histogram,
    # chunks for an epoch). cv's tags carry the train-other-960 prefix because
    # the set is a held-out slice of it; ls-100h's carry train-clean-100.
    corpora = [
        # ("cv-nosil", cv_features, "train-other-960", 4, 20),
        ("ls100-nosil", ls100_features, "train-clean-100", 16, 50),
    ]
    # One entry: at LM_ORDER = 5 the only LM available is this setup's own count
    # model. PHONEME_LM_ZIJIAN_3GRAM is a trigram ARPA and cannot stand in for a
    # 5-gram, which is why vq_gmm_tau drops it whenever the order is not 3.
    language_models = [("ours-5gram", None)]

    latex_report = LatexTableReport(
        columns=[
            "corpus", "lm", "init", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["corpus", "lm", "init"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised discrete-HMM training over a frozen {NUM_CODEWORDS}-entry "
            f"codebook, initialized with the analytic first EM step: the positional "
            f"unigram of the label prior, estimated from transcriptions alone and "
            f"consumed in one GEMM, with no search run for iteration 1 at all. "
            f"{NUM_EPOCHS} epochs under the {LM_ORDER}-gram count LM at beam "
            f"{BEAM_SIZE}, decoded on silence-free cv at LM scale {DECODE_LM_SCALE}. "
            f"The random-init arms under the same settings are not repeated here. The "
            f"same table counted from a reference alignment gives 82.4\\% held-out "
            f"segment accuracy and 14.4\\% PER, which is the ceiling this is read "
            f"against; measured before training, the analytic table carries "
            f"I(label; codeword) = 0.0016 nats against that table's 2.27, so epoch 0 "
            f"is expected to be near-degenerate and the trajectory is the result. "
            f"'mi' is I(label; codeword) under the model's own table, reference-free "
            f"and 0 for the degenerate all-labels-alike solution."
        ),
    )
    recog_results = []

    for corpus_name, features_job, corpus_key, histogram_chunks, num_chunks in corpora:
        # --- the initialization itself: no search, no epochs ------------------
        # Built once per corpus and reused by every language model arm: gamma is
        # a property of the label prior and the codebook, and the LM only enters
        # once an epoch runs.
        unigram = positional_unigram_table(
            features_hdf=features_job.out_features,
            centroids=COLLEAGUE_CENTROIDS_K512,
            lexicon=lexicon,
            num_labels=NUM_LABELS,
            corpus_key=corpus_key,
            segments=features_job.out_segments,
            num_bins=NUM_BINS,
            num_chunks=histogram_chunks,
            alias_prefix=f"guided_kmeans/{exp_dir}/{corpus_name}",
        )
        for name, path in (
            ("band_edges", unigram.band_edges),
            ("gamma_diagnostics.json", unigram.gamma_diagnostics),
            ("table_diagnostics.json", unigram.table_diagnostics),
            ("length_agreement.json", unigram.length_agreement),
            ("profiles", unigram.plots),
            ("table.npy", unigram.table),
        ):
            tk.register_output(f"guided_kmeans/{exp_dir}/{corpus_name}/{name}", path)

        # The initial table is a model in its own right and worth the same
        # diagnostics the trained ones get - it is shaped exactly like a mixture
        # weight matrix, label by codeword with rows summing to 1.
        tk.register_output(
            f"guided_kmeans/{exp_dir}/{corpus_name}/table_mi.json",
            MixtureDiagnosticsJob(unigram.table).out_diagnostics,
        )

        # The analytic initialization alone. The random-init arms that would sit
        # next to it here have been run already elsewhere, and re-running them
        # under these search settings would cost a full sweep to reproduce a
        # comparison that exists - add
        # ``("normal-0.02", None, 0.02, SEED)`` back to get it.
        initializations = [
            ("unigram", unigram.table, None, None),
        ]

        for (lm_name, lm_path), (init_name, table, sigma, seed) in product(
            language_models, initializations
        ):
            exp_name = f"{corpus_name}_{lm_name}_init-{init_name}"

            recognition_config, exp_result = build_vq_training(
                features=features_job.out_features,
                lm_path=lm_path,
                sigma=sigma,
                seed=seed,
                table=table,
                num_epochs=NUM_EPOCHS,
                num_chunks=num_chunks,
                lexicon=lexicon,
                lm_order=LM_ORDER,
                beam_size=BEAM_SIZE,
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
            diagnostics = {
                epoch: MixtureDiagnosticsJob(exp_result.out_artifacts["table"][epoch])
                for epoch in range(0, NUM_EPOCHS + 1)
            }
            for epoch, job in diagnostics.items():
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/table_diagnostics/{exp_name}_ep-{epoch}.json",
                    job.out_diagnostics,
                )

            recognition_config_decode = build_decode_config(
                lm_path,
                DECODE_LM_SCALE,
                DECODE_LOOP_PROB,
                lm_order=LM_ORDER,
                beam_size=BEAM_SIZE,
            )
            # Epoch 0 is the initialization itself, which is the point of the
            # comparison: whatever the analytic table is worth before any
            # training, it is worth it here.
            for recog_epoch in (0, NUM_EPOCHS // 2, NUM_EPOCHS):
                decode_config = DecodeConfig(
                    centroids=COLLEAGUE_CENTROIDS_K512,
                    model_dir=exp_result.out_models[recog_epoch],
                    recog_rasr_config=recognition_config_decode,
                    distance_scale=DISTANCE_SCALE,
                    subsampling=None,
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
                    alias_prefix=f"guided_kmeans/{exp_dir}"
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
                    params={"corpus": corpus_name, "lm": lm_name, "init": init_name},
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

        # --- §7.6: the analytic table against a real forward-backward pass ----
        # One epoch from a genuinely uniform table (sigma = 0 makes every entry
        # of NormalTableJob's draw equal) has to reproduce the analytic table,
        # because that is the case in which the two are the same computation.
        # cv only: the check is about correctness, not about scale, and one cv
        # epoch is cheap where an ls-100h one is not.
        #
        # Off by default - it is the one thing here that trains from a table this
        # method did not produce, and the config is otherwise a single arm.
        if not RUN_FB_CROSSCHECK or corpus_name != "cv-nosil":
            continue
        for lm_name, lm_path in language_models:
            uniform_table = NormalTableJob(
                NUM_LABELS, NUM_CODEWORDS, sigma=0.0, seed=SEED
            ).out_table
            _, uniform_result = build_vq_training(
                features=features_job.out_features,
                lm_path=lm_path,
                table=uniform_table,
                num_epochs=1,
                num_chunks=num_chunks,
                lexicon=lexicon,
                lm_order=LM_ORDER,
                beam_size=BEAM_SIZE,
                alias_prefix=f"guided_kmeans/{exp_dir}/{corpus_name}_{lm_name}_fb-crosscheck",
            )
            comparison = TableComparisonJob(
                reference=unigram.table,
                hypothesis=uniform_result.out_artifacts["table"][1],
            )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/crosscheck/{corpus_name}_{lm_name}.json",
                comparison.out_report,
            )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
