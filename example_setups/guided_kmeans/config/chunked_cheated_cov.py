"""
Chunked counterpart of ``cheated_clustering.py``.

Same experiment, executed as one job per epoch with the RASR search spread
over ``num_chunks`` cluster tasks instead of one node's worker pool. The
per-epoch decode/scoring loop below is unchanged from the original config -
``ChunkedClusteringExpResult`` exposes the same ``out_centroids[epoch]`` /
``out_covs[epoch]`` / ``out_statistics`` surface.

Two differences to be aware of when comparing against the original:

* Initialization is not a phase-0 pass over the corpus. Pass the starting
  centroids in explicitly (here: the cheating centroids the original config
  already uses via ``PreloadCentroidsInitializerConfig``). For a random start,
  produce the centroids in a separate job and pass its output here.
* ``num_chunks``/``num_workers`` are unhashed scheduling knobs; changing them
  reuses the same job directory. If chunk results from an earlier run with a
  different ``num_chunks`` are still around, the reduce step says so and the
  fix is to delete the job's ``work/`` directory.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import chunked_clustering
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, SegmentFile
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.corpus_setup import phoneme_corpus, setup_corpus
from i6_experiments.example_setups.guided_kmeans.setup.score import TaggedCorpusToTxtJob
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import LatexTableReport, clustering_statistics

exp_dir = "cov_cheating_seg"

input_data = {
    "train-clean-100-dbg": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/filtered_features_train-clean-100-dbg.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/train-clean-100-dbg/centroids.npy"),
        "cheating_covs": tk.Path("/u/mann/experiments/2026-06-09--guided-k-means/test/cheating_centroids_larissa/covs.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/train-clean-100-dbg-segments.txt"),
    },
    "ls-100": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/wav2vec2_ls100h.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids.npy"),
        "cheating_covs": tk.Path("/u/mann/experiments/2026-06-09--guided-k-means/test/cheating_centroids_larissa/covs.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/ls100h-segments.txt"),
    },
    "ls-100-segmented": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/segmented_features_wav2vec2_ls100h.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids.npy"),
        "cheating_covs": tk.Path("/u/mann/experiments/2026-06-09--guided-k-means/test/cheating_centroids_larissa/covs.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/ls100h-segments.txt"),
    },
}

rasr_path = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")


def run():
    corpus_file = phoneme_corpus(setup_corpus("train-clean-100"))
    # Reference for scoring each epoch's own recognition of the clustering
    # corpus (see chunked_clustering(score_reference=...)). This is the whole
    # corpus, not the sampled subset the decodes below use, and the recognition
    # is the guided one with this run's lm_scale - a convergence diagnostic, not
    # a number to compare against the decoding columns.
    guided_reference = TaggedCorpusToTxtJob(corpus_file).out_txt

    use_eow_phonemes = False
    use_pruning = False
    num_epochs = 10
    input_data_key = "ls-100-segmented"

    # 28234 seqs at ~17 s of search each: 30 chunks x 8 workers puts an epoch
    # at roughly half an hour, against ~9 h for the single-node pipeline.
    num_chunks = 30
    num_workers = 8
    lm_order = 3
    subsampling = None
    loop_probability, silence_loop_probability = 0.0, 0.0

    parameters = [
        # lm_scale
        # 1.0, 20.0, 30.0, 40.0, 50.0, 100.0, 1000.0
        # 40.0
        2.0
    ]

    recognition_config_decode = create_recog_rasr_config(
        lm_scale=40.0,
        emission_scale=1.0,
        transition_scale=None,
        loop_probability=0.3,
        silence_loop_probability=0.3,
        use_tree_search=False,
        max_beam_size=20000 if use_pruning else None,
        score_threshold=10000.0 if use_pruning else None,
        lm_order=3,
        use_eow_phonemes=use_eow_phonemes,
    )

    recognition_config_decode_cheating_seg = create_recog_rasr_config(
        lm_scale=40.0,
        emission_scale=1.0,
        transition_scale=None,
        loop_probability=0.0,
        silence_loop_probability=0.0,
        use_tree_search=False,
        max_beam_size=20000 if use_pruning else None,
        score_threshold=10000.0 if use_pruning else None,
        lm_order=3,
        use_eow_phonemes=use_eow_phonemes,
    )

    recog_results = []
    recog_results_cheat_seg = []

    # One table per decoding: the two differ only in the recognition config they are
    # decoded with, so they need separate rows but share the training statistics.
    # guided_per is the PER of the epoch's own guiding search over the whole
    # clustering corpus; the per/del/ins/sub block is the separate decoding of
    # the sampled subset. The last epoch has no guided score - no epoch ever
    # recognized with that model - so that cell stays blank.
    latex_columns = [
        "lm_scale",
        "epoch", "per", "del", "ins", "sub",
        "guided_per",
        "l1", "am_score", "lm_score",
    ]
    latex_report = LatexTableReport(
        columns=latex_columns,
        sort_by=["lm_scale"],
        epochs=(0, num_epochs - 1),
        caption=(
            "Chunked guided k-means with covariances, decoded with loop probability 0.3: "
            f"epoch 0 against epoch {num_epochs - 1}."
        ),
    )
    latex_report_cheat_seg = LatexTableReport(
        columns=latex_columns,
        sort_by=["lm_scale"],
        epochs=(0, num_epochs - 1),
        caption=(
            "Chunked guided k-means with covariances, decoded on segmented features with "
            f"cheating segmentation: epoch 0 against epoch {num_epochs - 1}."
        ),
    )

    for lm_scale in parameters:
        exp_name = (
            f"chunked_lm-{lm_order}-{lm_scale}"
            f"_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}"
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

        exp_result = chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=input_data[input_data_key]["features"],
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=40 if not use_eow_phonemes else 79,
            initial_centroids=input_data[input_data_key]["cheating_centroids"],
            subsampling=subsampling,
            rasr_path=rasr_path,
            num_chunks=num_chunks,
            num_workers=num_workers,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
            initial_covs=input_data[input_data_key]["cheating_covs"],
            score_reference=guided_reference,
        )

        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json", exp_result.out_statistics
        )

        for epoch, score in sorted(exp_result.out_guided_scores.items()):
            tk.register_output(
                f"guided_kmeans/{exp_dir}/guided/{exp_name}_epoch-{epoch}_per", score.wer
            )

        # once per experiment, shared by both decodings and all epochs.
        # epoch_offset=1: chunked_clustering keys its merged statistics 1..num_epochs,
        # with key e being the pass that ran with centroids[e-1].
        statistics = clustering_statistics(exp_result.out_statistics, name=exp_name, epoch_offset=1)

        # Unchanged from the original config: each epoch's centroids become
        # available as soon as that epoch's job finishes, so these decodes start
        # running while later epochs are still being computed.
        for recog_epoch in range(num_epochs + 1):
            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=1.0,
                subsampling=subsampling,
                covs=exp_result.out_covs[recog_epoch],
            )

            res = decode_and_score(
                exp_name + f"_epoch-{recog_epoch}",
                "train-clean-100-dbg",
                decode_config,
                dataset_config,
                rasr_path=rasr_path,
                device="cpu",
            )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per
            )
            recog_results.append(res)
            latex_report.add_row(
                result=res,
                params={"lm_scale": lm_scale},
                epoch=recog_epoch,
                statistics=statistics,
                values=exp_result.guided_score_row(recog_epoch),
            )

        # recognition on segmented features
        for recog_epoch in range(num_epochs + 1):
            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["ls-100-segmented"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                recog_rasr_config=recognition_config_decode_cheating_seg,
                distance_scale=1.0,
                subsampling=subsampling,
                covs=exp_result.out_covs[recog_epoch],
            )

            res = decode_and_score(
                exp_name + "_cheat_seg" + f"_epoch-{recog_epoch}",
                "train-clean-100-dbg",
                decode_config,
                dataset_config,
                rasr_path=rasr_path,
                device="cpu",
            )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/recognition/{exp_name}_cheat_seg_epoch-{recog_epoch}_per", res.per
            )
            recog_results_cheat_seg.append(res)
            latex_report_cheat_seg.add_row(
                result=res,
                params={"lm_scale": lm_scale},
                epoch=recog_epoch,
                statistics=statistics,
                values=exp_result.guided_score_row(recog_epoch),
            )


    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{input_data_key}.txt",
        values=create_report(recog_results),
        required=True,
    )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_ls-100-seg.txt",
        values=create_report(recog_results_cheat_seg),
        required=True,
    )

    latex_report.register(
        f"guided_kmeans/{exp_dir}/recognition/report_{input_data_key}_first_last.tex"
    )
    latex_report_cheat_seg.register(
        f"guided_kmeans/{exp_dir}/recognition/report_ls-100-seg_first_last.tex"
    )


def py():
    run()
