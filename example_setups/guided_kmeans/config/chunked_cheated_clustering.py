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
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import LatexTableReport, clustering_statistics

exp_dir = "chunked_cheating_lm"

input_data = {
    "train-clean-100-dbg": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/filtered_features_train-clean-100-dbg.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/train-clean-100-dbg/centroids.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/train-clean-100-dbg-segments.txt"),
    },
    "ls-100": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/wav2vec2_ls100h.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/ls100h-segments.txt"),
    },
    "ls-100-segmented": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/segmented_features_wav2vec2_ls100h.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/ls100h-segments.txt"),
    },
}

rasr_path = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")


def run():
    corpus_file = phoneme_corpus(setup_corpus("train-clean-100"))

    use_eow_phonemes = False
    use_pruning = False
    num_epochs = 10
    input_data_key = "ls-100"

    # 28234 seqs at ~17 s of search each: 30 chunks x 8 workers puts an epoch
    # at roughly half an hour, against ~9 h for the single-node pipeline.
    num_chunks = 30
    num_workers = 8

    subsampling, lm_scale, loop_probability, silence_loop_probability = None, 1.0, 0.0, 0.0
    lm_scale = 1.0

    parameters = [
        # transition scale, speech loop, silence loop
        (scale, lp, lp)
        for scale in [1000.0, 2000.0, 5000.0, 10000.0]
        for lp in [0.1, 0.3, 0.5, 0.7]
    ]

    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    recognition_config_decode = create_recog_rasr_config(
        lm_scale=10000.0,
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

    recog_results = []

    # The LM here is the cheating-segment LM, so the swept scale is the transition
    # scale (create_recog_rasr_config falls back to lm_scale when transition_scale is
    # None) - hence the parameter key, which drives the column header.
    latex_report = LatexTableReport(
        columns=[
            "transition_scale", "loop_probability",
            "epoch", "per", "del", "ins", "sub",
            "l1", "am_score", "lm_score",
        ],
        sort_by=["transition_scale", "loop_probability"],
        epochs=(0, num_epochs - 1),
        caption=(
            "Chunked guided k-means, cheating LM, LibriSpeech 100h: "
            f"epoch 0 against epoch {num_epochs - 1}."
        ),
    )

    for lm_scale, loop_probability, silence_loop_probability in parameters:

        exp_name = (
            f"chunked_sub-{subsampling}_lm-cheating_transition-{lm_scale}"
            f"_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}"
        )

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=None,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            use_tree_search=False,
            max_beam_size=20000 if use_pruning else None,
            score_threshold=10000.0 if use_pruning else None,
            lm_order=2,
            use_eow_phonemes=use_eow_phonemes,
            cheating=True,
            corpus=corpus_file,
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
        )

        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json", exp_result.out_statistics
        )

        # once per experiment, not per epoch, so all epochs share the same jobs.
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
            )

            res = decode_and_score(
                exp_name + f"_epoch-{recog_epoch}", "train-clean-100-dbg",
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
                params={"transition_scale": lm_scale, "loop_probability": loop_probability},
                epoch=recog_epoch,
                statistics=statistics,
            )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{input_data_key}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(
        f"guided_kmeans/{exp_dir}/recognition/report_{input_data_key}_first_last.tex"
    )


def py():
    run()
