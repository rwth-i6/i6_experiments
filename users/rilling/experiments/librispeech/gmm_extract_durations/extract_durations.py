from sisyphus import tk
from i6_experiments.users.hilmes.tools.tts.extract_alignment import ExtractDurationsFromRASRAlignmentJob
from i6_experiments.users.rossenbach.datasets.librispeech import get_ls_train_clean_100_tts_silencepreprocessed
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict


def extract_gmm_durations():
    bliss_corpus = get_ls_train_clean_100_tts_silencepreprocessed()
    ls_bliss = get_bliss_corpus_dict(audio_format="ogg")["train-clean-100"]

    extract_durations = ExtractDurationsFromRASRAlignmentJob(
        rasr_alignment=tk.Path(
            "/work/asr3/rossenbach/rilling/gmm_durations/alignment.cache.bundle"
        ),
        rasr_allophones=tk.Path(
            "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/lexicon/allophones/StoreAllophonesJob.boJyUrd9Bd89/output/allophones"
        ),
        bliss_corpus=ls_bliss,
    )

    extract_durations.add_alias("experiments/librispeech/gmm_durations/duration_extraction")
    tk.register_output(
        "experiments/librispeech/gmm_durations/duration_extraction/durations.hdf", extract_durations.out_durations_hdf
    )
