"""
See :class:`WhisperRecognitionJob`.

https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://huggingface.co/nyrahealth/CrisperWhisper
https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh
https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_eval.py
https://github.com/nyrahealth/CrisperWhisper

"""

from typing import Optional
import functools
from sisyphus import tk
from .whisper import WhisperRecognitionJob


def crisper_whisper_recog_score_wer(
    *, dataset_dir: Optional[tk.Path] = None, dataset_name: str = "tedlium", dataset_split: str = "test"
):
    """
    Do recognition with CrisperWhisper, mostly consistent to OpenASRLeaderboard.
    See :class:`WhisperRecognitionJob` below.
    """
    from i6_experiments.users.zeyer.datasets.huggingface.extract_text import ExtractTextFromHuggingFaceDatasetJob
    from i6_experiments.users.zeyer.datasets.huggingface.open_asr_leaderboard import (
        text_dict_normalize_file,
        download_esb_datasets_test_only_sorted,
    )
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import sclite_score_hyps_to_ref

    model_dir = download_crisper_whisper_model()
    if dataset_dir is None:
        dataset_dir = download_esb_datasets_test_only_sorted()

    recog_job = WhisperRecognitionJob(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        # TODO batch_size>1 still seems broken... https://github.com/huggingface/open_asr_leaderboard/issues/68
        batch_size=1,
    )
    tk.register_output(f"crisper_whisper.{dataset_name}.{dataset_split}.recog.txt.py.gz", recog_job.out_recog)
    ref_text_job = ExtractTextFromHuggingFaceDatasetJob(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    tk.register_output(f"{dataset_name}.{dataset_split}.ref.txt.py.gz", ref_text_job.out_text)
    # Should get: 2.89% WER with Phi4MI on Tedlium (test).
    # Without normalization: 6.25%
    # With hyp normalization: 11.41%
    # With ref+hyp normalization: 2.88%
    tk.register_output(
        f"crisper_whisper.{dataset_name}.{dataset_split}.wer.txt",
        sclite_score_hyps_to_ref(
            text_dict_normalize_file(recog_job.out_recog), ref_text_dict=text_dict_normalize_file(ref_text_job.out_text)
        ).main_measure_value,
    )


@functools.cache
def download_crisper_whisper_model() -> tk.Path:
    from .huggingface import DownloadHuggingFaceRepoJobV2

    dl_model = DownloadHuggingFaceRepoJobV2(repo_id="nyrahealth/CrisperWhisper", repo_type="model")
    tk.register_output("crisper-whisper-model", dl_model.out_hub_cache_dir)
    return dl_model.out_hub_cache_dir
