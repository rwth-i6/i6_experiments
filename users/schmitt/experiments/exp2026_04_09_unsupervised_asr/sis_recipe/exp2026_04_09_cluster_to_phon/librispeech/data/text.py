from sisyphus import tk

from i6_core.tools.download import DownloadJob
from i6_core.text.processing import HeadJob

from ....data.text import get_phonemized_data

FILTERED_LM_DATA = DownloadJob(
    url="https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt",
    target_filename="lm_corpus_minus_librivox",
).out_file
tk.register_output("data/librispeech/lm/lbs_lm_minus_librivox.raw", FILTERED_LM_DATA)


def get_phonemized_filtered_lm_data():
    get_phonemized_data(
        dataset_name="lm_minus_librivox",
        corpus_name="librispeech",
        text_file=HeadJob(
            FILTERED_LM_DATA,
            num_lines=10_000,
            zip_output=False,
        ).out,
    )
