from typing import Optional

from sisyphus import Path, tk

from i6_experiments.users.schmitt.datasets.utils.phonemize import PhonemizeTextDataJob

from ..default_tools import get_fairseq_root, get_lid_model, get_fasttext_python_exe


def get_phonemized_data(
    dataset_name: str,
    corpus_name: str,
    text_file: Path,
    language: str = "en",
    sil_prob: float = 0.25,
    lexicon_file: Optional[Path] = None,
    phoneme_file: Optional[Path] = None,
    seq_tag_file: Optional[Path] = None,
):
    prepare_text_job_training = PhonemizeTextDataJob(
        text_file=text_file,
        fairseq_root=get_fairseq_root(),
        python_exe=get_fasttext_python_exe(),
        lid_path=get_lid_model(),
        language=language,
        sil_prob=sil_prob,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_file,
        seq_tag_file=seq_tag_file,
    )
    phoneme_file = prepare_text_job_training.out_phoneme_text
    tk.register_output(f"data/{corpus_name}/text/phonemized/{dataset_name}.txt", phoneme_file)
