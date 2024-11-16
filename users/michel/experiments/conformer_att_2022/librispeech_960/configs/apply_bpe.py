from apptek_asr.data.common import CheckDuplicatesAcrossCorporaJob

from i6_core.corpus import CorpusToTxtJob
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict

from sisyphus import tk

def py():
    bpe_codes = tk.Path("/u/michel/setups/language_modelling/tedlium/data/bpe.codes")
    bpe_vocab = tk.Path("/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.dummy_count.vocab")
    subword_nmt_repo = CloneGitRepositoryJob(
        url="https://github.com/rsennrich/subword-nmt.git",
        commit="810ee1487a753870ebf90d91ccdb789158268d9f"
    ).out_repository

    txt_dict = get_text_data_dict()
    txt_dict["dev"] = CorpusToTxtJob(
        get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")["dev"]
    ).out_txt
    txt_dict["test"] = CorpusToTxtJob(
        get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")["test"]
    ).out_txt
    del txt_dict["background-data"]  # combination of all data, not needed here

    for name, text_file in txt_dict.items():
        applier = ApplyBPEToTextJob(
            text_file=text_file,
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab,
            subword_nmt_repo=subword_nmt_repo,
            gzip_output=True,
            mini_task=False,
        )
        tk.register_output(f"datasets/{name}_bpe.gz", applier.out_bpe_text)

    # check for overlap with the dev data
    for ngram in [0, 5, 7, 12]:
        for name, text_file in txt_dict.items():
            if name == "dev":
                continue
            overlapp_checker = CheckDuplicatesAcrossCorporaJob(
                input_sets=[txt_dict["dev"], text_file],
                ngram=ngram,
            )
            tk.register_output(f"datasets/overlapp_dev-{name}_{ngram}.report", overlapp_checker.out_report)