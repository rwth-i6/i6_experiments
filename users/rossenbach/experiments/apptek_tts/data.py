from sisyphus import tk

from i6_core.corpus.convert import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
from i6_core.text.processing import PipelineJob

from i6_experiments.users.rossenbach.datasets.librispeech import get_ls_train_clean_100_tts_silencepreprocessed

from apptek_tts.corpus.conversion import BlissToCsvJob

def export_metadata():
    ls100_spp_bliss = get_ls_train_clean_100_tts_silencepreprocessed().corpus_file
    ls100_spp_text = CorpusToTxtJob(ls100_spp_bliss).out_txt
    PipelineJob(
        ls100_spp_text,
        ["tr '[:upper:]' '[:lower:]'"],
        zip_output=False,
        mini_task=True,
    )
    ls100_spp_bliss_lc = CorpusReplaceOrthFromTxtJob(ls100_spp_bliss, ls100_spp_text).out_corpus
    csv = BlissToCsvJob(ls100_spp_bliss_lc, language="en-us").out_csv
    tk.register_output("ls100_spp.metadata.csv", csv)

