from sisyphus import tk
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.returnn.search import (
    SearchBPEtoWordsJob,
    ReturnnSearchFromFileJob,
    ReturnnComputeWERJob,
)


def asr_evaluation(
    config_file,
    corpus,
    returnn_root,
    returnn_python_exe,
    output_path,
    additional_parameters=None,
    segment_file=None,
):
    """
    Runs Returnn recognition with given Returnn file for the given corpus and calculates wer on the corpus
    :param config_file:
    :param corpus:
    :param returnn_root
    :param returnn_python_exe:
    :param output_path:
    :param additional_parameters:
    :param segment_file:
    :return:
    """
    cv_synth_ogg_job = BlissToOggZipJob(corpus, no_conversion=True)

    parameter_dict = {"ext_eval_zip": cv_synth_ogg_job.out_ogg_zip}
    parameter_dict.update(additional_parameters or {})

    recognition_job = ReturnnSearchFromFileJob(
        returnn_config_file=config_file,
        parameter_dict=parameter_dict,
        mem_rqmt=12,
        time_rqmt=1,
        output_mode="py",
    )
    recognition_job.add_alias(output_path + "/recognition")
    tk.register_output(output_path + "/asr_out", recognition_job.out_search_file)

    bpe_to_words = SearchBPEtoWordsJob(recognition_job.out_search_file)
    bpe_to_words.add_alias(output_path + "/bpe_to_words")
    tk.register_output(output_path + "/words_out", bpe_to_words.out_word_search_results)

    text_dict = CorpusToTextDictJob(
        corpus, segment_file=segment_file
    ).out_dictionary
    tk.register_output(output_path + "/reference_dict", text_dict)

    wer = ReturnnComputeWERJob(
        hypothesis=bpe_to_words.out_word_search_results,
        reference=text_dict,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    )
    wer.add_alias(output_path + "/wer_scoring")
    tk.register_output(output_path + "/WER", wer.out_wer)
    return wer.out_wer
