from __future__ import annotations
from typing import Optional, Union, Set

from sisyphus import tk
import os.path

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.training import GetBestTFCheckpointJob
from i6_core.returnn.training import AverageTFCheckpointsJob

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.default_tools import SCTK_BINARY_PATH

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    build_test_dataset,
)



def process_search_out(
        # prefix_name,
        # returnn_config,
        # checkpoint,
        # recognition_dataset,
        # recognition_reference,
        # recognition_bliss_corpus,
        # returnn_exe,
        # returnn_root,
        # mem_rqmt,
        # time_rqmt,
        use_sclite=True,
        recog_ext_pipeline=False,
        # remove_label: Optional[Union[str, Set[str]]] = None,
):
    def get_test_dataset_tuples(bpe_size):
        test_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                testset,
                use_raw_features=True,
                bpe_size=bpe_size,
            )
        return test_dataset_tuples

    from i6_core.returnn.search import (
        SearchBPEtoWordsJob,
        ReturnnComputeWERJob,
    )
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]
    exp_name = "rf_att_test_non_sorted"
    prefix_name = os.path.join(prefix_name, exp_name)

    test_set = "dev-other"
    bpe_size = 10000

    test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)
    # recognition_dataset = test_dataset_tuples[test_set][0],
    recognition_reference = test_dataset_tuples[test_set][1],
    recognition_bliss_corpus = test_dataset_tuples[test_set][2],

    # wierd bug workaround, to get rid of tuples with one entry
    recognition_reference = recognition_reference[0]
    recognition_bliss_corpus = recognition_bliss_corpus[0]

    search_bpe = tk.Path(
        "/u/luca.gaudino/debug/moh_att_import/search_out_full.py"#, hash_overwrite="DEBUG_SEARCH_OUTPUT"
    )

    remove_label = {"<s>", "<blank>"}  # blanks are removed in the network

    if remove_label:
        from i6_core.returnn.search import SearchRemoveLabelJob

        search_bpe = SearchRemoveLabelJob(search_bpe, remove_label=remove_label, output_gzip=True).out_search_results

    # if recog_ext_pipeline:
    #     # TODO check if SearchBeamJoinScoresJob makes sense.
    #     #   results are inconsistent.
    #     #   one potential explanation: the amount of merges per hyp is uneven, and maybe bad hyps have actual more
    #     #      entries in the beam due to confusions. then their sum will win over better hyps.
    #     #   another potential explanation: logsumexp is not correct with length norm.
    #     #      (btw, with length norm, it's not trivial to correct, as it uses the factor from the whole batch.)
    #     #   thus, we do not use it for now.
    #     #   if we would use it, only if there was some remove_label.
    #
    #     from i6_core.returnn.search import SearchTakeBestJob
    #
    #     search_bpe = SearchTakeBestJob(search_bpe, output_gzip=True).out_best_search_results

    search_words = SearchBPEtoWordsJob(search_bpe, output_gzip=recog_ext_pipeline).out_word_search_results

    if use_sclite:
        from i6_core.returnn.search import SearchWordsToCTMJob
        from i6_core.corpus.convert import CorpusToStmJob
        from i6_core.recognition.scoring import ScliteJob

        search_ctm = SearchWordsToCTMJob(
            recog_words_file=search_words,
            bliss_corpus=recognition_bliss_corpus,
        ).out_ctm_file

        stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

        sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH)
        tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
        tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    wer = ReturnnComputeWERJob(search_words, recognition_reference)

    tk.register_output(prefix_name + "/search_out_words.py", search_words)
    tk.register_output(prefix_name + "/wer", wer.out_wer)
    return wer.out_wer