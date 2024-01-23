from i6_core.returnn.forward import ReturnnForwardJob

def search_single(
        prefix_name,
        returnn_config,
        checkpoint,
        recognition_dataset,
        recognition_reference,
        recognition_bliss_corpus,
        returnn_exe,
        returnn_root,
        mem_rqmt,
        time_rqmt,
        use_sclite=False,
        remove_label: Optional[Union[str, Set[str]]] = None,
):
    """
    Run search for a specific test dataset

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param returnn_standalone.data.datasets.dataset.GenericDataset recognition_dataset:
    :param Path recognition_reference: Path to a py-dict format reference file
    :param Path returnn_exe:
    :param Path returnn_root:
    :param recog_ext_pipeline: the search output is the raw beam search output, all beams.
        still need to select best, and also still need to maybe remove blank/EOS/whatever.
    :param remove_label: for SearchRemoveLabelJob
    """
    from i6_core.returnn.search import (
        ReturnnSearchJobV2,
        SearchBPEtoWordsJob,
        ReturnnComputeWERJob,
    )

    search_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        # hdf_outputs=["recognition.txt"],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        # device="cpu",
        # cpu_rqmt=8,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
    )

    # recognition = search_job.out_hdf_files[f"recognition.txt"]

    # search_job = ReturnnSearchJobV2(
    #     search_data=recognition_dataset.as_returnn_opts(),
    #     model_checkpoint=checkpoint,
    #     returnn_config=returnn_config,
    #     log_verbosity=5,
    #     mem_rqmt=mem_rqmt,
    #     time_rqmt=time_rqmt,
    #     returnn_python_exe=returnn_exe,
    #     returnn_root=returnn_root,
    #     output_gzip=recog_ext_pipeline,
    # )
    search_job.add_alias(prefix_name + "/search_job")
    search_bpe = search_job.out_search_file

    # If we need to remove a label, do it early, before the SearchBeamJoinScoresJob,
    # otherwise SearchBeamJoinScoresJob would not have any effect.
    if remove_label:
        from i6_core.returnn.search import SearchRemoveLabelJob

        search_bpe = SearchRemoveLabelJob(search_bpe, remove_label=remove_label, output_gzip=True).out_search_results

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