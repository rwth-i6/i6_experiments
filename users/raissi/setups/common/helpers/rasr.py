
def build_crp(
        self,
        am_args,
        corpus_object,
        concurrent,
        segment_path,
        lexicon_args,
        cart_tree_path=None,
        allophone_file=None,
        lm_args=None,
):
    """
    constructs and returns a CommonRasrParameters from the given settings and files
    """
    crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(crp)
    crp.acoustic_model_config = am.acoustic_model_config(**am_args)
    rasr.crp_set_corpus(crp, corpus_object)
    crp.concurrent = concurrent
    crp.segment_path = segment_path

    crp.lexicon_config = rasr.RasrConfig()
    crp.lexicon_config.file = lexicon_args["filename"]
    crp.lexicon_config.normalize_pronunciation = lexicon_args[
        "normalize_pronunciation"
    ]

    if "add_all" in lexicon_args:
        crp.acoustic_model_config.allophones.add_all = lexicon_args["add_all"]
        crp.acoustic_model_config.allophones.add_from_lexicon = not lexicon_args["add_all"]


    if lm_args is not None:
        crp.language_model_config = rasr.RasrConfig()
        crp.language_model_config.type = lm_args["type"]
        crp.language_model_config.file = lm_args["filename"]
        crp.language_model_config.scale = lm_args["scale"]

    if allophone_file is not None:
        crp.acoustic_model_config.allophones.add_from_file = allophone_file

    self.crp = crp


