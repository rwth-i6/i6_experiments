import i6_core.rasr as rasr

def correct_rasr_FSA_bug(crp: rasr.CommonRasrParameters, apply_lemma_exit_penalty: bool =  True)-> rasr.CommonRasrParameters:
    """
    apply_lemma_exit_penalty: if set to False it, an additional lemma level exit penalty would be added to finial model FSA
                              this is not correct conceptually, since the word sequence is given and the training criterion
                              is constant with respect to this type of penalty
    """
    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    if apply_lemma_exit_penalty:
        transition_types = ["*", "silence"]
        if crp.acoustic_model_config.tdp.tying_type == "global-and-nonword":
            for nw in [0, 1]:
                transition_types.append(f"nonword-{nw}")
        for t in transition_types:
            crp.acoustic_model_config.tdp[t].exit = 0.0

    return crp