from .gmm_warmup_generative import _run_gmm_warmup_experiment


def eow_phon_ls960_1023_gmm_posterior_prior_warmup_generative_nce():
    return _run_gmm_warmup_experiment(
        prefix_name=(
            "users/barkoczi/experiments/gen_ctc/"
            "ls960_ctc_eow_phon_gmm_posterior_prior_warmup_generative_nce"
        ),
        gmm_network_module=(
            "ctc.conformer_1023."
            "i6modelsV1_VGG4LayerActFrontendV1_v6_generative_gmm_posterior_prior"
        ),
        warmup_name="gmm-hard-targets-posterior-prior-ce",
        handoff_name="gmm-posterior-prior-ce",
    )


py = eow_phon_ls960_1023_gmm_posterior_prior_warmup_generative_nce
