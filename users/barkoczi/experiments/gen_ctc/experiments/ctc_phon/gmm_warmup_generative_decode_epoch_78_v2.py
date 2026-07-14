from .gmm_warmup_generative import _run_gmm_warmup_experiment


def eow_phon_ls960_1023_gmm_warmup_generative_nce_decode_epoch_78_v2():
    return _run_gmm_warmup_experiment(
        decode_stage="handoff",
        decode_network_module=(
            "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first_v2"
        ),
        decode_checkpoint=78,
        decode_name="decode_ep078_pre_collapse_masked_prior_generative_posterior_v2",
    )


py = eow_phon_ls960_1023_gmm_warmup_generative_nce_decode_epoch_78_v2
