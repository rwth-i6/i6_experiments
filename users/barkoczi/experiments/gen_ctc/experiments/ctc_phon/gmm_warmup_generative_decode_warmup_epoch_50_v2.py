from .gmm_warmup_generative import _run_gmm_warmup_experiment


def eow_phon_ls960_1023_gmm_warmup_generative_nce_decode_warmup_epoch_50_v2():
    return _run_gmm_warmup_experiment(
        decode_stage="warmup",
        decode_network_module=(
            "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first_v2"
        ),
        decode_checkpoint=50,
        decode_name="decode_ep050_hard_targets_masked_prior_generative_posterior_v2",
    )


py = eow_phon_ls960_1023_gmm_warmup_generative_nce_decode_warmup_epoch_50_v2
