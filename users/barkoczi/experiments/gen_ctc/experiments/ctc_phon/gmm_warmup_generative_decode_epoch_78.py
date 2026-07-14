from .gmm_warmup_generative import _run_gmm_warmup_experiment


def eow_phon_ls960_1023_gmm_warmup_generative_nce_decode_epoch_78():
    return _run_gmm_warmup_experiment(
        decode_checkpoint=78,
        decode_name="decode_ep078_pre_collapse_generative_posterior_v2",
    )


py = eow_phon_ls960_1023_gmm_warmup_generative_nce_decode_epoch_78
