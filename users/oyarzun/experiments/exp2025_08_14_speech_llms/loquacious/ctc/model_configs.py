from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

v2 = {
    "aux_loss_layers": (4, 8),
    "num_enc_layers": 12,
    "num_heads": 8,
    "model_dim": 512,
    "sampling_rate": 16_000,
    "specaug_start": (5_000, 15_000, 25_000),
    "use_rf_init": True,
    "feature_extraction_config": {
        "class": "LogMelFeatureExtractionV1",
        "win_size": 0.025,
        "hop_size": 0.01,
        "f_min": 60,
        "f_max": 7600,
        "min_amp": 1e-10,
        "num_filters": 80,
        "center": False,
    },
    "network_module": "conformer_ctc_v1",
}

v3 = dict_update_deep(
    v2,
    {
        "dropout": 0.2
    }
)

v4 = dict_update_deep(
    v2,
    {
        "model_dim": 256,
    }
)
