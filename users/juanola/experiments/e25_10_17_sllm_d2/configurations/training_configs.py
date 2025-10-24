training_configs = {
    "v1": {
        "aux_loss_layers": (4, 8),
        "num_enc_layers": 12,
        "num_heads": 8,
        "encoder_dim": 512,
        "vocab_size": 10_240,
        "sampling_rate": 16_000,
        "specaug_start": (5_000, 15_000, 25_000),
        "use_rf_init": True,
        "bos_idx": 1,
        "eos_idx": 0,
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
    },
    "v2": {}  # ...
}