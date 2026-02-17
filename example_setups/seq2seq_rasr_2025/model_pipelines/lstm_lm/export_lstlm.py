from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.lstm_lm.i6_model_export import (
    export_scorer,
    export_state_initializer,
    export_state_updater,
)
from apptek_asr.lib.pytorch.networks import LstmLmModelV1Config
from i6_models.assemblies.lstm import LstmEncoderV1Config
from sisyphus import tk
from i6_core.returnn import PtCheckpoint
from i6_models.parts.lstm import LstmBlockV1Config

import torch

# When saving a general checkpoint, to be used for either inference or resuming training, you must save more than just the model’s state_dict.
# It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains.
# Other items that you may want to save are the epoch you left off on, the latest recorded training loss, external torch.nn.Embedding layers, etc.
# As a result, such a checkpoint is often 2~3 times larger than the model alone.


def py():
    #checkpoint = torch.load(ckpt_path, weights_only=True)
    #model.load_state_dict(checkpoint["model_state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #epoch = checkpoint["epoch"]
    #loss = checkpoint["loss"]

    ckpt_path = tk.Path("/nas/models/asr/hyoshimochi/setups/2025-09-lstm-lm-rasr/rasr_setup/lstm-lm.pt")
    # Checkpoint object pointing to a PyTorch checkpoint .pt file
    ckpt = PtCheckpoint(ckpt_path)

    #        "model_cfg": {
    #            "vocab_dim": 10240,
    #            "lstm_encoder_cfg": {
    #                "input_dim": 10240,
    #                "embed_dim": 512,
    #                "embed_dropout": 0.0,
    #                "lstm_layers_cfg": {
    #                    "input_dim": 512,
    #                    "hidden_dim": 2048,
    #                    "num_layers": 2,
    #                    "bias": True,
    #                    "dropout": 0.2,
    #                    "enforce_sorted": False,
    #                },
    #                "lstm_dropout": 0.2,
    #                "init_args": None,
    #            },
    #            "bottleneck_dim": None,
    #            "bottleneck_dropout": None,
    #            "use_nce": False,
    #            "init_args": {
    #                "init_args_w": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
    #                "init_args_b": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
    #            },
    #        }


    lstm_block_config_dict = {
        "input_dim": 512,
        "hidden_dim": 2048,
        "num_layers": 2,
        "bias": True,
        "dropout": 0.2,
        "enforce_sorted": False,
    }
    encoder_config_dict = {
        "input_dim": 10240,
        "embed_dim": 512,
        "embed_dropout": 0.0,
        "lstm_layers_cfg": LstmBlockV1Config(**lstm_block_config_dict),
        "lstm_dropout": 0.2,
        "init_args": None,
    }
    encoder_config = LstmEncoderV1Config(**encoder_config_dict)
    lm_config_dict = {
        "vocab_dim": 10240,
        "lstm_encoder_cfg": encoder_config,
        "bottleneck_dim": None,
        "bottleneck_dropout": None,
        "use_nce": False,
        "init_args": {
            "init_args_w": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
            "init_args_b": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
        },
    }
    lm_config = LstmLmModelV1Config(**lm_config_dict)

    initializer = export_state_initializer(lm_config, ckpt)
    scorer = export_scorer(lm_config, ckpt)
    updater = export_state_updater(lm_config, ckpt)

    tk.register_output("initializer", initializer)
    tk.register_output("scorer", scorer)
    tk.register_output("updater", updater)
