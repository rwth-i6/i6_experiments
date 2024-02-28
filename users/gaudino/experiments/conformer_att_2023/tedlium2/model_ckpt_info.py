from i6_core.returnn.training import Checkpoint
import sisyphus.toolkit as tk

# models paths
models = {
    "model_baseline": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.yB4JK4GDCxWG/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt",
    },
    # ctcScale models
    "model_ctc0.43_att1.0": {  # ctcScale 0.3
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.nCrQhRfqIRiZ/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.Yonvnwljktqh/output/prior.txt",
    },
    "model_ctc0.25_att1.0": {  # ctcScale 0.2
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.CknpN55pjOHo/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.MhkU9CYwTQy3/output/prior.txt",
    },
    "model_ctc0.2_att1.0": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.ro9g9W6DBJpW/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.gJiuTmxRwMVu/output/prior.txt",
    },
    # 1-y models
    "model_ctc0.9_att0.1": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.QVqAmKtGDWq5/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZuisZnA3H0Ke/output/prior.txt",
    },
    "model_ctc0.8_att0.2": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.shrNTbvD9wG6/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZbRLKK1N3AYy/output/prior.txt",
    },
    "model_ctc0.7_att0.3": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.NLTPqijTnIR8/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.udHpTjKj21rl/output/prior.txt",
    },
    "model_ctc0.6_att0.4": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.EbyFcDSWuaNS/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.BQhz6eKggFTX/output/prior.txt",
    },
    "model_ctc0.5_att0.5": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.AJ6o9pWzmXn3/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.zLtRxqFwjciQ/output/prior.txt",
    },
    "model_ctc0.4_att0.6": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.C0R1EFZxuEcM/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.YoWbUqoGxdX6/output/prior.txt",
    },
    "model_ctc0.3_att0.7": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.jGxeW6yzeoG7/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ypsBrM65Uj1k/output/prior.txt",
    },
    "model_ctc0.2_att0.8": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.6qWPnvXHalfJ/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.mHUoJaQFZ27b/output/prior.txt",
    },
    "model_ctc0.1_att0.9": {  # pre 4
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.MEtpESN5M4oD/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.I4aVOIk1CXmt/output/prior.txt",
    },
    "model_ctc0.001_att0.999": {  # pre 4
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.eEEAEAZQiFvO/output/model/average.index"
            )
        ),
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.mhjgjO6IUEPB/output/prior.txt",
    },
    # ctc loss different layers
    "model_ctc0.3_att0.7_lay6": {
        "ckpt": Checkpoint(
            tk.Path(
                "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/training/AverageTFCheckpointsJob.mlEl83XV5YX9/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.X7XyfXQgD3xG/output/prior.txt",
        "enc_layer_w_ctc": 6,
    },
    "model_ctc0.3_att0.7_lay8": {
        "ckpt": Checkpoint(
            tk.Path(
                "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/training/AverageTFCheckpointsJob.BJm3qbEaW5Tx/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.eP7zoAFYv3og/output/prior.txt",
        "enc_layer_w_ctc": 8,
    },
    "model_ctc0.3_att0.7_lay10": {
        "ckpt": Checkpoint(
            tk.Path(
                "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/training/AverageTFCheckpointsJob.kXhiucifOrAt/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.xE5gRKwcIqiU/output/prior.txt",
        "enc_layer_w_ctc": 10,
    },
    "model_ctc1.0_att1.0_lay6": {
        "ckpt": Checkpoint(
            tk.Path(
                "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/training/AverageTFCheckpointsJob.blMBlPQmI98T/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.n7bJXRdAxMzQ/output/prior.txt",
        "enc_layer_w_ctc": 6,
    },
    "model_ctc1.0_att1.0_lay8": {
        "ckpt": Checkpoint(
            tk.Path(
                "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/training/AverageTFCheckpointsJob.u6FZCXVWY47j/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.vDNVuXdu71fC/output/prior.txt",
        "enc_layer_w_ctc": 8,
    },
    "model_ctc1.0_att1.0_lay10": {
        "ckpt": Checkpoint(
            tk.Path(
                "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/training/AverageTFCheckpointsJob.Pxff6AKX9mkH/output/model/average.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.sCesAfPOg838/output/prior.txt",
        "enc_layer_w_ctc": 10,
    },
    # att only
    "model_att_only_currL": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.io6cKw6ETnHp/output/model/average.index"
            )
        ),
        "prior": "",
        "no_ctc": True,
    },
    "model_att_only_adjSpec": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.9f6nlw1UOxVO/output/model/average.index"
            )
        ),
        "prior": "",
        "no_ctc": True,
    },
    # ctc only
    "model_ctc_only": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400.index"
            )
        ),  # last
        "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.Ow9jQN0VEdlo/output/prior.txt",
        # how is this computed?
        "ctc_only": True,
    },
}