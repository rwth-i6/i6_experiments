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
    # gauss window
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTCdefault_last/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_control/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_control_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_gradClip5.0/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std0.1/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std0.1_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std0.5/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std0.5_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std10.0/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std10.0_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std2.0/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_std2.0_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_window10/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_window10_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_window20/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_window20_no_enc/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_window50/
    # base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12_onlyCTC_gaussWeights_window50_no_enc/
    "model_ctc_only_gauss1.0_win5": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.4dFO6QJQ4h7x/output/models/epoch.400.index"
            )
        ),
       "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ky9twIeNdnSy/output/prior.txt" ,
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 1.0,
        "use_enc": True,
    },
    "model_ctc_only_win1": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.0zYdcDMmqV69/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.Vaem3d9qn6nk/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 1,
        "gauss_std": 0.0,
        "use_enc": True,
    },
    "model_ctc_only_gauss0.1_win5": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.fyDjgnJKuyAi/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.SKiPIF4FFauQ/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 0.1,
        "use_enc": True,
    },
    "model_ctc_only_gauss0.5_win5": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.1DjID8GO0eX8/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.5FcoTsOlzOQ9/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 0.5,
        "use_enc": True,
    },
    "model_ctc_only_gauss2.0_win5": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.YiunxgR7jbIP/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.t60vT80MfG0Q/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 2.0,
        "use_enc": True,
    },
    "model_ctc_only_gauss10.0_win5": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.8gclCL9UPoC5/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.oVCdW0hUIWqa/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 10.0,
        "use_enc": True,
    },
    "model_ctc_only_gauss1.0_win10": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.DNbLpWOSkVNm/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.FXT7iFlXlmRG/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 10,
        "gauss_std": 1.0,
        "use_enc": True,
    },
    "model_ctc_only_gauss1.0_win20": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.NaSSUF2CCQjC/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.oWmJBoaKTPc8/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 20,
        "gauss_std": 1.0,
        "use_enc": True,
    },
    "model_ctc_only_gauss1.0_win50": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.xtrvlmco4Kz6/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.06YYtz8OyO6m/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 20,
        "gauss_std": 1.0,
        "use_enc": True,
    },
    # no enc so only c_t
    "model_ctc_only_gauss1.0_win5_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.4v01A22bWufz/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.dqQTti5anInd/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 1.0,
        "use_enc": False,
    },
    "model_ctc_only_win1_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.Jr8vnK7r1Ps7/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.eM5XKItcbCfs/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 1,
        "gauss_std": 0.0,
        "use_enc": False,
    },
    "model_ctc_only_gauss0.1_win5_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.mCDQso8aoCRC/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.KA1zej08CKHN/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 0.1,
        "use_enc": False,
    },
    "model_ctc_only_gauss0.5_win5_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.RXdcWniIsEjb/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.tiXWpx4cwZp2/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 0.5,
        "use_enc": False,
    },
    "model_ctc_only_gauss2.0_win5_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.nb0a9524rtZk/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.LE81vHzY0zMV/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 2.0,
        "use_enc": False,
    },
    "model_ctc_only_gauss10.0_win5_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.NJr0phivEKNl/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.EAqoXhWbk8ip/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 5,
        "gauss_std": 10.0,
        "use_enc": False,
    },
    "model_ctc_only_gauss1.0_win10_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.RtBg9xA505e2/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.3bPXQh7bvC2z/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 10,
        "gauss_std": 1.0,
        "use_enc": False,
    },
    "model_ctc_only_gauss1.0_win20_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.miqhweaUVtKT/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.oqpkq3LDUqAd/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 20,
        "gauss_std": 1.0,
        "use_enc": False,
    },
    "model_ctc_only_gauss1.0_win50_noEnc": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/training/ReturnnTrainingJob.fgGQTQyT3JB5/output/models/epoch.400.index"
            )
        ),
        "prior": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.Jd0pNVRmw2yp/output/prior.txt",
        "ctc_only": True,
        "gauss_window": 50,
        "gauss_std": 1.0,
        "use_enc": False,
    },
}