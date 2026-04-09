from dataclasses import asdict
from sisyphus import tk

from ...config import get_feature_variance_config
from ...data.common import DatasetSettings
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import compute_feature_variance
from ...data.phon import build_eow_phon_training_datasets


def wav2vec2_hf_feature_variance_train():
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_wav2vec2_hf_feature_variance_train"

    settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )

    from ...pytorch_networks.ctc.wav2vec2_hf_ctc_v1_cfg import ModelConfig

    model_config = ModelConfig(
        label_target_size=79,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False,
        aux_ctc_loss_layers=[3, 6, 9, -1],
        aux_ctc_loss_scales=[0.3, 0.3, 0.3, 1.0],
    )

    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=settings,
    )

    network_module = "ctc.wav2vec2_hf_ctc_v1"
    net_args = {"model_config_dict": asdict(model_config)}
    returnn_config = get_feature_variance_config(
        forward_dataset=train_data.prior,
        network_module=network_module,
        config={"batch_size": 500 * 16000},
        net_args=net_args,
        forward_args={},
        debug=False,
    )
    checkpoint = tk.Path(
        "/work/asr4/zyang/mini/work/i6_core/returnn/training/ReturnnTrainingJob.U3AWP7d5sCOC/output/models/epoch.200.pt"
    )
    _, job = compute_feature_variance(
        prefix_name=prefix_name,
        returnn_config=returnn_config,
        checkpoint=checkpoint,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=[f"variance_{layer}.txt" for layer in model_config.aux_ctc_loss_layers],
        use_gpu=True,
    )
    job.rqmt["gpu_mem"] = 24


py = wav2vec2_hf_feature_variance_train
