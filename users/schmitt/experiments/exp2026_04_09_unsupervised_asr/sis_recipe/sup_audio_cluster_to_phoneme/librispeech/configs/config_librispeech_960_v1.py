import copy
from typing import List

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from sisyphus import tk


def get_keep_epochs(num_epochs: int) -> List[int]:
    if num_epochs == 1_000:
        return [250, 500, 750, 1_000]


base_num_epochs = 1_000
num_gpus = 2

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering=None,
)
train_data = build_training_datasets(settings=settings)
test_data_dict = build_test_datasets()

base_config = {
    "__network_module": "definitions.conformer_aed_discrete_shared_v1.Model",
    "__train_step_module": "train_steps.aed_denoising_discrete.train_step",
    "__baseline_alias": "v1",
    "__forward_step_module": "recognition.discrete_audio_aed.forward_step.forward_step",
    "__callback_module": "recognition.discrete_audio_aed.callback.RecognitionToTextDictCallback",
    "__rasr_forward_step_module": "recognition.discrete_audio_aed.rasr.forward_step.forward_step_v1",
    "__rasr_callback_module": "recognition.discrete_audio_aed.rasr.callback.RecognitionToTextDictCallback",
    "train_rqmt": {
        "cpu_rqmt": 6,
    },
    "general": {
        "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        "behavior_version": 25,
        "default_data_key": "data",
        "default_target_key": "target",
    },
    "training": {
        "__num_gpus": num_gpus,
        "__num_epochs": base_num_epochs,
        "__lr_opts": {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": [
                0,
                0.45 * base_num_epochs,
                0.9 * base_num_epochs,
                base_num_epochs,
            ],
            "piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
        },
        "grad_scaler": None,
        # "torch_amp": "bfloat16",
        "batch_size": 15_000,
        **optimizer_configs.v1,
        # "max_seq_length": {"audio": 19.5 * sampling_rate},  # 19.5 seconds
        "max_seqs": 200,
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,
    },
    "recog": {
        "batch_size": 15_000,
    },
    "model_args": {
        "text_aux_loss_layers": (),
        "audio_aux_loss_layers": (),
        "num_enc_layers": 3,
        "num_text_dec_layers": 3,
        "num_audio_dec_layers": 3,
        "num_heads": 8,
        "model_dim": 512,
        "share_decoder": True,
        "text_out_dim": train_data.datastreams["target"].vocab_size,
        "audio_out_dim": train_data.datastreams["data"].vocab_size,
    },
    "train_args": {
        "aux_loss_scales": (),
        "ce_loss_scale": 1.0,
        "masked_ce_loss_scale": 0.0,
        "masking_opts": {
            "mask_prob": 0.0,
            "min_span": 0,  # 1
            "max_span": 0,  # 3
        },
    },
}


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        # run the shared-encoder PCA analysis (models/analysis/encoder_state_pca) on the final
        # checkpoint, plotting the audio vs. text encoder states of a few dev-other sequences.
        # also summarize avg pairwise cosine similarities in the original (pre-PCA) feature space.
        analysis_opts={
            "checkpoints": [base_num_epochs],
            "max_plotted_seqs": 20,
            "cosine_similarity_summary": True,
        },
    )
