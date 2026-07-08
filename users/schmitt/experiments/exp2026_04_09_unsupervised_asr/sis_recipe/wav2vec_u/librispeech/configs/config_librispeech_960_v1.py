import copy
from typing import List

from i6_experiments.common.setups.serialization import Import, PartialImport

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from ... import __setup_base_name__
from .....models.recognition.wav2vec_u.decoder_config import DecoderConfig


def get_keep_epochs(num_epochs: int) -> List[int]:
    if num_epochs == 15:
        return [5, 10, 15]

settings = DatasetSettings(
    train_partition_epoch=1,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(settings=settings)
test_data_dict = build_test_datasets()

num_gpus = 1

# fairseq wav2vec-U trains for max_update=150000 optimizer steps (config/gan/w2vu.yaml), not epochs.
# 1 sub-epoch with partition_epoch 1 takes approximately 30 minutes and does 10k steps
# -> we want to do 150k steps like the wav2vec-u paper -> 10k * 15 = 150k
base_num_epochs = 15

# fairseq w2vu.yaml GAN hyperparameters (model: wav2vec_u). Our phoneme vocab has no dedicated pad, so
# we append one extra output class as the pad index (real phonemes 0..V-1 are never masked as pad).
_phon_vocab_size = train_data.datastreams["phon_indices"].vocab_size

base_config = {
    "__network_module": "definitions.wav2vec_u.Model",
    "__train_step_module": "train_steps.wav2vec_u.train_step",
    "__baseline_alias": "v1",
    # GAN recognition: audio features -> phoneme argmax (no beam search). Reuses the AED callback +
    # sclite scoring. Only used when recognition is enabled (the GAN-only run uses skip_eval=True).
    "__forward_step_module": "recognition.wav2vec_u.forward_step.forward_step",
    "__callback_module": "recognition.discrete_audio_aed.callback.RecognitionToTextDictCallback",
    "train_rqmt": {
        "cpu_rqmt": 24,  # 48gb nodes have 96 gpus in total
        "mem_rqmt": 64,
        "gpu_mem": 48,
        "time_rqmt": 24,
    },
    "general": {
        "torch_dataloader_opts": {"num_workers": 6},  # fairseq dataset.num_workers: 6
        "behavior_version": 25,
        "default_data_key": "data",  # 512-dim speech features (fairseq task.data)
        "default_target_key": "phon_indices",  # unpaired phoneme text (fairseq task.text_data)
    },
    "training": {
        "__num_gpus": num_gpus,
        "__num_epochs": base_num_epochs,
        "stop_on_nonfinite_train_score": False,
        # fairseq uses a *fixed* LR per group (no warmup / decay). We keep the effective LR constant at
        # the discriminator LR (5e-4) and scale the generator group down via its lr multiplier (0.8 ->
        # 4e-4) in `wav2vec_u_param_groups`.
        "__lr_opts": {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": [0, base_num_epochs],
            "piecewise_values": [5e-4, 5e-4],
        },
        "grad_scaler": None,  # fairseq fp16: false -> train in fp32
        # fairseq dataset.batch_size: 160 sentences (no token cap; task.max_length: null). Cap by
        # sentence count and keep the frame budget high so max_seqs is the binding constraint.
        "batch_size": 3_200_000,
        "max_seqs": 320,
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,  # fairseq optimization.clip_norm: 5.0
        # fairseq composite optimizer: two Adam groups (generator wd=0 lr=4e-4, discriminator wd=1e-4
        # lr=5e-4, both betas=(0.5,0.98) eps=1e-6). `GanAlternatingAdamW` is one AdamW that steps only
        # the active group each update (the train step picks it), while RETURNN's global grad clip
        # still sees both groups' grads (matching fairseq's clip scope). `param_groups_custom`
        # provides the per-group weight_decay + lr multiplier.
        "optimizer": {
            "class": CodeWrapper("GanAlternatingAdamW"),
            "epsilon": 1e-6,
            "betas": (0.5, 0.98),
            "param_groups_custom": CodeWrapper("wav2vec_u_param_groups"),
        },
    },
    "model_args": {
        "output_dim": _phon_vocab_size + 1,  # +1 dedicated pad class
        "pad_idx": _phon_vocab_size,
        "input_dim": 512,
        "discriminator_dim": 384,
        "discriminator_depth": 2,
        "discriminator_kernel": 6,
        "discriminator_causal": True,
        "discriminator_max_pool": False,
        "discriminator_act_after_linear": False,
        "discriminator_dropout": 0.0,
        "discriminator_weight_norm": False,
        "generator_kernel": 4,
        "generator_stride": 1,
        "generator_bias": False,
        "generator_dropout": 0.1,
        "smoothness_weight": 0.5,
        "smoothing": 0.0,
        "smoothing_one_sided": False,
        "gradient_penalty": 1.5,
        "code_penalty": 4.0,
        "gumbel": False,
        "hard_gumbel": False,
        "temp": (2, 0.1, 0.99995),
        "segmentation": {"type": "JOIN", "mean_pool_join": False, "remove_zeros": False},
    },
    "train_args": {
        "features_key": "data",
        "text_key": "phon_indices",
    },
}


# per-group optimizer settings: generator lr = 5e-4 * 0.8 = 4e-4 (wd 0), discriminator lr = 5e-4 * 1.0
# (wd 1e-4). Matches fairseq config/gan/w2vu.yaml optimizer.groups.
wav2vec_u_param_groups = PartialImport(
    code_object_path=(
        "i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.models.optim.wav2vec_u_param_groups"
    ),
    import_as="wav2vec_u_param_groups",
    hashed_arguments={
        "generator_weight_decay": 0.0,
        "generator_lr_multiplier": 0.8,  # 5e-4 -> 4e-4
        "discriminator_weight_decay": 1e-4,
        "discriminator_lr_multiplier": 1.0,  # 5e-4
    },
    unhashed_arguments={},
    unhashed_package_root=None,
)


# custom AdamW that steps only the active group each update (fairseq composite-optimizer behavior);
# referenced by the optimizer's `class` via CodeWrapper("GanAlternatingAdamW").
wav2vec_u_optimizer_class = Import(
    code_object_path=(
        "i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.models.optim.GanAlternatingAdamW"
    ),
    import_as="GanAlternatingAdamW",
    unhashed_package_root=None,
)


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        # wav2vec-U recognition options (no beam search; audio features -> phoneme argmax). Only used
        # when recognition is enabled (skip_eval below).
        decoder_config=DecoderConfig(),
        # GAN stage only: no ASR recognition/scoring here (fairseq selects a checkpoint via an
        # unsupervised metric, done separately).
        skip_eval=True,
        additional_configs=[
            ReturnnConfig(
                config={},
                python_prolog=[Collection([wav2vec_u_optimizer_class, wav2vec_u_param_groups])],
            )
        ],
    )
