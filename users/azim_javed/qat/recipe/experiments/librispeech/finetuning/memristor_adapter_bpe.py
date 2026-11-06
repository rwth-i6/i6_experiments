from dataclasses import replace
from typing import Optional

from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.text.processing import HeadJob
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.setups.serialization import ExternalImport
from sisyphus import tk
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.datasets import get_default_bpe_train_data, get_default_bpe_cv_data
from ....model_pipelines.common.adapters import ErrorAdapterConfig
from ....model_pipelines.common.learning_rates import OCLRConfig
from ....model_pipelines.common.memrecog import convert_model_for_memristor
from ....model_pipelines.common.optimizer import RAdamConfig
from ....model_pipelines.common.train import FinetuneOptions, TrainOptions, TrainedModel, train
from ....model_pipelines.qat_ffnn_transducer.adaptation.config import MemristorAdapterConfig
from ....model_pipelines.qat_ffnn_transducer.adaptation.pytorch_modules import MemristorAdapterModel
from ....model_pipelines.qat_ffnn_transducer.adaptation.train import get_train_step_import
from ....model_pipelines.qat_ffnn_transducer.mem_inited.pytorch_modules import (
    QATFFNNTransducerConfig,
    QATFFNNTransducerJointADCConfig,
)
from ....model_pipelines.qat_ffnn_transducer.pytorch_modules import (
    QATFFNNTransducerConfig as DigitalQATFFNNTransducerConfig,
    QATFFNNTransducerModel as DigitalQATFFNNTransducerModel,
)
from ....tools import synaptogen_ml_root


def run(
    descriptor: str,
    base_model: TrainedModel[DigitalQATFFNNTransducerConfig],
    train_options: Optional[TrainOptions] = None,
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    correction_settings: Optional[CycleCorrectionSettings] = None,
    num_cycles: int = 1,
    seed: int = 42,
    joint_l1_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l2_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
) -> TrainedModel[MemristorAdapterConfig]:
    if converter_hardware_settings is None:
        converter_hardware_settings = DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0)
    if pos_enc_converter_hardware_settings is None:
        pos_enc_converter_hardware_settings = DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0)
    if train_options is None:
        bpe_size = vocab_to_bpe_size(base_model.model_config.target_size - 1)
        train_options = get_train_options(bpe_size=bpe_size)

    checkpoint, converted_config = convert_model_for_memristor(
        checkpoint=base_model.get_checkpoint(),
        config=base_model.model_config,
        model_class=DigitalQATFFNNTransducerModel,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        num_cycles=num_cycles,
        alias=descriptor,
    )
    tk.register_output(f"training/{descriptor}/memristor_checkpoint", checkpoint.path)
    if joint_l1_converter_hardware_settings is not None or joint_l2_converter_hardware_settings is not None:
        base_config = QATFFNNTransducerJointADCConfig(
            **vars(converted_config),
            joint_l1_converter_hardware_settings=(joint_l1_converter_hardware_settings or converter_hardware_settings),
            joint_l2_converter_hardware_settings=(joint_l2_converter_hardware_settings or converter_hardware_settings),
        )
    else:
        base_config = QATFFNNTransducerConfig(**vars(converted_config))
    model_config = get_model_config(base_config)
    return train(
        descriptor=descriptor,
        model_class=MemristorAdapterModel,
        model_config=model_config,
        options=train_options,
        train_step_import=get_train_step_import(),
        finetune_options=FinetuneOptions(
            label="memristor_backbone",
            filename=checkpoint.path,
            ignore_missing=False,
            allowed_missing_prefix=["adapters."],
        ),
        seed=seed,
        extra_python_prolog=[ExternalImport(synaptogen_ml_root)],
        plot=True,
        move_to_hpc=True,
    )


def get_model_config(base_config: QATFFNNTransducerConfig) -> MemristorAdapterConfig:
    return MemristorAdapterConfig(
        base_config=base_config,
        adapters=[ErrorAdapterConfig(location="output_log_probs")],
    )


def get_train_options(
    bpe_size: int = 128,
    num_epochs: int = 10,
    num_segments: Optional[int] = 960,
    gpu_mem_rqmt: float = 48,
) -> TrainOptions:
    train_data = get_default_bpe_train_data(bpe_size)
    cv_data = get_default_bpe_cv_data(bpe_size)
    if num_segments is not None:
        subsets = SegmentCorpusJob(
            bliss_corpus=get_bliss_corpus_dict("wav")["train-other-960"],
            num_segments=num_segments,
        )
        data = replace(train_data, speed_perturbation=False)
        train_data = replace(data, segment_file=subsets.out_single_segment_files[1])
        cv_data = replace(
            data,
            segment_file=HeadJob(subsets.out_single_segment_files[2], num_lines=2, zip_output=False).out,
            partition_epoch=1,
            seq_ordering="sorted",
        )
    partition_epoch = train_data.partition_epoch
    return TrainOptions(
        train_data_config=train_data,
        cv_data_config=cv_data,
        save_epochs=[epoch * partition_epoch for epoch in sorted({max(1, num_epochs // 2), num_epochs})],
        batch_size=120 * 16000,
        accum_grad_multiple_step=1,
        optimizer_config=RAdamConfig(epsilon=1e-8, weight_decay=0.0, decoupled_weight_decay=True),
        lr_config=OCLRConfig(
            init_lr=1e-4,
            peak_lr=1e-3,
            decayed_lr=1e-4,
            final_lr=1e-4,
            inc_epochs=(num_epochs // 2) * partition_epoch,
            dec_epochs=(num_epochs - num_epochs // 2) * partition_epoch,
            final_epochs=0,
        ),
        gradient_clip=1.0,
        num_workers_per_gpu=1,
        automatic_mixed_precision=False,
        gpu_mem_rqmt=gpu_mem_rqmt,
        max_seqs=1,
        max_seq_length=None,
    )
