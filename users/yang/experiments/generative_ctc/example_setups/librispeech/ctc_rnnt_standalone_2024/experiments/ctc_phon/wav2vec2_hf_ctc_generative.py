import copy
from dataclasses import asdict
from typing import cast

import numpy as np
from sisyphus import Job, Task, tk
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import MINI_RETURNN_ROOT, RETURNN_EXE
from ...lm import get_4gram_binary_lm
from ...pipeline import prepare_asr_model, search, training
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
from ...pytorch_networks.ctc.wav2vec2_hf_ctc_generative_cfg import ModelConfig


class CtcGenerativeLmTuneSummaryJob(Job):
    def __init__(self, *, tune_rows, selected_lm):
        self.tune_rows = tune_rows
        self.selected_lm = selected_lm
        self.out_report = self.output_path("ctc_generative_lm_tuning_summary.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _read_value(value) -> float:
        if hasattr(value, "get"):
            return float(value.get())
        with open(tk.uncached_path(value), "rt") as f:
            text = f.read().strip()
        return float(text)

    def run(self):
        tune_results = [
            (float(row["lm_weight"]), self._read_value(row["wer"]))
            for row in self.tune_rows
        ]
        tune_results.sort(key=lambda item: item[1])
        selected_lm = self._read_value(self.selected_lm)
        with open(self.out_report.get_path(), "wt") as f:
            f.write("dev-other_lm_tuning\n")
            f.write("lm_weight\twer\n")
            for lm_weight, wer in tune_results:
                f.write(f"{lm_weight:g}\t{wer:.4f}\n")
            f.write("\nselected\n")
            f.write(f"best_lm_weight\t{selected_lm:g}\n")


def eow_phon_ls960_wav2vec2_hf_ctc_generative():
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_wav2vec2_hf_generative"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )

    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_other_dataset_tuples = {"dev-other": build_test_dataset(dataset_key="dev-other", settings=train_settings)}
    test_other_dataset_tuples = {"test-other": build_test_dataset(dataset_key="test-other", settings=train_settings)}

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)
    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    def run_and_search(
        *,
        run_name: str,
        model_config: ModelConfig,
        num_epochs: int,
        batch_size: int = 120,
        accum_grad_multiple_step: int = 2,
        init_lr: float = 1e-5,
        peak_lr: float = 5e-5,
        decode_epochs=(200,),
        decode_layers=(6,),
        gpu_mem: int = 24,
        search_batch_size: int = 50 * 16000,
        lm_weights=(0.6, 0.8, 1.0, 1.2, 0.4),
        beam_size: int = 1024,
        beam_size_token: int = 12,
        beam_threshold: float = 14.0,
    ):
        network_module = "ctc.wav2vec2_hf_ctc_generative"
        training_name = prefix_name + "/" + network_module + "." + run_name
        epoch_1 = int(num_epochs * 0.4)
        epoch_2 = num_epochs - 2 * epoch_1
        train_config = {
            "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(init_lr, peak_lr, epoch_1))
            + list(np.linspace(peak_lr, init_lr, epoch_1))
            + list(np.linspace(init_lr, 1e-06, epoch_2)),
            "batch_size": batch_size * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": accum_grad_multiple_step,
            "torch_amp_options": {"dtype": "bfloat16"},
            "gradient_clip_norm": 1.0,
            "num_workers_per_gpu": 2,
        }
        train_args = {
            "config": train_config,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_speed_perturbation": True,
            "debug": False,
        }

        train_job = training(training_name, train_data, train_args, num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = gpu_mem

        for epoch in decode_epochs:
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args,
                with_prior=False,
                datasets=None,
                get_specific_checkpoint=epoch,
            )

            for decode_layer_index in decode_layers:
                tune_parameters = []
                tune_values = []
                tune_rows = []
                for lm_weight in lm_weights:
                    lm_name = ("%g" % lm_weight).replace(".", "p")
                    decoder_config = DecoderConfig(
                        lexicon=get_text_lexicon(),
                        returnn_vocab=label_datastream.vocab,
                        beam_size=beam_size,
                        beam_size_token=beam_size_token,
                        arpa_lm=arpa_4gram_lm,
                        beam_threshold=beam_threshold,
                        lm_weight=lm_weight,
                        prior_scale=0.0,
                        prior_file=None,
                        decode_layer_index=decode_layer_index,
                    )

                    dev_search_name = training_name + f"/decode_dev_other_ep{epoch}_layer_{decode_layer_index}/lm{lm_name}"
                    _search_jobs, wers = search(
                        dev_search_name,
                        forward_config={"batch_size": search_batch_size, "num_workers_per_gpu": 0},
                        asr_model=copy.deepcopy(asr_model),
                        decoder_module="ctc.decoder.flashlight_ctc_v1",
                        decoder_args={"config": asdict(decoder_config)},
                        test_dataset_tuples=dev_other_dataset_tuples,
                        use_gpu=True,
                        **default_returnn,
                    )
                    tune_parameters.append((lm_weight,))
                    tune_values.append(wers[dev_search_name + "/dev-other"])
                    tune_rows.append({"lm_weight": lm_weight, "wer": wers[dev_search_name + "/dev-other"]})

                pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                    parameters=tune_parameters,
                    values=tune_values,
                    mode="minimize",
                )
                pick_optimal_params_job.add_alias(
                    training_name + f"/decode_dev_other_ep{epoch}_layer_{decode_layer_index}/pick_best_lm"
                )
                summary_job = CtcGenerativeLmTuneSummaryJob(
                    tune_rows=tune_rows,
                    selected_lm=pick_optimal_params_job.out_optimal_parameters[0],
                )
                summary_job.add_alias(training_name + f"/decode_ep{epoch}_layer_{decode_layer_index}/lm_tuning_summary")
                tk.register_output(
                    training_name + f"/decode_ep{epoch}_layer_{decode_layer_index}/lm_tuning_summary.txt",
                    summary_job.out_report,
                )
                best_decoder_config = DecoderConfig(
                    lexicon=get_text_lexicon(),
                    returnn_vocab=label_datastream.vocab,
                    beam_size=beam_size,
                    beam_size_token=beam_size_token,
                    arpa_lm=arpa_4gram_lm,
                    beam_threshold=beam_threshold,
                    lm_weight=pick_optimal_params_job.out_optimal_parameters[0],
                    prior_scale=0.0,
                    prior_file=None,
                    decode_layer_index=decode_layer_index,
                )
                test_search_name = training_name + f"/decode_test_other_ep{epoch}_layer_{decode_layer_index}/best_lm"
                _search_jobs, test_wers = search(
                    test_search_name,
                    forward_config={"batch_size": search_batch_size, "num_workers_per_gpu": 0},
                    asr_model=copy.deepcopy(asr_model),
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(best_decoder_config)},
                    test_dataset_tuples=test_other_dataset_tuples,
                    use_gpu=True,
                    **default_returnn,
                )

        return train_job

    model_config = ModelConfig(
        label_target_size=vocab_size_without_blank,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=True,
        final_dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        activation_dropout=0.1,
        layerdrop=0.05,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        gradient_checkpointing=False,
        aux_ctc_loss_layers=[9],
        aux_ctc_loss_scales=[1.0],
        sampling_type="batch",
        sampling_ratio=0.1,
        share_samples=False,
        ratio_corrector=1.0,
        input_time_batch_norm=False,
        input_residual_linear=False,
        generator_kernel=1,
        generator_stride=1,
        generator_dilation=1,
        generator_bias=True,
    )

    # run_and_search(
    #     run_name="base_pretrained_aux3369final_gennce_ctcfix_lr5e-5_bs120x2_amp_noprior",
    #     model_config=model_config,
    #     num_epochs=200,
    #     batch_size=120,
    #     accum_grad_multiple_step=2,
    #     init_lr=1e-5,
    #     peak_lr=5e-5,
    #     decode_epochs=(200,),
    #     decode_layers=(9,),
    #     gpu_mem=24,
    # )

    model_config_kernel4_stride2 = copy.deepcopy(model_config)
    model_config_kernel4_stride2.generator_kernel = 4
    model_config_kernel4_stride2.generator_stride = 2
    model_config_kernel4_stride2.sampling_ratio = 0.5
    model_config_kernel4_stride2.share_samples = True

    run_and_search(
        run_name="base_generative_forzen_enc",
        model_config=model_config_kernel4_stride2,
        num_epochs=200,
        batch_size=150,
        accum_grad_multiple_step=4,
        init_lr=1e-5,
        peak_lr=5e-5,
        decode_epochs=(200,),
        decode_layers=(9,),
        gpu_mem=24,
    )
    model_config_kernel4_stride2_trainable = copy.deepcopy(model_config_kernel4_stride2)
    model_config_kernel4_stride2_trainable.freeze_encoder =False
    model_config_kernel4_stride2_trainable.freeze_feature_encoder =False

    run_and_search(
        run_name="base_generative_trainable_enc",
        model_config=model_config_kernel4_stride2_trainable,
        num_epochs=200,
        batch_size=150,
        accum_grad_multiple_step=4,
        init_lr=1e-5,
        peak_lr=1e-4,
        decode_epochs=(200,),
        decode_layers=(9,),
        gpu_mem=24,
    )


py = eow_phon_ls960_wav2vec2_hf_ctc_generative
