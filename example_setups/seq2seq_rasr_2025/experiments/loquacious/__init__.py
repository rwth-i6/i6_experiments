from typing import Dict, List, Optional, Tuple

from i6_core.returnn import PtCheckpoint
from sisyphus import tk

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from ...model_pipelines.common.train import TrainedModel
from . import recognition, training


def run_small(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "ctc_bpe": training.small.ctc_bpe.run(descriptor="ctc_bpe"),
        "ffnn_transducer_bpe": training.small.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_phoneme": training.small.ffnn_transducer_phoneme.run(descriptor="ffnn_transducer_phoneme"),
    }

    recog_results = []
    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"], train_corpus_key="train.small"))
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_bpe"], train_corpus_key="train.small")
    )
    recog_results.extend(recognition.ffnn_transducer_phoneme.run(model=models["ffnn_transducer_phoneme"]))

    if report_filename is not None:
        register_recog_report(recog_results, filename=report_filename)
    return models, recog_results


def run_medium(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "aed_bpe": training.medium.aed_bpe.run(descriptor="aed_bpe"),
        "combination_model_bpe": training.medium.combination_model_bpe.run(descriptor="combination_model_bpe"),
        "ctc_bpe": training.medium.ctc_bpe.run(descriptor="ctc_bpe"),
        "ctc_bpe_phoneme": training.medium.ctc_bpe_phoneme.run(descriptor="ctc_bpe_phoneme"),
        "ffnn_transducer_bpe": training.medium.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_phoneme": training.medium.ffnn_transducer_phoneme.run(descriptor="ffnn_transducer_phoneme"),
        "aed_byte": training.medium.aed_byte.run(descriptor="aed_byte"),
        "ctc_byte": training.medium.ctc_byte.run(descriptor="ctc_byte"),
        "ffnn_transducer_byte": training.medium.ffnn_transducer_byte.run(descriptor="ffnn_transducer_byte"),
    }

    recog_results = []
    recog_results.extend(recognition.aed_bpe.run(model=models["aed_bpe"], train_corpus_key="train.medium"))
    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"], train_corpus_key="train.medium"))
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_bpe"], train_corpus_key="train.medium")
    )
    recog_results.extend(recognition.ffnn_transducer_phoneme.run(model=models["ffnn_transducer_phoneme"]))
    recog_results.extend(recognition.aed_byte.run(model=models["aed_byte"]))
    recog_results.extend(recognition.ctc_byte.run(model=models["ctc_byte"], train_corpus_key="train.medium"))
    recog_results.extend(recognition.ffnn_transducer_byte.run(model=models["ffnn_transducer_byte"]))

    decoder_huggingface_cache_dir = tk.Path(
        "/work/asr4/berger/rasr_dev/label_scorer/setup/speech_llm/robin_decoding/DownloadHuggingFaceRepoJob.eP43CafWC8I4/output/hub_cache"
    )

    tokenizer_huggingface_repo_dir = tk.Path(
        "/work/asr4/berger/rasr_dev/label_scorer/setup/speech_llm/robin_decoding//DownloadHuggingFaceRepoJobV2.PUGzhO2dOEpK/output/hub_cache/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
    )

    checkpoint = PtCheckpoint(
        tk.Path(
            "/work/asr4/berger/rasr_dev/label_scorer/setup/speech_llm/robin_decoding/ReturnnTrainingJob.L61aRGCVj2Yh/output/models/epoch.025.pt"
        )
    )

    model_kwargs = {
        "encoder_opts": {
            "class": "ConformerEncoderV1",
            "enc_build_dict": {
                "class": "returnn.frontend.encoder.conformer.ConformerEncoder",
                "input_layer": {
                    "class": "returnn.frontend.encoder.conformer.ConformerConvSubsample",
                    "out_dims": [32, 64, 64],
                    "filter_sizes": [(3, 3), (3, 3), (3, 3)],
                    "pool_sizes": [(1, 2)],
                    "strides": [(1, 1), (3, 1), (2, 1)],
                },
                "num_layers": 18,
                "out_dim": 1024,
                "encoder_layer": {
                    "class": "returnn.frontend.encoder.conformer.ConformerEncoderLayer",
                    "ff": {
                        "class": "returnn.frontend.encoder.conformer.ConformerPositionwiseFeedForward",
                        "activation": {"class": "rf.relu_square"},
                        "with_bias": False,
                    },
                    "num_heads": 8,
                },
            },
            "sampling_rate": 16000,
            "specaug_start": (5000, 15000, 25000),
        },
        "adapter_opts": {
            "class": "LinearAdapterWithConcatDownsampling",
            "downsampling_factor": 2,
        },
        "decoder_opts": {
            "class": "Qwen2DecoderV1",
            "hf_hub_cache_dir": decoder_huggingface_cache_dir,
            "device": "cpu",
        },
        "freeze_encoder_params": False,
        "freeze_decoder_params": True,
        "decoder_lora_opts": {
            "target_modules": ["q_proj", "v_proj"],
            "r": 320,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none",
            "use_rslora": True,
        },
        "encoder_lora_opts": None,
        "aux_loss_layers": (18,),
    }

    recog_results.extend(
        recognition.speech_llm.run(
            model_descriptor="slm_robin",
            model_kwargs=model_kwargs,
            checkpoint=checkpoint,
            huggingface_repo_dir=tokenizer_huggingface_repo_dir,
        )
    )

    variants = []
    for ctc_scale in [0.0, 0.3]:
        for length_norm_scale in [0.0, 1.0]:
            for beam_size in [3, 4, 5]:
                for score_threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                    variant = recognition.speech_llm.default_lexfree_recog_variant()
                    variant.descriptor += (
                        f"_beam-{beam_size}_score-{score_threshold}_ln-{length_norm_scale}_ctc-{ctc_scale}"
                    )
                    variant.search_algorithm_params.max_beam_sizes = [64, beam_size] if ctc_scale else [beam_size]
                    variant.search_algorithm_params.score_thresholds = (
                        [10.0, score_threshold] if ctc_scale else [score_threshold]
                    )
                    variant.search_algorithm_params.length_norm_scale = length_norm_scale
                    variant.ctc_score_scale = ctc_scale
                    variants.append(variant)
    recog_results.extend(
        recognition.speech_llm.run(
            model_descriptor="slm_robin",
            model_kwargs=model_kwargs,
            checkpoint=checkpoint,
            huggingface_repo_dir=tokenizer_huggingface_repo_dir,
            variants=variants,
        )
    )

    # recog_results.extend(
    #     recognition.ctc_byte_speech_llm.run(
    #         model=models["ctc_byte"],
    #         speech_lm_model_kwargs=model_kwargs,
    #         speech_lm_checkpoint=checkpoint,
    #         huggingface_repo_dir=tokenizer_huggingface_repo_dir,
    #     )
    # )

    if report_filename is not None:
        register_recog_report(recog_results, filename=report_filename)

    return models, recog_results


def run_large(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "transformer_lm_bpe": training.large.transformer_lm_bpe.run(descriptor="transformer_lm_bpe"),
        "transformer_lm_word": training.large.transformer_lm_word.run(descriptor="transformer_lm_word"),
    }
    model_config = training.large.transformer_lm_bpe.get_model_config(bpe_size=10000)
    train_options = training.large.transformer_lm_bpe.get_train_options(bpe_size=10000)
    models["transformer_lm_bpe"] = training.large.transformer_lm_bpe.run(
        descriptor="transformer_lm_bpe-10k", model_config=model_config, train_options=train_options
    )

    return models, []
