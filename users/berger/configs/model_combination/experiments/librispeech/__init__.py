from typing import Dict, List, Optional, Tuple

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from ...model_pipelines.common.train import TrainedModel
from . import dynamic_qwen, lexicon_oov, nbest_qwen, oov_length, recognition, training


def run_all(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "aed_bpe": training.aed_bpe.run(descriptor="aed_bpe"),
        "aed_bpe_i6_decoder": training.aed_bpe_i6_decoder.run(descriptor="aed_bpe_i6_decoder"),
        "ctc_bpe": training.ctc_bpe.run(descriptor="ctc_bpe"),
        "ctc_phoneme": training.ctc_phoneme.run(descriptor="ctc_phoneme"),
        "ffnn_transducer_bpe": training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_pruned_bpe": training.ffnn_transducer_pruned_bpe.run(descriptor="ffnn_transducer_pruned_bpe"),
        "full_ctx_transducer_bpe": training.full_ctx_transducer_bpe.run(descriptor="full_ctx_transducer_bpe"),
        "aed_byte": training.aed_byte.run(descriptor="aed_byte"),
        "ctc_byte": training.ctc_byte.run(descriptor="ctc_byte"),
        "ffnn_transducer_byte": training.ffnn_transducer_byte.run(descriptor="ffnn_transducer_byte"),
    }

    recog_results = []
    recog_results.extend(recognition.aed_bpe.run(model=models["aed_bpe"]))
    recog_results.extend(recognition.aed_bpe_i6_decoder.run(model=models["aed_bpe_i6_decoder"]))
    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"]))
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_bpe"]))
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_pruned_bpe"]))
    recog_results.extend(recognition.full_ctx_transducer_bpe.run(model=models["full_ctx_transducer_bpe"]))
    recog_results.extend(recognition.ctc_phoneme.run(model=models["ctc_phoneme"]))
    recog_results.extend(recognition.aed_ctc_bpe.run(aed_model=models["aed_bpe"], ctc_model=models["ctc_bpe"]))
    recog_results.extend(
        recognition.aed_ctc_bpe.run(
            aed_model=models["aed_bpe"],
            ctc_model=models["ctc_bpe"],
            variants=recognition.aed_ctc_bpe.word_exit_tuning_variants(),
            corpora=["dev-other"],
        )
    )
    recog_results.extend(recognition.aed_byte.run(model=models["aed_byte"]))
    recog_results.extend(recognition.ctc_byte.run(model=models["ctc_byte"]))
    recog_results.extend(recognition.ffnn_transducer_byte.run(model=models["ffnn_transducer_byte"]))
    # Externally trained model, so there is nothing to train here.
    recog_results.extend(recognition.ctc_granite.run())
    # Completed Granite LM-scale/text-normalization sweeps remain opt-in.
    recog_results.extend(
        lexicon_oov.run(
            ctc_model=models["ctc_bpe"],
            transducer_model=models["ffnn_transducer_bpe"],
        )
    )
    # Retire old-build prototypes and completed alpha/beta/beam grids from the
    # active report. Definitions and job artifacts remain available; retain only
    # selected same-build references alongside the new word-Transformer grid.
    recog_results.extend(oov_length.run_retained_comparisons(ctc_model=models["ctc_bpe"]))
    # Retain the selected open/closed Transformer controls; retire the completed
    # cost sweep from the active report in favor of independent beam-cap tuning.
    recog_results.extend(oov_length.run_granite_transformer_pruning_tuning())
    # Test full-string Qwen fusion on open-tree candidates before implementing
    # dynamic in-search tokenization. Closed-tree sweeps remain callable/opt-in.
    nbest_qwen.run()
    # Pin the new no-lexicon dynamic word-tokenization prototype separately.
    dynamic_qwen.run()
    dynamic_qwen.run_rescoring_transfer()
    dynamic_qwen.run_tightened_beam_tuning()

    if report_filename is not None:
        register_recog_report(recog_results, filename=report_filename)
    return models, recog_results
