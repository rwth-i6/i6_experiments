"""Closed-tree Qwen Base shallow fusion, independently pinned and correctness-gated."""

from dataclasses import replace
from sisyphus import tk

from ... import tools
from ...data.oov_multitoken_smoke import OovMultiTokenDecoderSmokeJob
from ...data.oov_length_smoke import (
    OovDelayedPenaltyProbeJob,
    OovFirstPiecePruningProbeJob,
    RequireOovPruningProbeJob,
)


def run_preflight():
    binary = tools.get_oov_multitoken_rasr_binary_path()
    prefix = "tools/rasr-oov-8c1ac793"
    smoke = OovMultiTokenDecoderSmokeJob(binary, tools.returnn_python_exe)
    delayed = OovDelayedPenaltyProbeJob(binary, tools.returnn_python_exe)
    first = OovFirstPiecePruningProbeJob(binary, tools.returnn_python_exe)
    pruning = RequireOovPruningProbeJob(delayed.out_report, first_piece_report=first.out_report)
    for name, job in (
        ("multi-token-smoke", smoke),
        ("delayed-penalty", delayed),
        ("first-piece", first),
        ("pruning-acceptance", pruning),
    ):
        job.add_alias(f"{prefix}/{name}")
        tk.register_output(f"{prefix}/{name}", job.out_report)
        if hasattr(job, "out_logs"):
            tk.register_output(f"{prefix}/{name}-logs", job.out_logs)
    return dict(
        binary=binary, correctness=smoke, delayed_penalty_probe=delayed, first_piece_probe=first, pruning=pruning
    )


def recognition_variants(
    binary, standard_lexicon, qwen_lexicon, qwen_model, qwen_vocab, word_trafo_model, word_trafo_vocab
):
    """Dev-only scale sweep plus matched controls on the same decoder build."""
    from ...data.librispeech.lm import ArpaLmParams, TransformerLmParams, OnnxTokenLmParams
    from .recognition import ctc_granite

    conditions = [
        ("closed-no-lm", standard_lexicon, None),
        ("closed-4gram-lm0.3", standard_lexicon, ArpaLmParams(scale=0.3)),
        (
            "closed-word-trafo-lm0.3_gpu-stateless_lm-batch1",
            standard_lexicon,
            TransformerLmParams(
                scale=0.3,
                onnx_model=word_trafo_model,
                vocab_file=word_trafo_vocab,
                use_gpu=True,
                use_kv_cache=False,
                max_batch_size=1,
            ),
        ),
    ]
    for scale in (0.0, 0.1, 0.3, 0.5):
        lm = OnnxTokenLmParams(qwen_model, qwen_vocab, scale=scale) if scale else None
        conditions.append((f"closed-qwen3-0.6b-base-lm{scale:g}", qwen_lexicon, lm))
    variants = []
    for name, lexicon, lm in conditions:
        variant = ctc_granite.default_offline_tree_recog_variant()
        variant.descriptor = f"llm-8c1ac793_{name}"
        variant.rasr_binary_path_override = binary
        variant.prepared_lexicon_file = lexicon
        variant.prior_scale = variant.blank_penalty = 0.0
        variant.search_algorithm_params = replace(
            variant.search_algorithm_params,
            word_lm_params=lm,
            max_beam_sizes=[256],
            score_thresholds=[16.0],
            max_word_end_beam_size=256,
            word_end_score_threshold=16.0,
            sentence_end_fallback=False,
            recombination_mode="on",
            unknown_word_fallback="disabled",
        )
        if isinstance(lm, OnnxTokenLmParams):
            variant.search_mode_params = replace(variant.search_mode_params, mem_rqmt=48, gpu_mem_rqmt=24)
        elif isinstance(lm, TransformerLmParams) and lm.use_gpu:
            variant.search_mode_params = replace(variant.search_mode_params, mem_rqmt=32, gpu_mem_rqmt=24)
        variants.append(variant)
    greedy = ctc_granite.default_offline_lexfree_recog_variant()
    greedy.descriptor = "llm-8c1ac793_lexfree-greedy"
    greedy.rasr_binary_path_override = binary
    variants.append(greedy)
    return variants


def focused_qwen_variants(template, variant_lexicon, variant_model, variant_vocab, formatter):
    """Bounded dev-only follow-up; isolate text policy from scale and pruning.

    Keep the completed comparisons unchanged. No test-set selection, token-count
    averaging, second LM, or fabricated Qwen unknown token is introduced.
    """
    from ...data.librispeech.lm import OnnxTokenLmParams

    variants = []
    # Two matched zero-scale controls, four tight-budget scales, two threshold
    # comparisons, and two original-text controls at the identical tight budget.
    conditions = [("text-variants-v1", 0.0, t) for t in (8.0, 16.0)]
    conditions += [("text-variants-v1", s, 8.0) for s in (0.05, 0.1, 0.2, 0.3)]
    conditions += [("text-variants-v1", s, 16.0) for s in (0.05, 0.2)]
    conditions += [("lowercase", s, 8.0) for s in (0.1, 0.3)]
    for policy, scale, threshold in conditions:
        alternative_text = policy == "text-variants-v1"
        model = variant_model if alternative_text else template.search_algorithm_params.word_lm_params.onnx_model
        vocab = variant_vocab if alternative_text else template.search_algorithm_params.word_lm_params.vocab_file
        variants.append(
            replace(
                template,
                descriptor=(
                    f"llm-8c1ac793_qwen-focus-v2-{policy}-lm{scale:g}"
                    f"_beam64_wordbeam8_thr{threshold:g}"
                ),
                prepared_lexicon_file=variant_lexicon if alternative_text else template.prepared_lexicon_file,
                prototype_traceback_formatter_serializers=formatter if alternative_text else None,
                search_algorithm_params=replace(
                    template.search_algorithm_params,
                    word_lm_params=OnnxTokenLmParams(model, vocab, scale=scale) if scale else None,
                    max_beam_sizes=[64],
                    max_word_end_beam_size=8,
                    score_thresholds=[threshold],
                    word_end_score_threshold=threshold,
                ),
                search_mode_params=replace(template.search_mode_params, mem_rqmt=64),
            )
        )
    return variants


def run():
    from i6_core.corpus.convert import CorpusToTxtJob
    from ...data.huggingface import DownloadHuggingFaceRepoJob, get_hf_ctc_vocab_file
    from ...data.librispeech import datasets
    from ...data.oov_neural_lm import get_gpu_word_transformer_params
    from ...data.llm_fusion import (
        QWEN_REPO,
        QWEN_REVISION,
        QwenLexiconJob,
        ExportQwenLmJob,
        ValidateQwenRasrJob,
    )
    from . import oov_length
    from .oov_prototype import _validated_lexicon
    from .recognition import ctc_granite

    preflight = run_preflight()
    granite_repo, old_lexicons = oov_length.setup_granite_lexicons(oov_length.run_preflight(), approaches=("closed",))
    prefix = "datasets/LibriSpeech/llm-8c1ac793/qwen3-0.6b-base"
    lexicon = _validated_lexicon(
        old_lexicons["closed"],
        get_hf_ctc_vocab_file(granite_repo, blank_index=0),
        preflight["binary"],
        f"{prefix}/acoustic-labels",
        decoder_smoke_report=preflight["correctness"].out_report,
    )
    download = DownloadHuggingFaceRepoJob(
        QWEN_REPO, revision=QWEN_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    )
    download.add_alias(f"{prefix}/download")
    repo = download.out_repo_dir
    tk.register_output(f"{prefix}/repo", repo)
    export = ExportQwenLmJob(repo)
    export.add_alias(f"{prefix}/export")
    tk.register_output(f"{prefix}/model.onnx", export.out_model)
    tk.register_output(f"{prefix}/hf-comparison", export.out_report)
    text = CorpusToTxtJob(datasets.get_default_score_corpus("dev-other").bliss_corpus_file).out_txt
    lex = QwenLexiconJob(lexicon, repo, validation_text=text)
    lex.add_alias(f"{prefix}/lexicon")
    tk.register_output(f"{prefix}/tokenization", lex.out_report)
    tk.register_output(f"{prefix}/lm.vocab.txt", lex.out_vocab)
    check = ValidateQwenRasrJob(
        lex.out_lexicon,
        repo,
        export.out_model,
        lex.out_vocab,
        export.out_report,
        preflight["binary"],
        preflight["correctness"].out_report,
        preflight["pruning"].out_report,
        tools.returnn_python_exe,
    )
    check.add_alias(f"{prefix}/native-scoring")
    tk.register_output(f"{prefix}/native-scoring", check.out_report)
    tk.register_output(f"{prefix}/native-scoring-logs", check.out_logs)
    tk.register_output(f"{prefix}/lexicon", check.out_lexicon)
    word_lm = get_gpu_word_transformer_params((lexicon,), "tools/rasr-oov-8c1ac793")
    variants = recognition_variants(
        preflight["binary"],
        lexicon,
        check.out_lexicon,
        export.out_model,
        lex.out_vocab,
        word_lm.onnx_model,
        word_lm.vocab_file,
    )
    # Keep completed controls unchanged; new identities isolate the LM text policy.
    from ...data.qwen_text_variants import (
        QwenTextVariantLexiconJob,
        ExportQwenTextVariantsJob,
        SCORE_CASES as VARIANT_SCORE_CASES,
        VARIANT_NATIVE_RUNNER,
    )
    from ...data.librispeech.lm import OnnxTokenLmParams

    variant_prefix = f"{prefix}/text-variants-v1"
    variant_lex = QwenTextVariantLexiconJob(lexicon, repo)
    variant_export = ExportQwenTextVariantsJob(repo, variant_lex.out_report, export.out_report)
    variant_check = ValidateQwenRasrJob(
        variant_lex.out_lexicon,
        repo,
        variant_export.out_model,
        variant_lex.out_vocab,
        variant_export.out_report,
        preflight["binary"],
        preflight["correctness"].out_report,
        preflight["pruning"].out_report,
        tools.returnn_python_exe,
        runner=VARIANT_NATIVE_RUNNER,
        variant_aliases=variant_lex.out_report,
        score_cases=VARIANT_SCORE_CASES,
    )
    for name, job in (("lexicon", variant_lex), ("export", variant_export), ("native-scoring", variant_check)):
        job.add_alias(f"{variant_prefix}/{name}")
    for name, output in (
        ("lexicon", variant_check.out_lexicon),
        ("tokenization", variant_lex.out_report),
        ("lm.vocab.txt", variant_lex.out_vocab),
        ("model.onnx", variant_export.out_model),
        ("hf-comparison", variant_export.out_report),
        ("native-scoring", variant_check.out_report),
        ("native-scoring-logs", variant_check.out_logs),
    ):
        tk.register_output(f"{variant_prefix}/{name}", output)
    template = next(v for v in variants if v.descriptor.endswith("closed-qwen3-0.6b-base-lm0.3"))
    from i6_experiments.common.setups.serialization import Import
    from i6_experiments.common.setups.serialization import Call
    from ...model_pipelines.common.qwen_traceback import QwenTextVariantTracebackFormatter

    formatter = [
        Import(QwenTextVariantTracebackFormatter),
        Call(QwenTextVariantTracebackFormatter.__name__, return_assign_variables="traceback_formatter"),
    ]
    for scale in (0.1, 0.3, 0.5):
        for word_beam in (8, 16):
            variants.append(
                replace(
                    template,
                    descriptor=(
                        f"llm-8c1ac793_closed-qwen3-0.6b-base-text-variants-v1-lm{scale:g}"
                        f"_beam64_wordbeam{word_beam}_thr16"
                    ),
                    prepared_lexicon_file=variant_check.out_lexicon,
                    prototype_traceback_formatter_serializers=formatter,
                    search_algorithm_params=replace(
                        template.search_algorithm_params,
                        word_lm_params=OnnxTokenLmParams(variant_export.out_model, variant_lex.out_vocab, scale=scale),
                        max_beam_sizes=[64],
                        max_word_end_beam_size=word_beam,
                    ),
                    # Up to six syntactic alternatives per word; acoustic tree prefixes are shared.
                    search_mode_params=replace(template.search_mode_params, mem_rqmt=64),
                )
            )
    variants.extend(
        focused_qwen_variants(
            template, variant_check.out_lexicon, variant_export.out_model, variant_lex.out_vocab, formatter
        )
    )
    return ctc_granite.run(huggingface_repo_dir=granite_repo, variants=variants, corpora=["dev-other"])
