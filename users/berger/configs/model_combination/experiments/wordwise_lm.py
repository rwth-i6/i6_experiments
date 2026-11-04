"""Development-only dynamic word fusion for our continuation-marked BPE models."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

from i6_core.rasr import WriteRasrConfigJob
from i6_core.returnn import CodeWrapper
from i6_core.util import DelayedFormat
from sisyphus import tk

from i6_experiments.common.setups.serialization import Call, ExplicitHash, Import

from ..model_pipelines.common.experiment_context import ExperimentContext
from ..model_pipelines.common.recog_rasr_config import (
    LexiconfreeTimesyncRecogParams,
    get_lexiconfree_timesync_recog_config,
)

QWEN_MODELS = {
    "0.6b": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
    "1.7b": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
}


def conditions():
    # Historical uncorrected pilots: preserve their descriptors and job hashes.
    return [
        ("greedy", None, 0.0, 1),
        ("beam32", None, 0.0, 32),
        ("ls-bpe-trafo-lm0.8", "bpe", 0.8, 32),
        ("ls-word-trafo-lm0.8", "word", 0.8, 32),
        ("qwen-0.6b-lm0.3", "qwen-0.6b", 0.3, 32),
        ("qwen-1.7b-lm0.3", "qwen-1.7b", 0.3, 32),
    ]


def corrected_conditions():
    """Dev-only scale retuning; correction 0.2 matches the existing LM recipes.

    This is a starting correction value, not a claim that it is optimal for
    every external LM. Keep pruning/rewards fixed to isolate score calibration.
    """
    result = [("beam32", None, 0.0, 32)]
    for label, kind, scales in (
        ("ls-bpe-trafo", "bpe", (0.4, 0.8, 1.2)),
        ("ls-word-trafo", "word", (0.4, 0.8, 1.2)),
        ("qwen-0.6b", "qwen-0.6b", (0.3, 0.5, 0.7)),
        ("qwen-1.7b", "qwen-1.7b", (0.3, 0.5, 0.7)),
    ):
        result.extend((f"{label}-lm{scale:g}", kind, scale, 32) for scale in scales)
    return result


def unknown_penalty_conditions():
    return [
        (f"ls-word-trafo-lm{scale:g}-unk-beta{penalty:g}", "word", scale, 32, penalty)
        for scale in (0.4, 0.8)
        for penalty in (1.0, 2.0, 4.0, 8.0, 12.0)
    ]


def recognition_config(
    vocab, scorers, *, blank, ctc, kind, scale, beam, artifacts,
    threshold=8.0, pre_beam=128, unknown_penalty=0.0,
):
    if unknown_penalty and (kind != "word" or scale <= 0):
        raise ValueError("UNK word penalty requires a word LM and positive scale")
    base = get_lexiconfree_timesync_recog_config(
        vocab,
        scorers,
        LexiconfreeTimesyncRecogParams(
            max_beam_sizes=[beam],
            collapse_repeated_labels=ctc,
            score_thresholds=[0.0 if beam == 1 else threshold],
            log_stepwise_statistics=False,
            recombination_mode="on",
        ),
        blank_index=blank,
    ).creator
    config, post = copy.deepcopy(base.config), copy.deepcopy(base.post_config)
    s = config.lib_rasr.search_algorithm
    if kind is not None:
        s.type = "llm-timesync-beam-search"  # RNA consumes one AM frame per label, unlike standard RNN-T.
        s.word_piece_convention, s.word_piece_marker = "continuation-marked", "@@"
        s.pre_llm_max_beam_size, s.pre_llm_score_threshold = pre_beam, threshold
        s.llm_scale, s.word_penalty = scale, 0.0
        s.llm.max_batch_size = 8
        if kind.startswith("qwen-"):
            s.llm.type, s.llm.model = "wordwise-qwen", artifacts[kind]
            s.llm.outer_lm_scale, s.llm.token_reward = scale, 0.5
            s.word_penalty = -1.0
        else:
            s.llm.type, s.llm.kind = "wordwise-transformer", kind
            s.llm.model, s.llm.vocab, codes = artifacts[kind]
            if codes is not None:
                s.llm.bpe_codes = codes
            if unknown_penalty:
                s.llm.type = "wordwise-transformer-unk-penalty"
                s.llm.unknown_word_penalty = unknown_penalty
                s.llm.outer_lm_scale = scale
    return WriteRasrConfigJob(config, post).out_config


def lm_artifacts():
    """Reuse canonical training job identities, regardless of the current AM context."""
    from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab, get_subword_nmt_bpe_v2

    # Initialize the existing experiment package first: its recognition modules
    # also import data.librispeech.lm and otherwise create a circular import.
    from .librispeech import training  # noqa: F401 -- required import order, see above
    from ..data.huggingface import DownloadHuggingFaceRepoJob
    from ..data.librispeech import lm
    from ..model_pipelines.transformer_lm.export import export_model_stateless

    with ExperimentContext("librispeech"):
        bpe_cfg, bpe_ckpt = lm._get_bpe_transformer_lm_artifacts(bpe_size=128, num_layers=96)
        bpe_onnx = export_model_stateless(bpe_cfg, bpe_ckpt, last_hidden_only=True)
        word_cfg, word_ckpt = lm._get_transformer_lm_artifacts(layers=lm.TransformerLmParams().layers)
        word_onnx = export_model_stateless(word_cfg, word_ckpt, last_hidden_only=True)
        word_vocab = get_lm_vocab(output_prefix="").vocab
        bpe = get_subword_nmt_bpe_v2("train-other-960", 128)
    result = {"bpe": (bpe_onnx, bpe.bpe_vocab, bpe.bpe_codes), "word": (word_onnx, word_vocab, None)}
    for size, (repo, revision) in QWEN_MODELS.items():
        result[f"qwen-{size}"] = DownloadHuggingFaceRepoJob(
            repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
        ).out_repo_dir
    return result


def acoustic_models():
    """Only existing BPE acoustic training jobs; no new/v2 training variants."""
    from .librispeech import training as ls
    from .loquacious import training as loq

    for context, train, names in (
        ("librispeech", ls, ("ctc_bpe", "ffnn_transducer_bpe")),
        ("loquacious_medium", loq.medium, ("ctc_bpe", "ffnn_transducer_bpe")),
    ):
        with ExperimentContext(context):
            models = {name: getattr(train, name).run(descriptor=name) for name in names}
        for name, model in models.items():
            yield context, name, model


def run(*, models=None, artifacts=None, include_wide_diagnostic=False):
    from .. import tools
    from ..data.librispeech import bpe as ls_bpe, datasets as ls_data
    from ..data.loquacious import bpe as loq_bpe, datasets as loq_data
    from ..model_pipelines.common import dynamic_llm, dynamic_qwen_tuned, rnnt_llm, wordwise_transformer
    from ..model_pipelines.common import wordwise_unknown
    from ..model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
    from ..model_pipelines.common.recog import OfflineRecogParameters, recog_rasr_offline
    from ..model_pipelines.common.report import register_recog_report
    from ..model_pipelines.common.serializers import get_model_serializers
    from ..model_pipelines.ffnn_transducer.pytorch_modules import FFNNTransducerEncoder
    from ..model_pipelines.full_ctx_transducer.pytorch_modules import LstmTransducerEncoder

    version = hashlib.sha256(
        b"\0".join(
            Path(m.__file__).read_bytes() for m in (wordwise_transformer, dynamic_llm, dynamic_qwen_tuned, rnnt_llm)
        )
    ).hexdigest()
    binary = tools.get_llm_rnnt_search_rasr_binary_path()  # Also contains the new CTC/RNA API and @@ assembler.
    gate = wordwise_transformer.BpeWordwiseGateJob(binary, code_version=version)
    gate.add_alias("wordwise-lm/dev/topology-gate")
    tk.register_output("wordwise-lm/dev/topology-gate", gate.out_report)
    tk.register_output("wordwise-lm/dev/topology-gate.log", gate.out_log)
    artifacts = lm_artifacts() if artifacts is None else artifacts
    results = []
    for context, name, model in acoustic_models() if models is None else models:
        ls = context == "librispeech"
        dataset = "dev-other" if ls else "dev.short"
        ctc = name == "ctc_bpe"
        module_name = (
            "ctc_bpe"
            if ctc
            else "full_ctx_transducer_bpe"
            if name == "full_ctx_transducer_bpe"
            else "ffnn_transducer_bpe"
        )
        if ls:
            from .librispeech import recognition
        else:
            from .loquacious import recognition
        recog = getattr(recognition, module_name)
        data_module, bpe_module = (ls_data, ls_bpe) if ls else (loq_data, loq_bpe)
        train_key = "train.small" if context == "loquacious_small" else "train.medium"
        blank = model.model_config.target_size - 1
        bpe_size = bpe_module.vocab_to_bpe_size(blank)
        vocab_kwargs = {} if ls else dict(corpus_key=train_key)
        vocab = bpe_module.get_bpe_vocab_file(bpe_size, add_blank=True, **vocab_kwargs)
        # Preserve the waveform processing used to train these models. Unlike
        # HF models, removing pre-emphasis would change the AM input domain.
        data = data_module.get_default_recog_data(dataset)
        corpus = data_module.get_default_score_corpus(dataset)
        with ExperimentContext(context):
            runs = [(condition, 0.0, 0.0) for condition in conditions()]
            runs.extend((condition, 0.2, 0.0) for condition in corrected_conditions())
            runs.extend((condition[:4], 0.2, condition[4]) for condition in unknown_penalty_conditions())
            if include_wide_diagnostic and ls and name == "ffnn_transducer_bpe":
                # One diagnostic matching the old 3.8% tree run's main
                # score/pruning settings; the dynamic search remains different.
                runs.append((("ls-word-trafo-lm0.8-wide-beam1024-thr16", "word", 0.8, 1024), 0.2, 0.0))
            for (label, kind, scale, beam), correction, unknown_penalty in runs:
                if correction:
                    label += f"-{'prior' if ctc else 'ilm'}{correction:g}"
                variant_cls = recog.CTCRecogVariant if ctc else recog.TransducerRecogVariant
                variant = variant_cls(
                    descriptor=f"wordwise-{label}",
                    search_algorithm_params=LexiconfreeTimesyncRecogParams(
                        max_beam_sizes=[beam], collapse_repeated_labels=ctc
                    ),
                    search_mode_params=OfflineRecogParameters(mem_rqmt=64, gpu_mem_rqmt=48),
                    **{("prior_scale" if ctc else "ilm_scale"): correction},
                )
                # Reuse the original corpus-specific prior estimation and ILM
                # export. Never subtract an external LM as an ILM proxy.
                if ctc:
                    kwargs = {} if ls else dict(train_corpus_key=train_key)
                    serializers = recog._get_model_serializers(model=model, variant=variant, **kwargs)
                    scorers = [get_no_op_label_scorer_config()]
                else:
                    encoder = LstmTransducerEncoder if name == "full_ctx_transducer_bpe" else FFNNTransducerEncoder
                    serializers = get_model_serializers(encoder, model.model_config)
                    scorers = recog._get_label_scorer_configs(model=model, variant=variant)
                    if name == "full_ctx_transducer_bpe":
                        for component in ("scorer_model", "state_initializer_model", "state_updater_model"):
                            scorers[0][component].session.execution_provider_type = "cuda"
                serializers.serializer_objects.append(ExplicitHash(version))
                if unknown_penalty:
                    serializers.serializer_objects.append(
                        ExplicitHash(hashlib.sha256(Path(wordwise_unknown.__file__).read_bytes()).hexdigest())
                    )
                hook = None
                if kind is not None:
                    register_hook = (
                        wordwise_unknown.RegisterWordwiseUnknownPenalty
                        if unknown_penalty else wordwise_transformer.RegisterWordwiseTransformer
                    )
                    serializers.serializer_objects.extend(
                        [
                            Import("sisyphus.tk"),
                            Import(register_hook),
                            Call(
                                register_hook.__name__,
                                kwargs=[("gate_report", DelayedFormat('tk.Path("{}")', gate.out_report))],
                                return_assign_variables="wordwise_lm_init_hook",
                            ),
                        ]
                    )
                    hook = CodeWrapper("wordwise_lm_init_hook")
                config = recognition_config(
                    vocab, scorers, blank=blank, ctc=ctc, kind=kind, scale=scale, beam=beam, artifacts=artifacts,
                    unknown_penalty=unknown_penalty,
                    **(dict(threshold=16.0, pre_beam=1024) if "-wide-beam1024-thr16" in label else {}),
                )
                result = recog_rasr_offline(
                    descriptor=f"{model.descriptor}__recog_wordwise-{label}",
                    recog_rasr_config_file=config,
                    recog_data_config=data,
                    recog_corpus=corpus,
                    encoder_serializers=serializers,
                    sample_rate=16000,
                    params=variant.search_mode_params,
                    checkpoint=model.get_checkpoint(),
                    rasr_binary_path_override=binary,
                    rasr_init_hook=hook,
                    normalize_hf_score=True,
                )
                results.append(replace(result, descriptor=f"{context}/{result.descriptor}"))
    with ExperimentContext("wordwise-lm/dev"):
        register_recog_report(results, filename="summary")
    return results
