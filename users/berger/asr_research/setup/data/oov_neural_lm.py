"""Preflight for the existing word-level ONNX LM, without adapting its scores."""

import json
import re
import shutil

import numpy as np
from sisyphus import Job, Task, tk

from .librispeech.lexicon_oov import _read_lm_index_vocabulary
from .oov_diagnostics import read_lexicon


WORD_LM_WORK = "/work/asr4/berger/sisyphus_work_dirs/librispeech/20241106_seq2seq_template_setups"
WORD_LM_CACHED_MODEL = tk.Path(
    f"{WORD_LM_WORK}/i6_experiments/example_setups/seq2seq_rasr_2025/"
    "model_pipelines/common/onnx_export/AddOnnxMetadataJob.wJhl1BGaB8Lo/output/model.onnx"
)
WORD_LM_VOCAB = tk.Path(
    f"{WORD_LM_WORK}/i6_core/lm/vocabulary/LmIndexVocabularyFromLexiconJob.lGRAjpScZIsf/output/lm.vocab.txt"
)


def get_original_word_transformer_stateless_model():
    """Re-export the exact completed word LM; do not schedule LM training.

    Checkpoint/config copied from TorchOnnxExportJob.SLHrUJ7Gtfum, the source
    of AddOnnxMetadataJob.wJhl1BGaB8Lo. Explicit dimensions avoid drift with
    training defaults or a different vocabulary construction.
    """
    from i6_core.returnn import PtCheckpoint
    from ..model_pipelines.transformer_lm.export import export_model_stateless
    from ..model_pipelines.transformer_lm.pytorch_modules import (
        TransformerLmConfig,
        PositionalEncodingConfig,
        TransformerBlockConfig,
        TransformerLinearConfig,
        TransformerMHSAConfig,
    )

    cfg = TransformerLmConfig(
        embed_dim=128,
        hid_dim=512,
        vocab_dim=200002,
        num_layers=96,
        pos_enc_config=PositionalEncodingConfig(embed_dim=128, max_len=5000, dropout=0.0),
        block_config=TransformerBlockConfig(
            linear_config=TransformerLinearConfig(input_dim=512, ff_dim=2048, output_dim=512, dropout=0.0),
            mhsa_config=TransformerMHSAConfig(input_dim=512, num_heads=8, dropout=0.0),
        ),
        dropout=0.0,
    )
    checkpoint = PtCheckpoint(
        tk.Path(f"{WORD_LM_WORK}/i6_core/returnn/training/ReturnnTrainingJob.UCMntNBI6vDq/output/models/epoch.030.pt")
    )
    return export_model_stateless(model_config=cfg, checkpoint=checkpoint)


def get_gpu_word_transformer_params(lexicons, prefix):
    """Full-history GPU LM, gated on UNK mapping and cached-model equivalence."""
    from .librispeech.lm import TransformerLmParams

    model = get_original_word_transformer_stateless_model()
    check = ValidateOovNeuralLmJob(
        model,
        WORD_LM_VOCAB,
        tuple(lexicons),
        use_kv_cache=False,
        reference_model=WORD_LM_CACHED_MODEL,
    )
    check.add_alias(f"{prefix}/word-trafo-gpu-stateless-validation")
    tk.register_output(f"{prefix}/word-trafo-gpu-stateless-validation", check.out_report)
    tk.register_output(f"{prefix}/word-trafo-gpu-stateless-model", model)
    return TransformerLmParams(onnx_model=model, vocab_file=check.out_vocab, use_gpu=True, use_kv_cache=False)


def check_unknown_mapping(lexicons, vocab, unknown_token):
    """Check lexical UNK and the explicit word-LM indices (not acoustic ids)."""
    indices, size = _read_lm_index_vocabulary(vocab)
    for token in (unknown_token, "<s>", "</s>", "THE"):
        if token not in indices:
            raise ValueError(f"Required LM token {token!r} is absent from the vocabulary")
    for path in lexicons:
        unknown = [l for l in read_lexicon(path).findall("lemma") if l.get("special") == "unknown"]
        if len(unknown) != 1 or [t.text for t in unknown[0].findall("synt/tok")] != [unknown_token]:
            raise ValueError(f"Lexicon {path} must have exactly one unknown lemma with synt={unknown_token!r}")
    return indices, size


def probe_unknown_scores(session, indices, vocab_size, unknown_token, use_kv_cache=True, reference_session=None):
    """Check NLL normalization/UNK outputs and feeding UNK back as a history id.

    Check either interface on full prefixes. Optionally compare every output
    column to the original cached export, including a longer prefix. This is
    a finite smoke test, not a proof for all histories.
    """
    inputs = {x.name: x for x in session.get_inputs()}
    integer_types = {"tensor(int32)": np.int32, "tensor(int64)": np.int64}
    cache_names = [name for name in inputs if re.fullmatch(r"state_l\d+_[kv]_in", name)]
    if use_kv_cache and not cache_names:
        raise ValueError("Expected the KV-cached word Transformer LM")
    allowed = {"tokens", "tokens:size1"}
    if use_kv_cache:
        allowed.update({"state_l000_k_in:size1", *cache_names})
    if set(inputs) != allowed:
        raise ValueError(f"Unexpected ONNX input interface: {set(inputs)}")
    reports = []
    histories = [("<s>",), ("<s>", unknown_token), ("<s>", "THE")]
    if reference_session is not None:
        histories.append(("<s>",) + ("THE", unknown_token) * 8)
    for history in histories:
        feed = {
            "tokens": np.asarray([[indices[t] for t in history]], dtype=integer_types[inputs["tokens"].type]),
            "tokens:size1": np.asarray([len(history)], dtype=integer_types[inputs["tokens:size1"].type]),
        }
        if use_kv_cache:
            feed["state_l000_k_in:size1"] = np.asarray([0], dtype=integer_types[inputs["state_l000_k_in:size1"].type])
        for name in cache_names:
            shape = inputs[name].shape
            if len(shape) != 3 or not isinstance(shape[-1], int) or inputs[name].type != "tensor(float)":
                raise ValueError(f"Unsupported cache shape/type: {name}, {shape}, {inputs[name].type}")
            feed[name] = np.zeros((1, 0, shape[-1]), dtype=np.float32)
        scores = session.run(["scores"], feed)[0]
        expected_shape = (1, len(history), vocab_size) if use_kv_cache else (1, vocab_size)
        if scores.shape != expected_shape:
            raise ValueError(f"LM output {scores.shape} does not match the word vocabulary ({vocab_size})")
        costs = (scores[0, -1] if use_kv_cache else scores[0]).astype(np.float64)
        if not np.isfinite(costs).all() or costs.min() < -1e-5:
            raise ValueError("Expected finite negative-log probabilities, not logits or log probabilities")
        mass = float(np.exp(-costs).sum())
        if not np.isclose(mass, 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError(f"LM score normalization failed: probability mass={mass}")
        unk_cost = float(costs[indices[unknown_token]])
        unk_probability = float(np.exp(-unk_cost))
        if unk_probability <= 0:
            raise ValueError(f"No usable probability for {unknown_token!r} after {history}")
        reports.append(
            dict(
                history=list(history), unknown_cost=unk_cost, unknown_probability=unk_probability, probability_mass=mass
            )
        )
        if reference_session is not None:
            ref_inputs = {x.name: x for x in reference_session.get_inputs()}
            ref_feed = {
                "tokens": feed["tokens"].astype(integer_types[ref_inputs["tokens"].type]),
                "tokens:size1": feed["tokens:size1"].astype(integer_types[ref_inputs["tokens:size1"].type]),
                "state_l000_k_in:size1": np.asarray([0], dtype=integer_types[ref_inputs["state_l000_k_in:size1"].type]),
            }
            for name, spec in ref_inputs.items():
                if re.fullmatch(r"state_l\d+_[kv]_in", name):
                    ref_feed[name] = np.zeros((1, 0, spec.shape[-1]), dtype=np.float32)
            reference = reference_session.run(["scores"], ref_feed)[0]
            if reference.shape != (1, len(history), vocab_size):
                raise ValueError(f"Incorrect cached reference output shape: {reference.shape}")
            ref_costs = reference[0, -1].astype(np.float64)
            difference = float(np.max(np.abs(costs - ref_costs)))
            if not np.isfinite(ref_costs).all() or not np.allclose(costs, ref_costs, atol=2e-3, rtol=1e-5):
                raise ValueError(f"Stateless/cached word LM mismatch after {history}: max_abs_error={difference}")
            reports[-1]["reference_max_abs_error"] = difference
    return reports


class ValidateOovNeuralLmJob(Job):
    """Release an unchanged vocabulary only after the actual ONNX probe passes."""

    __sis_hash_exclude__ = {"use_kv_cache": True, "reference_model": None}

    def __init__(
        self,
        onnx_model: tk.Path,
        vocab: tk.Path,
        lexicons: tuple[tk.Path, ...],
        unknown_token="<UNK>",
        use_kv_cache=True,
        reference_model=None,
    ):
        self.onnx_model, self.vocab, self.lexicons = onnx_model, vocab, lexicons
        self.unknown_token = unknown_token
        self.use_kv_cache, self.reference_model = use_kv_cache, reference_model
        self.out_vocab = self.output_path("lm.vocab.txt")
        self.out_report = self.output_path("validation.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 1})

    def run(self):
        import onnxruntime as ort

        indices, size = check_unknown_mapping(self.lexicons, self.vocab, self.unknown_token)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        session = ort.InferenceSession(
            self.onnx_model.get_path(), sess_options=options, providers=["CPUExecutionProvider"]
        )
        reference = None
        if self.reference_model is not None:
            reference = ort.InferenceSession(
                self.reference_model.get_path(), sess_options=options, providers=["CPUExecutionProvider"]
            )
        probes = probe_unknown_scores(
            session, indices, size, self.unknown_token, use_kv_cache=self.use_kv_cache, reference_session=reference
        )
        with open(self.out_report.get_path(), "w") as f:
            json.dump(
                dict(
                    passed=True,
                    vocab_size=size,
                    unknown_token=self.unknown_token,
                    unknown_output_index=indices[self.unknown_token],
                    unknown_history_input_index=indices[self.unknown_token],
                    bos_index=indices["<s>"],
                    eos_index=indices["</s>"],
                    probes=probes,
                ),
                f,
                indent=2,
            )
        shutil.copyfile(self.vocab.get_path(), self.out_vocab.get_path())
