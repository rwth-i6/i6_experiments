"""Dynamic BPE word scoring: token ids, teacher forcing and development-only wiring."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.wordwise_transformer import (
    BpeWordwiseGateJob,
    WordwiseTransformerBackend,
    read_index_vocabulary,
)


def test_vocabulary_aliases_and_dense_indices(tmp_path):
    word = tmp_path / "word.vocab"
    word.write_text("<s> 0\n</s> 0\n<UNK> 1\nHELLO 2\nWORLD 3\n")
    assert read_index_vocabulary(word)["WORLD"] == 3
    bpe = tmp_path / "bpe.vocab"
    bpe.write_text(repr({"<s>": 0, "</s>": 0, "<unk>": 1, "HE@@": 2, "LLO": 3}))
    assert read_index_vocabulary(bpe, bpe=True)["HE@@"] == 2
    word.write_text("</s> 0\n<UNK> 1\nHELLO 3\n")
    with pytest.raises(ValueError, match="dense"):
        read_index_vocabulary(word)


def test_word_and_bpe_retokenization_are_independent_of_am_ids():
    word = WordwiseTransformerBackend.__new__(WordwiseTransformerBackend)
    word.kind, word.vocab, word.unknown = "word", {"HELLO": 2, "WORLD": 3}, 1
    assert word.tokenize(["hello", " WORLD", "UNSEEN"]) == [[2], [3], [1]]
    with pytest.raises(ValueError, match="one completed"):
        word.tokenize(["HELLO WORLD"])
    bpe = WordwiseTransformerBackend.__new__(WordwiseTransformerBackend)
    bpe.kind, bpe.vocab, bpe.unknown = "bpe", {"HE@@": 4, "LLO": 2}, 1
    surfaces = []
    bpe.encoder = SimpleNamespace(segment_sentence=lambda s: surfaces.append(s) or ["HE@@", "LLO"])
    assert bpe.tokenize([" hello"]) == [[4, 2]]
    assert surfaces == ["HELLO"]


def test_last_hidden_projection_matches_full_sequence():
    import torch
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.transformer_lm.pytorch_modules import (
        TransformerLm,
    )

    # Exercise the production forward with a small deterministic network.
    model = TransformerLm.__new__(TransformerLm)
    torch.nn.Module.__init__(model)
    model.embed = torch.nn.Embedding(11, 8)
    model.positional_encoding = model.input_linear = torch.nn.Identity()
    model.transformer_blocks = torch.nn.ModuleList()
    model.output_layernorm = torch.nn.LayerNorm(8)
    model.dropout = torch.nn.Dropout(0.0)
    model.output_linear = torch.nn.Linear(8, 11)
    model.causal_mask = torch.ones(5, 5, dtype=torch.bool).tril()
    model.eval()
    tokens = torch.tensor([[0, 2, 3, 4, 5], [0, 6, 7, 0, 0]])
    lengths = torch.tensor([5, 3])
    mask = torch.arange(5)[None] < lengths[:, None]
    projected_shapes = []
    handle = model.output_linear.register_forward_pre_hook(lambda _, args: projected_shapes.append(args[0].shape))
    with torch.inference_mode():
        full = model(tokens, mask)
        last = model(tokens, mask, last_positions=lengths - 1)
    handle.remove()
    torch.testing.assert_close(last[:, 0], full[torch.arange(2), lengths - 1])
    assert projected_shapes == [torch.Size([5, 2, 8]), torch.Size([1, 2, 8])]


@pytest.mark.parametrize("batch_size", [1, 8])
def test_teacher_forcing_and_eos_accounting(batch_size):
    backend = WordwiseTransformerBackend.__new__(WordwiseTransformerBackend)
    backend.vocab, backend.checked, backend.batch_size = {"</s>": 0, "<unk>": 1, "A": 2, "B": 3}, False, batch_size
    seen = []

    def forward(outputs, inputs):
        assert outputs == ["scores"]
        tokens, lengths = inputs["tokens"], inputs["tokens:size1"]
        assert tokens.dtype == lengths.dtype == np.int32
        costs = []
        for row, length in zip(tokens, lengths):
            prefix = row[:length].tolist()
            seen.append(prefix)
            probabilities = np.roll(np.array([0.1, 0.2, 0.3, 0.4]), prefix[-1])
            costs.append(-np.log(probabilities))
        return [np.array(costs)]

    backend.session = SimpleNamespace(run=forward)
    requests = [SimpleNamespace(prefix=[0], tokens=[2, 3]), SimpleNamespace(prefix=[0, 2, 3], tokens=[0])]
    costs = backend.score(requests)
    assert seen == [[0], [0, 2], [0, 2, 3]]
    assert costs[0] == pytest.approx([-np.log(0.3), -np.log(0.2)])
    assert costs[1] == pytest.approx([-np.log(0.2)])
    assert backend.checked
    assert backend.initial_tokens() == backend.sentence_end_tokens() == [0]


def test_ctc_rna_native_gate(tmp_path, monkeypatch):
    binary = Path(
        "/work/asr4/berger/sisyphus_work_dirs/librispeech/20241106_seq2seq_template_setups/i6_core/tools/compile/CMakeJob.f37Fc6lWurlz/output/install"
    )
    if not binary.exists():
        pytest.skip("Pinned native RASR build unavailable")
    job = BpeWordwiseGateJob(tk.Path(str(binary)), code_version="test")
    job.out_report, job.out_log = tk.Path(str(tmp_path / "validation.json")), tk.Path(str(tmp_path / "rasr.log"))
    monkeypatch.chdir(tmp_path)
    job.run()
    assert '"passed": true' in (tmp_path / "validation.json").read_text()


def test_development_matrix_and_registered_jobs(monkeypatch):
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments import wordwise_lm
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import recog, report

    calls = []
    original = recog.recog_rasr_offline

    def capture(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(recog, "recog_rasr_offline", capture)
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import recognition as ls_recognition
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import recognition as loq_recognition

    correction_calls = []

    def capture_correction(original, field):
        def wrapped(*args, **kwargs):
            variant = kwargs["variant"]
            correction_calls.append((field, variant.descriptor, getattr(variant, field)))
            return original(*args, **kwargs)
        return wrapped

    for package in (ls_recognition, loq_recognition):
        for module, helper, field in (
            (package.ctc_bpe, "_get_model_serializers", "prior_scale"),
            (package.ffnn_transducer_bpe, "_get_label_scorer_configs", "ilm_scale"),
        ):
            monkeypatch.setattr(module, helper, capture_correction(getattr(module, helper), field))
    reports = []
    monkeypatch.setattr(report, "register_recog_report", lambda results, **kwargs: reports.append((results, kwargs)))
    results = wordwise_lm.run()
    expected = 4 * (
        len(wordwise_lm.conditions()) + len(wordwise_lm.corrected_conditions())
        + len(wordwise_lm.unknown_penalty_conditions())
    )
    assert len(results) == len(calls) == expected == 116
    assert {row.corpus_name for row in results} == {"dev-other", "dev.short"}
    assert len({(row.descriptor, row.corpus_name) for row in results}) == expected
    assert reports[0][1] == {"filename": "summary"}
    assert len(correction_calls) == expected
    assert sum(scale == 0.2 for _, _, scale in correction_calls) == 92
    for field, descriptor, scale in correction_calls:
        assert scale == (0.2 if f"-{'prior' if field == 'prior_scale' else 'ilm'}0.2" in descriptor else 0.0)
    for args in calls:
        config = args["recog_rasr_config_file"].creator.config.lib_rasr.search_algorithm
        assert config.log_stepwise_statistics is False
        assert args["normalize_hf_score"] is True
        assert args["checkpoint"] is not None
        assert (
            args["rasr_binary_path_override"].creator.source_folder.creator.commit
            == "d000386c08a9b9213238e3cafdeb4291dc0b1775"
        )
        assert config.collapse_repeated_labels == args["descriptor"].startswith("ctc_bpe")
        code = args["encoder_serializers"].get()
        if "-prior0.2" in args["descriptor"]:
            assert "prior_scale=0.2" in code
            assert "prior_file=" in code
        if config.type == "llm-timesync-beam-search":
            assert config.word_piece_convention == "continuation-marked"
            assert config.word_piece_marker == "@@"
            wide = "-wide-beam1024-thr16" in args["descriptor"]
            assert config.max_beam_size == ([1024] if wide else [32])
            assert config.pre_llm_max_beam_size == (1024 if wide else 128)
            assert config.score_threshold == ([16.0] if wide else [8.0])
            assert config.pre_llm_score_threshold == (16.0 if wide else 8.0)
            if wide:
                assert config.llm_scale == 0.8
                assert config.llm.kind == "word"
                assert args["descriptor"].endswith("-ilm0.2")
            assert args["rasr_init_hook"] is not None
            code = args["encoder_serializers"].get()
            penalized = "-unk-beta" in args["descriptor"]
            hook = "RegisterWordwiseUnknownPenalty" if penalized else "RegisterWordwiseTransformer"
            assert f"wordwise_lm_init_hook = {hook}" in code
            if penalized:
                assert config.llm.type == "wordwise-transformer-unk-penalty"
                assert config.llm.kind == "word"
                assert config.llm.outer_lm_scale == config.llm_scale
                assert config.llm.unknown_word_penalty in {1.0, 2.0, 4.0, 8.0, 12.0}
            if config.llm.type == "wordwise-qwen":
                assert "WordwiseOnnxRuntimeJob" not in code
                repo = config.llm.model.creator
                assert (repo.repo_id, repo.revision) in wordwise_lm.QWEN_MODELS.values()
            else:
                assert config.llm.kind in {"bpe", "word"}
                assert "WordwiseOnnxRuntimeJob" not in code
        else:
            assert "WordwiseOnnxRuntimeJob" not in args["encoder_serializers"].get()


@pytest.mark.parametrize("scale", [0.4, 0.8])
def test_unknown_word_penalty_is_outside_lm_scale(monkeypatch, scale):
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.wordwise_unknown import (
        UnknownPenaltyWordwiseBackend,
    )

    def init(self, config):
        self.kind, self.unknown = "word", 1
        self.session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

    monkeypatch.setattr(WordwiseTransformerBackend, "__init__", init)
    monkeypatch.setattr(WordwiseTransformerBackend, "score", lambda self, requests: [[3.0, 5.0, 7.0], [2.0]])
    backend = UnknownPenaltyWordwiseBackend({"unknown-word-penalty": 4.0, "outer-lm-scale": scale})
    requests = [SimpleNamespace(prefix=[0], tokens=[2, 1, 0]), SimpleNamespace(prefix=[0, 1], tokens=[1])]
    costs = backend.score(requests)
    # Known token and EOS unchanged. Every UNK occurrence pays beta once,
    # including after an earlier UNK; the history itself is not penalized.
    assert costs[0] == pytest.approx([3.0, 5.0 + 4.0 / scale, 7.0])
    assert costs[1] == pytest.approx([2.0 + 4.0 / scale])
    assert scale * costs[0][1] - scale * 5.0 == pytest.approx(4.0)
