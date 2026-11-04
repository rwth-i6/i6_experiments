"""Small real Job.run regressions for OOV data and LM adaptation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sisyphus import tk
from i6_core.util import uopen

from i6_experiments.users.berger.data.librispeech.lexicon_oov import (
    BoostArpaUnknownProbabilityJob,
    CollapseOovScoresInOnnxJob,
    CountLexiconOovJob,
    NormalizeArpaUnknownTokenJob,
    ReduceLexiconToTargetOovJob,
)


@pytest.mark.parametrize("factor", [0.25, 1.0, 4.0])
def test_arpa_factor_multiplies_probability_and_only_changes_predicted_unk(tmp_path, factor):
    source = tmp_path / "source.arpa"
    source.write_text(
        "\\data\\\nngram 1=3\nngram 2=3\n\\1-grams:\n"
        "-2 <UNK> -0.7\n-1 THE -0.8\n-3 2012\n\\2-grams:\n"
        "-2 THE <UNK> -0.4\n-3 <UNK> 2012\n-4 <UNK> THE\n\\end\\\n"
    )
    job = BoostArpaUnknownProbabilityJob(tk.Path(str(source)), factor)
    job.out_arpa_lm = tk.Path(str(tmp_path / "result.gz"))
    job.run()
    with uopen(job.out_arpa_lm, "rt") as stream:
        text = stream.read()
    unigrams = text.split("\\1-grams:\n")[1].split("\\2-grams:")[0].splitlines()
    bigrams = text.split("\\2-grams:\n")[1].split("\\end\\")[0].splitlines()
    for row in (unigrams[0], bigrams[0]):
        assert 10 ** float(row.split()[0]) == pytest.approx(0.01 * factor, rel=1e-6)
    assert unigrams[0].split()[2] == "-0.7"  # backoff is not the predicted label
    assert bigrams[0].split()[3] == "-0.4"
    assert unigrams[1:] == ["-1 THE -0.8", "-3 2012"]
    assert bigrams[1:] == ["-3 <UNK> 2012", "-4 <UNK> THE"]


def test_arpa_unknown_case_is_explicit_and_missing_token_fails(tmp_path):
    source = tmp_path / "source.arpa"
    source.write_text("\\data\\\nngram 1=1\n\\1-grams:\n-2 <unk>\n\\end\\\n")
    boost = BoostArpaUnknownProbabilityJob(tk.Path(str(source)), 2)
    boost.out_arpa_lm = tk.Path(str(tmp_path / "boost.gz"))
    with pytest.raises(ValueError, match="no n-gram"):
        boost.run()
    rename = NormalizeArpaUnknownTokenJob(tk.Path(str(source)))
    rename.out_arpa_lm = tk.Path(str(tmp_path / "renamed.gz"))
    rename.run()
    boost.arpa_lm = rename.out_arpa_lm
    boost.run()
    with uopen(boost.out_arpa_lm, "rt") as stream:
        assert "<UNK>" in stream.read()


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_arpa_factor_rejects_invalid_values(factor):
    with pytest.raises(ValueError, match="finite and positive"):
        BoostArpaUnknownProbabilityJob(tk.Path("unused"), factor)


@pytest.mark.parametrize("opset,shape", [(17, (3, 5)), (18, (2, 3, 5))])
def test_onnx_collapse_preserves_mass_known_columns_and_history_indices(tmp_path, opset, shape):
    import onnx
    import onnxruntime as ort
    from onnx import helper, TensorProto

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input_costs"], ["scores"]),
            helper.make_node("Identity", ["tokens"], ["seen_history"]),
        ],
        "tiny-word-lm",
        [
            helper.make_tensor_value_info("input_costs", TensorProto.FLOAT, list(shape)),
            helper.make_tensor_value_info("tokens", TensorProto.INT64, [None, None]),
        ],
        [
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, list(shape)),
            helper.make_tensor_value_info("seen_history", TensorProto.INT64, [None, None]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)
    source = tmp_path / "source.onnx"
    onnx.save(model, source)
    vocab, removed = tmp_path / "vocab", tmp_path / "removed"
    vocab.write_text("<UNK> 2\nTHE 0\nREMOVED_A 3\nREMOVED_B 1\n<s> 4\n</s> 4\n")
    removed.write_text("1\tREMOVED_A REMOVED_B\n")
    job = CollapseOovScoresInOnnxJob(tk.Path(str(source)), tk.Path(str(vocab)), tk.Path(str(removed)))
    job.out_onnx_model, job.out_report, job.out_removed_indices = [
        tk.Path(str(tmp_path / name)) for name in ("adapted.onnx", "report.json", "indices")
    ]
    job.run()
    probabilities = np.random.default_rng(3).uniform(0.01, 1, shape).astype("float32")
    probabilities /= probabilities.sum(-1, keepdims=True)
    costs = -np.log(probabilities)
    history = np.array([[4, 0, 2]], dtype="int64")
    session = ort.InferenceSession(job.out_onnx_model.get_path(), providers=["CPUExecutionProvider"])
    adapted, seen = session.run(None, {"input_costs": costs, "tokens": history})
    np.testing.assert_array_equal(seen, history)
    np.testing.assert_allclose(adapted[..., [0, 4]], costs[..., [0, 4]], atol=1e-6)
    np.testing.assert_allclose(np.exp(-adapted[..., 2]), probabilities[..., [1, 2, 3]].sum(-1), rtol=1e-6)
    np.testing.assert_allclose(np.exp(-adapted).sum(-1), 1, atol=1e-6)
    assert np.all(adapted[..., [1, 3]] > 1e20)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["unknown_index"] == 2 and report["num_removed_indices"] == 2
    with pytest.raises(ValueError, match="missing from"):
        bad = CollapseOovScoresInOnnxJob(tk.Path(str(source)), tk.Path(str(vocab)), tk.Path(str(removed)), "<unk>")
        bad.run()


def corpus_and_lexicon(tmp_path):
    from i6_core.lib.lexicon import Lexicon, Lemma
    from i6_core.util import write_xml

    corpus = tmp_path / "corpus.xml"
    corpus.write_text(
        '<corpus name="dev"><recording name="r" audio="unused.wav">'
        '<segment name="s" start="0" end="1"><orth>A A A B C OOV</orth></segment>'
        "</recording></corpus>"
    )
    lexicon = Lexicon()
    lexicon.add_phoneme("p", variation="none")
    for word in ("A", "B", "C"):
        lexicon.add_lemma(Lemma(orth=[word], phon=["p"], synt=[word]))
    lexicon.add_lemma(Lemma(orth=["[UNK]"], special="unknown", synt=["<UNK>"]))
    path = tmp_path / "lex.xml"
    write_xml(str(path), lexicon.to_xml())
    return tk.Path(str(corpus)), tk.Path(str(path))


def redirect_outputs(job, tmp_path):
    for key, value in vars(job).copy().items():
        if key.startswith("out_"):
            if isinstance(value, tk.Path):
                setattr(job, key, tk.Path(str(tmp_path / key)))
            else:
                setattr(job, key, SimpleNamespace(set=lambda value: None))


def test_count_and_reduction_use_token_frequency_and_preserve_specials(tmp_path):
    from i6_core.lib.lexicon import Lexicon

    corpus, lexicon = corpus_and_lexicon(tmp_path)
    count = CountLexiconOovJob([corpus], lexicon)
    redirect_outputs(count, tmp_path)
    count.run()
    baseline = json.loads((tmp_path / "out_report").read_text())
    assert baseline["num_tokens"] == 6 and baseline["num_oov_tokens"] == 1
    assert baseline["token_oov_rate"] == pytest.approx(1 / 6)
    frequencies = tmp_path / "training.tsv"
    frequencies.write_text("100\tA\n2\tB\n1\tC\n")
    reduce = ReduceLexiconToTargetOovJob(lexicon, [corpus], tk.Path(str(frequencies)), 0.5)
    redirect_outputs(reduce, tmp_path)
    reduce.run()
    reduced = Lexicon()
    reduced.load(reduce.out_lexicon.get_path())
    assert [lemma.orth for lemma in reduced.lemmata] == [["A"], ["[UNK]"]]
    report = json.loads((tmp_path / "out_report").read_text())
    assert report["num_oov_tokens"] == 3 and report["achieved_token_oov_rate"] == 0.5
    assert "1\tC" in (tmp_path / "out_removed_lemmas").read_text()
    assert reduced.lemmata[-1].special == "unknown"
    with pytest.raises(ValueError, match="below the baseline"):
        invalid = ReduceLexiconToTargetOovJob(lexicon, [corpus], tk.Path(str(frequencies)), 0)
        invalid.run()
