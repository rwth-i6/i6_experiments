"""Graph-level tests of ``lm/word_lm.py`` and ``lm/hlg.py`` (no job runs): the null's seed is one
constant in both the in-house and the official code path, the child's ``PYTHONPATH`` root, and the
three graph kinds of ``get_hlg`` with their banked names, ladders, budgets and expected builds."""

import os

import pytest

import i6_experiments
from i6_experiments.users.wu.experiments.unsupervised_asr.lm import hlg, lexlat_k2_official, word_lm


def test_the_two_nulls_share_one_seed():
    assert word_lm.DERANGEMENT_SEED == lexlat_k2_official.DERANGEMENT_SEED == 0


def test_child_pythonpath_root_holds_i6_experiments():
    root = hlg._src_root()
    assert os.path.samefile(os.path.join(root, "i6_experiments"),
                            os.path.dirname(i6_experiments.__file__))
    assert hlg._OFFICIAL_CHILD == lexlat_k2_official.__name__


@pytest.mark.parametrize(
    "kind, shuffled, name, ladder, time_rqmt, theta",
    [
        ("inhouse_3gram", False, "lexlat_20/hlg_word_boundary", (0.0, 0.5, 2.0), 6.0, 0.0),
        ("inhouse_3gram", True, "k2shuf_20/hlg_word_boundary", (0.0, 0.5, 2.0), 6.0, 0.0),
        ("official_4gram", False, "off4g_word_boundary/hlg",
         (0.0, 0.5, 2.0, 5.0, 8.0, 12.0, 16.0), 8.0, 5.0),
        ("official_4gram", True, "off4g_word_boundary/hlg_shuffled",
         (0.0, 0.5, 2.0, 5.0, 8.0, 12.0, 16.0), 8.0, 5.0),
        ("official_3gram_1e7", False, "off3g_1e7_word_boundary/hlg", (0.0, 0.5, 2.0), 6.0, 0.0),
    ],
)
def test_get_hlg(kind, shuffled, name, ladder, time_rqmt, theta):
    out = hlg.get_hlg(kind, shuffled=shuffled)
    assert sorted(out) == ["expected_build", "hlg", "resources", "stats"]
    job = out["hlg"].creator
    assert job.name == name and job.prune_ladder == ladder and job.shuffled == shuffled
    assert job.backoff_loops == "word_boundary" and job.escape is True and job.sil_prob == 0.5
    assert (job.d_min, job.recognizer_stride) == (2, 3)
    assert job.rqmt == {"cpu": 16, "mem": 200.0, "time": time_rqmt}
    assert out["expected_build"] == {"backoff_loops": "word_boundary", "escape": True,
                                     "sil_prob": 0.5, "theta": theta}
    res = out["resources"].creator
    if kind == "inhouse_3gram":
        assert isinstance(res, word_lm.LexiconTrieBuildJob)
    else:
        assert isinstance(res, hlg.LexlatOfficialResourcesJob)
        assert job.escape_resources.creator is hlg.get_hlg("inhouse_3gram")["resources"].creator
        # the null prices with the TREATMENT's resources
        assert res.build_json.creator.shuffled is False


def test_get_hlg_refuses():
    with pytest.raises(ValueError):
        hlg.get_hlg("official_3gram_1e7", shuffled=True)
    with pytest.raises(ValueError):
        hlg.get_hlg("official_3gram_3e7")


def test_lexicon_trie_trains_its_word_lm_with_i6_core_jobs():
    """The word trigram is i6_core's ``KenLMplzJob`` + ``CreateBinaryLMJob`` on the window replay's
    word lines, with the arguments that issue the source's ``lmplz`` / ``build_binary`` command lines."""
    from i6_core.lm.kenlm import CreateBinaryLMJob, KenLMplzJob

    from i6_experiments.users.wu.experiments.unsupervised_asr import default_tools
    from i6_experiments.users.wu.experiments.unsupervised_asr.lm import prior_gap, word_window

    trie = word_lm.get_lexicon_trie()
    window = trie.window_words.creator
    assert isinstance(window, word_window.WordWindowReplayJob)
    assert trie.window_stats.creator is window
    assert (window.n_window_lines, window.held_stride, window.sample_seed) == (1_010_000, 101, 0)
    assert window.bliss_lexicon.get_path() == trie.bliss_lexicon.get_path()
    lmplz = trie.word_lm_arpa.creator
    assert isinstance(lmplz, KenLMplzJob)
    assert [t.get_path() for t in lmplz.text] == [window.out_words.get_path()]
    assert (lmplz.order, lmplz.interpolate_unigrams, lmplz.pruning, lmplz.vocabulary) == (3, True, None, None)
    assert lmplz.discount_fallback == list(prior_gap.DISCOUNT_FALLBACK) == [0.5, 1.0, 1.5]
    assert lmplz.rqmt["mem"] == word_lm.LMPLZ_MEM_GB == 24  # lmplz -S 24G, the source's call
    binary = trie.word_lm_binary.creator
    assert isinstance(binary, CreateBinaryLMJob)
    assert binary.arpa_lm.get_path() == lmplz.out_lm.get_path()
    kenlm = default_tools.get_kenlm_binary_path().get_path()
    assert lmplz.kenlm_binary_folder.get_path() == binary.kenlm_binary_folder.get_path() == kenlm
    assert trie.word_lm_order == lmplz.order == word_lm.WORD_LM_ORDER
