"""
This is about Sisyphus :func:`sis_hash_helper`,
which is the core function to define hashes for any objects,
which defines all the hashes of the Sisyphus recipes here.

There are a number of more complex objects where the hashing might break.
Here we want to test all those, including maybe some more basic types.

Note: When adding a new test, it's totally ok to just put whatever :func:`sis_hash_helper`
currently returns. The test will show if the hash changes in the future.

See also:
    sisyphus/tests/hash_unittest.py
"""


from sisyphus.hash import sis_hash_helper


def test_int():
    assert sis_hash_helper(42) == b"(int, 42)"


def test_tuple_deep():
    assert sis_hash_helper((1, 2, (3, 4, (5, 6)))) == (
        b"(tuple, (int, 1), (int, 2), (tuple, (int, 3), (int, 4), (tuple, (int, 5), (int, 6))))"
    )


def test_datasets_librispeech_ogg_zip():
    from .datasets.librispeech import LibrispeechOggZip

    ds = LibrispeechOggZip()
    assert sis_hash_helper(ds) == (
        b"(LibrispeechOggZip, (dict, (tuple, (str, 'audio'), (NoneType)), (tuple, (str"
        b", 'audio_dim'), (NoneType)), (tuple, (str, 'eval_subset'), (int, 3000)), (tu"
        b"ple, (str, 'main_key'), (NoneType)), (tuple, (str, 'train_audio_preprocess')"
        b", (function, (tuple, (str, 'i6_experiments.users.zeyer.speed_pert.librosa_09"
        b"_10_11_kaiser_fast'), (str, 'speed_pert_librosa_09_10_11_kaiser_fast')))), ("
        b"tuple, (str, 'train_audio_random_permute'), (bool, False)), (tuple, (str, 't"
        b"rain_epoch_split'), (int, 20)), (tuple, (str, 'train_epoch_wise_filter'), (d"
        b"ict, (tuple, (tuple, (int, 1), (int, 5)), (dict, (tuple, (str, 'max_mean_len"
        b"'), (int, 1000)))))), (tuple, (str, 'train_sort_laplace_num_seqs'), (int, 10"
        b"00)), (tuple, (str, 'vocab'), (NoneType)), (tuple, (str, 'with_eos_postfix')"
        b", (bool, False))))"
    )


def test_datasets_librispeech_lm():
    from .datasets.librispeech import LibrispeechLmDataset
    from .datasets.utils.bytes import Utf8BytesVocab

    ds = LibrispeechLmDataset(vocab=Utf8BytesVocab())
    assert sis_hash_helper(ds) == (
        b"(LibrispeechLmDataset, (dict, (tuple, (str, 'eval_subset'), (int, 3000)), (t"
        b"uple, (str, 'main_key'), (NoneType)), (tuple, (str, 'train_epoch_split'), (i"
        b"nt, 20)), (tuple, (str, 'train_sort_laplace_num_seqs'), (int, 1000)), (tuple"
        b", (str, 'vocab'), (Utf8BytesVocab, (dict, (tuple, (str, 'dim'), (int, 256)),"
        b" (tuple, (str, 'opts'), (dict)))))))"
    )


def _func_ref(a: int, b: int) -> int:
    return a + b


def test_func_ref():
    assert (
        sis_hash_helper(_func_ref)
        == b"(function, (tuple, (str, 'i6_experiments.users.zeyer.sis_test_hashs'), (str, '_func_ref')))"
    )


def test_functools_partial():
    from functools import partial

    obj = partial(int, 42)
    assert sis_hash_helper(obj) == (
        b"(partial, (dict,"
        b" (tuple, (str, 'args'), (tuple, (int, 42))),"
        b" (tuple, (str, 'func'), (type,"
        b" (tuple, (str, 'builtins'), (str, 'int')))),"
        b" (tuple, (str, 'keywords'), (dict))))"
    )


def test_Dim():
    from returnn.tensor import Dim

    obj = Dim(None, name="time")
    assert sis_hash_helper(obj) == b"(Dim, (dict, (tuple, (str, 'dim'), (NoneType))))"

    obj = Dim(None, name="time", kind=Dim.Types.Spatial)
    assert (
        sis_hash_helper(obj)
        == b"(Dim, (dict, (tuple, (str, 'dim'), (NoneType)), (tuple, (str, 'kind'), (str, 'spatial'))))"
    )

    obj = Dim(42, name="vocab")
    assert sis_hash_helper(obj) == b"(Dim, (dict, (tuple, (str, 'dim'), (int, 42))))"


def test_PiecewiseLinear():
    from returnn.util.math import PiecewiseLinear

    obj = PiecewiseLinear({1: 1.5, 5: 2.5})
    assert str(obj) == "PiecewiseLinear({1: 1.5, 5: 2.5})"
    assert sis_hash_helper(obj) == (
        b"(PiecewiseLinear, (dict,"
        b" (tuple, (str, 'values'), (dict, (tuple, (int, 1), (float, 1.5)), (tuple, (int, 5), (float, 2.5))))))"
    )
