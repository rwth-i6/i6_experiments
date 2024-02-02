from typing import Union
from sisyphus import tk
from i6_core.corpus import FilterCorpusRemoveUnknownWordSegmentsJob
from i6_experiments.users.berger.helpers import SeparatedCorpusObject
from i6_core.meta.system import CorpusObject


def filter_unk_in_corpus_object(corpus_object: Union[CorpusObject, SeparatedCorpusObject], lexicon: tk.Path) -> None:
    if isinstance(corpus_object, CorpusObject):
        assert corpus_object.corpus_file is not None
        corpus_object.corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=corpus_object.corpus_file,
            bliss_lexicon=lexicon,
            all_unknown=False,
        ).out_corpus
    elif isinstance(corpus_object, SeparatedCorpusObject):
        corpus_object.primary_corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=corpus_object.primary_corpus_file,
            bliss_lexicon=lexicon,
            all_unknown=False,
        ).out_corpus
        corpus_object.secondary_corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=corpus_object.secondary_corpus_file,
            bliss_lexicon=lexicon,
            all_unknown=False,
        ).out_corpus
        corpus_object.mix_corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=corpus_object.mix_corpus_file,
            bliss_lexicon=lexicon,
            all_unknown=False,
        ).out_corpus
    else:
        raise NotImplementedError
