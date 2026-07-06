from sisyphus import tk, Job

from functools import cache

from dataclasses import dataclass

from i6_core.text import HeadJob
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon, get_g2p_augmented_bliss_lexicon_dict


@dataclass(frozen=True)
class CorpusSetupResult:
    corpus: tk.Path
    lexicon: tk.Path
    segments: tk.Path | None = None

@cache
def setup_corpus(key="train-clean-100") -> CorpusSetupResult:
    corpora = get_bliss_corpus_dict(audio_format="ogg")

    corpus = corpora[key]

    segment_corpus = SegmentCorpusJob(corpus, num_segments=1)
    all_segments = segment_corpus.out_single_segment_files[1]

    if key == "train-clean-100":
        split_segments = ShuffleAndSplitSegmentsJob(
            all_segments,
            split={
                "relevant": 0.1,
                "remainder": 0.9,
            }
        )

        segment_file = split_segments.out_segments["relevant"]
        tk.register_output("datasets/LibriSpeech/segments/train-clean-100-10%.txt", segment_file)

        # Lexicon
        # lex = get_bliss_lexicon(use_stress_marker=False)
        debug_segments = HeadJob(all_segments, num_lines=10, zip_output=False)
        tk.register_output("datasets/LibriSpeech/segments/small_debug.txt", debug_segments.out)

    lexica = get_g2p_augmented_bliss_lexicon_dict()
    lex = lexica[key]


    return CorpusSetupResult(corpus, lex, all_segments)